# pylint: disable=no-member
import pickle
import warnings

import numpy as np
import torch
from myosuite.envs.myo.myochallenge.baoding_v1 import BaodingEnvV1
from sb3_contrib import RecurrentPPO
from sklearn.metrics import classification_report, confusion_matrix
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from envs.environment_factory import EnvironmentFactory
from models.classifier import TaskClassifier

warnings.filterwarnings("ignore")

render = False

env_name = "CustomMyoBaodingBallsP2"
N_OBS_PER_TRIAL = 13

# Model and env for CCW + CW (on which classifier is trained)
load_folder = "trained_models/baoding_phase2/alberto_518/"
PATH_TO_NORMALIZED_BASE_ENV = load_folder + "training_env.pkl"
PATH_TO_BASE_NET = load_folder + "best_model.zip"

# Model and env for hold task
load_folder_hold = "output/training/2022-11-03/23-01-01_final_mixture-hold_from-beta02/"
PATH_TO_NORMALIZED_HOLD_ENV = (
    load_folder_hold + "rl_model_vecnormalize_1120000_steps.pkl"
)
PATH_TO_HOLD_NET = load_folder_hold + "rl_model_1120000_steps.zip"

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    "goal_time_period": [4, 6],  # phase 2: (4, 6)
    # "limit_init_angle": 3.14159,
    # "beta_init_angle": (0.2,0.2),
    "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
}


def load_normalized_envs(task: str):
    if task == "hold":
        env_path = PATH_TO_NORMALIZED_HOLD_ENV
    elif task == "base":
        env_path = PATH_TO_NORMALIZED_BASE_ENV
    else:
        raise ValueError(f"Task {task} not recognized")

    env = EnvironmentFactory.register(env_name, **config)
    envs = DummyVecEnv([lambda: env] * 1)

    return VecNormalize.load(env_path, envs)


def load_model_and_env(task: str):

    if task == "hold":
        model_path = PATH_TO_HOLD_NET
        custom_objects = {
            "learning_rate": lambda _: 1e-4,
            "lr_schedule": lambda _: 1e-4,
            "clip_range": lambda _: 0.2,
        }
    elif task == "base":
        model_path = PATH_TO_BASE_NET
        custom_objects = {
            "learning_rate": lambda _: 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    else:
        raise ValueError(f"Task {task} not recognized")

    envs = load_normalized_envs(task)
    model = RecurrentPPO.load(
        model_path, env=envs, custom_objects=custom_objects, device="cpu"
    )

    return model, envs


class SuperModel:
    def __init__(self):
        # load trained classifier, models and envs
        self.model_base, self.env_base = load_model_and_env("base")
        self.model_hold, self.env_hold = load_model_and_env("hold")

        self.env_base.training = False
        self.env_hold.training = False

        self.classifier = self.load_classifier()
        self.scaler = self.load_data_scaler()

        # initialize variables to track observations for classifier
        self.obs_for_classifier = []
        self.timestep = 0
        self.use_hold_net = False
        self.just_switched = False
        self.current_task = 1

    def load_classifier(self):
        classifier = TaskClassifier(N_OBS_PER_TRIAL)
        classifier.load_state_dict(
            torch.load("./output/classifier/task_classifier_alberto-518.pt")
        )
        return classifier

    def load_data_scaler(self):
        with open("./output/classifier/scaler_alberto-518.pkl", "rb") as f:
            scaler = pickle.load(f)
        return scaler

    @property
    def env(self):
        if self.use_hold_net:
            return self.env_hold
        return self.env_base

    def collect_obs(self, obs) -> None:
        self.obs_for_classifier.append(obs[29:47].copy())

    def update_task(self, observations: torch.Tensor) -> None:
        task_id = torch.round(torch.sigmoid(self.classifier(observations)))
        self.current_task = 0 if task_id == 0 else 1

    def process_before_action(self, obs, episode_start) -> None:
        if episode_start:
            self.use_hold_net = False
            self.just_switched = False
            self.timestep = 0
            self.obs_for_classifier = []

        if self.timestep < 13:
            self.collect_obs(obs)

        if self.timestep == 12:
            observations = np.concatenate(self.obs_for_classifier).reshape((1, -1))
            observations = self.scaler.transform(observations)
            self.update_task(torch.FloatTensor(observations))

            if self.current_task == 0:
                self.use_hold_net = True
                self.just_switched = True

        self.timestep += 1

    def predict(self, obs, state, episode_start, deterministic):

        self.process_before_action(obs, episode_start)
        obs = self.env.normalize_obs(obs)

        if self.use_hold_net:
            if self.just_switched:
                state = None
                self.just_switched = False
            return self.model_hold.predict(obs, state, episode_start, deterministic)

        return self.model_base.predict(obs, state, episode_start, deterministic)


def eval_perf(eval_env: BaodingEnvV1, eval_model: SuperModel):
    # Enjoy trained agent
    perfs, lens, lstm_states, cum_rew, step = [], [], None, 0, 0
    obs = eval_env.reset()
    episode_starts = np.ones((1,), dtype=bool)
    classifier_targets, classifier_preds = [], []
    num_episodes = 0

    for _ in range(2000 * 200):
        if render:
            eval_env.sim.render(mode="window")
        action, lstm_states = eval_model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, _ = eval_env.step(action)

        episode_starts = dones
        cum_rew += rewards
        step += 1

        if step == 20:
            classifier_targets.append([eval_env.which_task.value])
            classifier_preds.append([eval_model.current_task])

        if dones:
            num_episodes += 1
            episode_starts = np.ones((1,), dtype=bool)
            lstm_states = None
            obs = eval_env.reset()
            lens.append(step)
            perfs.append(cum_rew)
            cum_rew, step = 0, 0

            if num_episodes % 10 == 0:
                # print performance metrics
                print(f"\nEpisode {num_episodes}")
                print(
                    f"Average len: {np.mean(lens):.2f} +/- {np.std(lens)/np.sqrt(num_episodes):.2f}"
                )
                print(
                    f"Average rew: {np.mean(perfs):.2f} +/- {np.std(perfs)/np.sqrt(num_episodes):.2f}\n"
                )

            if num_episodes % 100 == 0:
                num_errors = np.sum(
                    abs(
                        np.array(classifier_targets).clip(0, 1)
                        - np.array(classifier_preds)
                    )
                )
                print(
                    f"\nClassifier inaccuracy = {num_errors/len(classifier_targets)*100:.1f}%\n\n"
                )

                print(PATH_TO_BASE_NET, PATH_TO_HOLD_NET)

    # classifier metrics
    classifier_targets = np.array(classifier_targets).clip(0, 1)
    classifier_preds = np.array(classifier_preds)
    print("Wrong preds = ", np.sum(classifier_targets - classifier_preds))
    print("Total preds = ", len(classifier_targets))

    confusion_matrix(classifier_targets, classifier_preds)
    print(classification_report(classifier_targets, classifier_preds))


def main() -> None:

    model = SuperModel()
    eval_env = EnvironmentFactory.register(env_name, **config)

    print(
        "\n\nEvaluating performance of the supermodel with classifier on random task...\n"
    )
    eval_perf(eval_env, model)

    print(PATH_TO_HOLD_NET)


if __name__ == "__main__":
    main()
