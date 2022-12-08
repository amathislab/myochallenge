# pylint: disable=no-member
import pickle
import os
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
from definitions import ROOT_DIR


warnings.filterwarnings("ignore")

device = "cuda"
env_name = "CustomMyoBaodingBallsP2"
N_OBS_PER_TRIAL = 13

# Model and env for CCW + CW (on which classifier is trained)
PATH_TO_NORMALIZED_BASE_ENV = [
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_26880000_steps.pkl",
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_27840000_steps.pkl",
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_28000000_steps.pkl",
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_41120000_steps.pkl",
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_41440000_steps.pkl",
    "trained_models/winning_ensemble/base/rl_model_vecnormalize_41280000_steps.pkl",
]
PATH_TO_BASE_NET = [
    "trained_models/winning_ensemble/base/rl_model_26880000_steps.zip",
    "trained_models/winning_ensemble/base/rl_model_27840000_steps.zip",
    "trained_models/winning_ensemble/base/rl_model_28000000_steps.zip",
    "trained_models/winning_ensemble/base/rl_model_41120000_steps.zip",
    "trained_models/winning_ensemble/base/rl_model_41440000_steps.zip",
    "trained_models/winning_ensemble/base/rl_model_41280000_steps.zip",
]

# Model and env for hold task
PATH_TO_NORMALIZED_HOLD_ENV = [
    "trained_models/winning_ensemble/hold/rl_model_vecnormalize_1.pkl",
    "trained_models/winning_ensemble/hold/rl_model_vecnormalize_2.pkl",
    "trained_models/winning_ensemble/hold/rl_model_vecnormalize_3.pkl",
    "trained_models/winning_ensemble/hold/rl_model_vecnormalize_4.pkl",
    "trained_models/winning_ensemble/hold/rl_model_vecnormalize_5.pkl",
]
PATH_TO_HOLD_NET = [
    "trained_models/winning_ensemble/hold/rl_model_1.zip",
    "trained_models/winning_ensemble/hold/rl_model_2.zip",
    "trained_models/winning_ensemble/hold/rl_model_3.zip",
    "trained_models/winning_ensemble/hold/rl_model_4.zip",
    "trained_models/winning_ensemble/hold/rl_model_5.zip",
]

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

    envs = []
    for path in env_path:
        envs.append(
            VecNormalize.load(
                path,
                DummyVecEnv([lambda: EnvironmentFactory.register(env_name, **config)]),
            )
        )

    return envs


def load_model_and_env(task: str):

    if task == "hold":
        model_paths = PATH_TO_HOLD_NET
        custom_objects = {
            "learning_rate": lambda _: 1e-4,
            "lr_schedule": lambda _: 1e-4,
            "clip_range": lambda _: 0.2,
        }
    elif task == "base":
        model_paths = PATH_TO_BASE_NET
        custom_objects = {
            "learning_rate": lambda _: 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    else:
        raise ValueError(f"Task {task} not recognized")

    envs = load_normalized_envs(task)
    models = []
    for env, model_path in zip(envs, model_paths):
        model = RecurrentPPO.load(
            model_path, env=env, custom_objects=custom_objects, device=device
        )
        models.append(model)

    return models, envs


class SuperModel:
    def __init__(self):
        # load trained classifier, models and envs
        self.models_base, self.envs_base = load_model_and_env("base")
        self.models_hold, self.envs_hold = load_model_and_env("hold")

        for env in self.envs_base + self.envs_hold:
            env.training = False
            env.norm_reward = False

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
            torch.load(os.path.join(ROOT_DIR, "trained_models/winning_ensemble/classifier/classifier.pt"), map_location=torch.device('cpu'))
        )
        return classifier

    def load_data_scaler(self):
        with open(os.path.join(ROOT_DIR, "trained_models/winning_ensemble/classifier/scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return scaler

    @property
    def envs(self):
        if self.use_hold_net:
            return self.envs_hold
        return self.envs_base

    def normalize_obs(self, obs) -> list:
        obs_list = []
        envs = self.envs_hold if self.use_hold_net else self.envs_base
        for env in envs:
            obs_list.append(env.normalize_obs(obs))
        return obs_list

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
        obs_list = self.normalize_obs(obs)

        if self.use_hold_net:
            for obs in obs_list:
                actions, lstm_states = [], []
                if self.just_switched:
                    state = None
                    self.just_switched = False
                action, lstm_state = self.model_hold.predict(
                    obs, state, episode_start, deterministic
                )
                actions.append(action)
                lstm_states.append(lstm_state)
        else:
            return self.model_base.predict(obs, state, episode_start, deterministic)


def eval_perf(eval_env: BaodingEnvV1, mixture_model: SuperModel):
    # Enjoy trained agent
    perfs, lens, effort, cum_rew, step, eff = [], [], [], 0, 0, 0
    obs = eval_env.reset()
    episode_starts = np.ones((1,), dtype=bool)
    classifier_targets, classifier_preds = [], []
    num_episodes = 0

    actions_hold = []
    lstm_states_hold = [None] * len(PATH_TO_HOLD_NET)
    lstm_states_base = [None] * len(PATH_TO_BASE_NET)
    eval_models = mixture_model.models_base

    for _ in range(2000 * 200):
        mixture_model.process_before_action(obs, episode_starts)
        envs = mixture_model.envs
        actions_hold = []
        actions_base = []

        if mixture_model.use_hold_net:

            eval_models = mixture_model.models_hold

            if mixture_model.just_switched:
                mixture_model.just_switched = False
                lstm_states_hold = [None] * len(PATH_TO_HOLD_NET)

            for i, eval_model in enumerate(eval_models):
                action, lstm_states_hold[i] = eval_model.predict(
                    envs[i].normalize_obs(obs),
                    state=lstm_states_hold[i],
                    episode_start=episode_starts,
                    deterministic=True,
                )
                actions_hold.append(action)
                action_to_take = np.mean(actions_hold, axis=0)
        else:

            eval_models = mixture_model.models_base

            for i, eval_model in enumerate(eval_models):
                action, lstm_states_base[i] = eval_model.predict(
                    envs[i].normalize_obs(obs),
                    state=lstm_states_base[i],
                    episode_start=episode_starts,
                    deterministic=True,
                )
                actions_base.append(action)
                action_to_take = np.mean(actions_base, axis=0)

        # compute effort
        act_mag = (
            np.linalg.norm(eval_env.obs_dict["act"], axis=-1) / eval_env.sim.model.na
            if eval_env.sim.model.na != 0
            else 0
        )

        # step in the environment
        obs, rewards, dones, _ = eval_env.step(action_to_take)
        episode_starts = dones
        cum_rew += rewards
        step += 1
        eff += act_mag

        if step == 13:
            classifier_targets.append([eval_env.which_task.value])
            classifier_preds.append([mixture_model.current_task])

        if dones:
            num_episodes += 1
            lens.append(step)
            perfs.append(cum_rew)
            effort.append(eff / step if step != 0 else 0)

            obs = eval_env.reset()
            cum_rew, step, eff = 0, 0, 0
            episode_starts = np.ones((1,), dtype=bool)

            eval_models = mixture_model.models_base
            lstm_states_base = [None] * len(PATH_TO_BASE_NET)
            lstm_states_hold = [None] * len(PATH_TO_HOLD_NET)

            if num_episodes % 10 == 0:
                # print performance metrics
                print(f"\nEpisode {num_episodes}")
                print(
                    f"Average len: {np.mean(lens):.2f} +/- {np.std(lens)/np.sqrt(num_episodes):.2f}"
                )
                print(
                    f"Average rew: {np.mean(perfs):.2f} +/- {np.std(perfs)/np.sqrt(num_episodes):.2f}"
                )
                print(f"Average eff: {np.mean(effort):.5f} +/- {np.std(effort):.5f}\n")

                # if num_episodes % 100 == 0:
                # classifier metrics
                num_errors = np.sum(
                    abs(
                        np.array(classifier_targets).clip(0, 1)
                        - np.array(classifier_preds)
                    )
                )
                print(
                    f"Classifier inaccuracy = {num_errors/len(classifier_targets)*100:.1f}%\n"
                )

                print("Base nets:")
                for path in PATH_TO_BASE_NET:
                    print(path)

                print("\nHold nets:")
                for path in PATH_TO_HOLD_NET:
                    print(path)

    # classifier metrics
    classifier_targets = np.array(classifier_targets).clip(0, 1)
    classifier_preds = np.array(classifier_preds)
    confusion_matrix(classifier_targets, classifier_preds)
    print(classification_report(classifier_targets, classifier_preds))


def main() -> None:

    model = SuperModel()
    eval_env = EnvironmentFactory.register(env_name, **config)

    print(
        "\n\nEvaluating performance of the supermodel with classifier on random task...\n"
    )
    eval_perf(eval_env, model)


if __name__ == "__main__":
    main()
