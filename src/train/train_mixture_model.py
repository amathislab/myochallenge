import copy
import json
import os
import shutil
from calendar import c
from datetime import datetime

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory
from src.metrics.custom_callbacks import EvaluateLSTM
from src.metrics.sb_callbacks import EnvDumpCallback

# saving criteria
saving_criteria = "score"  # score

# Model and env for CCW + CW (on which classifier is trained)
load_folder = "trained_models/baoding_phase2/alberto_518"
PATH_TO_NORMALIZED_BASE_ENV = load_folder + "/training_env.pkl"
PATH_TO_BASE_NET = load_folder + "/best_model.zip"

# Model and env for hold task
load_folder_hold = "output/training/2022-11-03/17-10-36_mixture-hold_beta0202/"
PATH_TO_NORMALIZED_HOLD_ENV = load_folder_hold + "final_env.pkl"
PATH_TO_HOLD_NET = load_folder_hold + "final_model.pkl"

# Tensorboard log (will save best model during evaluation)
# now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + "_final_mixture-hold_2"
now = "2022-11-03/23-01-01_final_mixture-hold_from-beta02"
TENSORBOARD_LOG = os.path.join("output", "training", now)


# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 5,
        "pos_dist_2": 5,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    # custom params for curriculum learning
    "enable_rsi": False,
    "rsi_probability": 0,
    "balls_overlap": False,
    "overlap_probability": 0,
    "noise_fingers": 0,
    # "limit_init_angle": np.pi,
    # "beta_init_angle": [0.2,0.2],   # caution: doesn't work if limit_init_angle = False
    "goal_time_period": [4, 6],  # phase 2: (4, 6)
    "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    "obj_size_range": (
        0.018,
        0.024,
    ),  # (0.018, 0.024   # Object size range. Nominal 0.022
    # "beta_ball_size": [0.9,0.9],
    "obj_mass_range": (
        0.030,
        0.300,
    ),  # (0.030, 0.300)   # Object weight range. Nominal 43 gms
    # "beta_ball_mass": [0.9,0.9],
    "obj_friction_change": (0.2, 0.001, 0.00002),  # (0.2, 0.001, 0.00002)
    # Task
    "task_choice": "random",
}


config_hold = copy.deepcopy(config)
config_hold.update({
    "goal_time_period": [1e100, 1e100],

    "base_model_path": PATH_TO_BASE_NET,
    "base_env_path": PATH_TO_NORMALIZED_BASE_ENV,
    "base_env_name": "CustomMyoBaodingBallsP2",
    "base_env_config": config,
})


# Function that creates and monitors vectorized environments
def make_parallel_envs(
    env_name, env_config, num_env, start_index=0
):  # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.register(env_name, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    env_name = "MixtureModelBaodingEnv"

    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    with open(      # pylint: disable=unspecified-encoding
        os.path.join(TENSORBOARD_LOG, "config.json"), "w"
    ) as file:
        json.dump(config_hold, file)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config_hold, num_env=16)
    envs = VecNormalize.load(PATH_TO_NORMALIZED_HOLD_ENV, envs)

    # Callbacks for score and for effort
    config_score, config_effort = copy.deepcopy(config_hold), copy.deepcopy(config_hold)

    config_score.update(
        {
            "weighted_reward_keys": {
                "pos_dist_1": 0,
                "pos_dist_2": 0,
                "act_reg": 0,
                "solved": 5,
                "alive": 0,
                "done": 0,
                "sparse": 0,
            },
            # score on the final noise distribution
            "noise_fingers": 0,
            "limit_init_angle": False,
            "beta_init_angle": False,
            "beta_ball_size": False,
            "beta_ball_mass": False,
        }
    )

    config_effort.update(
        {
            "weighted_reward_keys": {
                "pos_dist_1": 0,
                "pos_dist_2": 0,
                "act_reg": 1,
                "solved": 0,
                "alive": 0,
                "done": 0,
                "sparse": 0,
            },
            # effort on the final noise distribution
            "noise_fingers": 0,
            "limit_init_angle": False,
            "beta_init_angle": False,
            "beta_ball_size": False,
            "beta_ball_mass": False,
        }
    )

    env_score = EnvironmentFactory.register(env_name, **config_score)
    env_effort = EnvironmentFactory.register(env_name, **config_effort)

    score_callback = EvaluateLSTM(
        eval_freq=1_200_000, eval_env=env_score, name="eval/score", num_episodes=20
    )
    # effort_callback = EvaluateLSTM(
    #     eval_freq=5000, eval_env=env_effort, name="eval/effort", num_episodes=10
    # )

    # Evaluation Callback

    # Create vectorized environments:
    if saving_criteria=="score":
        eval_envs = make_parallel_envs(env_name, config_score, num_env=1)
    elif saving_criteria=="dense_rewards":
        eval_envs = make_parallel_envs(env_name, config_hold, num_env=1)
    else:
        raise ValueError("Unrecognized saving criteria")


    eval_envs = VecNormalize.load(PATH_TO_NORMALIZED_HOLD_ENV, eval_envs)
    env_dump_callback = EnvDumpCallback(TENSORBOARD_LOG, verbose=0)

    eval_callback = EvalCallback(
        eval_envs,
        callback_on_new_best=env_dump_callback,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=20_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000, save_path=TENSORBOARD_LOG, save_vecnormalize=True
    )

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    custom_objects = {
        "lr_schedule": lambda _: 3e-5,
        "learning_rate": lambda _: 3e-5,
        "clip_range": 0.2,
        "n_steps": 4096,
        "batch_size": 4096,
        "ent_coef": 0.0,
        "n_epochs": 10,
        # "vf_coef": 1,
    }
    model = RecurrentPPO.load(
        PATH_TO_HOLD_NET,
        env=envs,
        tensorboard_log=TENSORBOARD_LOG,
        device="cuda:0",
        custom_objects=custom_objects
    )

    # Train and save model
    model.learn(
        total_timesteps=20_000_000,
        callback=[eval_callback, score_callback, checkpoint_callback], # effort_callback],
        reset_num_timesteps=True,
    )

    model.save(os.path.join(TENSORBOARD_LOG, "final_model.pkl"))
    envs.save(os.path.join(TENSORBOARD_LOG, "final_env.pkl"))
