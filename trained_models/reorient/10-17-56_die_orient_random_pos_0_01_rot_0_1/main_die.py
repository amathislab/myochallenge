import copy
import json
import os
import shutil
from calendar import c
from datetime import datetime

import numpy as np
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory
from src.metrics.custom_callbacks import EvaluateLSTM
from src.metrics.sb_callbacks import EnvDumpCallback

env_name = "CustomMyoReorientP1"

# whether this is the first task of the curriculum (True) or it is loading a previous task (False)
FIRST_TASK = False

# Path to normalized Vectorized environment (if not first task)
PATH_TO_NORMALIZED_ENV = "output/training/2022-10-11/19-10-08_die_orient_random_pos_0_003_rot_0_1/training_env.pkl"  # "trained_models/normalized_env_original"

# Path to pretrained network (if not first task)
PATH_TO_PRETRAINED_NET = "output/training/2022-10-11/19-10-08_die_orient_random_pos_0_003_rot_0_1/best_model.zip"  # "trained_models/best_model.zip"

# Tensorboard log (will save best model during evaluation)
now = (
    datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    + "_die_orient_random_pos_0_01_rot_0_1"
)
TENSORBOARD_LOG = os.path.join("output", "training", now)


# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist": 1,
        "rot_dist": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    # "noise_palm": 0.1,
    # "noise_fingers": 0.1,
    "goal_pos": (-0.01, 0.01),  # phase 2: (-0.020, 0.020)
    "goal_rot": (-0.1, 0.1),  # phase 2: (-3.14, 3.14)
    "drop_th": 0.2,
}

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.register(env_name, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    with open(os.path.join(TENSORBOARD_LOG, "config.json"), "w") as file:
        json.dump(config, file)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=16)

    # Normalize environment:
    if FIRST_TASK:
        envs = VecNormalize(envs)
    else:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)

    # Callback to evaluate dense rewards
    env_dump_callback = EnvDumpCallback(TENSORBOARD_LOG, verbose=0)

    eval_callback = EvalCallback(
        envs,
        callback_on_new_best=env_dump_callback,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=2500,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    # Callbacks for score and for effort

    config_score, config_effort = copy.deepcopy(config), copy.deepcopy(config)

    config_score["weighted_reward_keys"].update(
        {
            "pos_dist": 0,
            "rot_dist": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        }
    )

    config_effort["weighted_reward_keys"].update(
        {
            "pos_dist": 0,
            "rot_dist": 0,
            "act_reg": 1,
            "alive": 0,
            "solved": 0,
            "done": 0,
            "sparse": 0,
        }
    )

    env_score = EnvironmentFactory.register(env_name, **config_score)
    env_effort = EnvironmentFactory.register(env_name, **config_effort)

    score_callback = EvaluateLSTM(
        eval_freq=5000, eval_env=env_score, name="eval/score", num_episodes=10
    )
    effort_callback = EvaluateLSTM(
        eval_freq=5000, eval_env=env_effort, name="eval/effort", num_episodes=10
    )

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    if FIRST_TASK:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            envs,
            verbose=2,
            tensorboard_log=TENSORBOARD_LOG,
            batch_size=32,
            n_steps=128,
            learning_rate= 2.55673e-05,
            ent_coef = 3.62109e-06,
            clip_range= 0.3,
            gamma=0.99,
            gae_lambda=0.9,
            max_grad_norm = 0.7,
            vf_coef = 0.430793,
            n_epochs=10,
            policy_kwargs=dict(
                ortho_init=False,
                log_std_init=-2,
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            ),
        )
    else:
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET, env=envs, tensorboard_log=TENSORBOARD_LOG
        )

    # Train and save model
    model.learn(
        total_timesteps=10_000_000,
        callback=[eval_callback, score_callback, effort_callback],
        reset_num_timesteps=True,
    )

    model.save(os.path.join(TENSORBOARD_LOG, "final_model.pkl"))
    envs.save(os.path.join(TENSORBOARD_LOG, "final_env.pkl"))
