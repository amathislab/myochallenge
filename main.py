import os
import numpy as np
import torch.nn as nn
import json
import shutil
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.envs.environment_factory import EnvironmentFactory
from src.metrics.sb_callbacks import EnvDumpCallback


env_name = "CustomMyoBaodingBallsP1"

# whether this is the first task of the curriculum (True) or it is loading a previous task (False)
FIRST_TASK = False

# Path to normalized Vectorized environment (if not first task)
PATH_TO_NORMALIZED_ENV = "output/training/2022-09-23_12-16-54/training_env.pkl"  # "trained_models/normalized_env_original"

# Path to pretrained network (if not first task)
PATH_TO_PRETRAINED_NET = "output/training/2022-09-23_12-16-54/best_model.zip"  # "trained_models/best_model.zip"

# Tensorboard log (will save best model during evaluation)
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TENSORBOARD_LOG = os.path.join("output", "training", now)


# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 1,
        "done": 0,
        "sparse": 0,
    },
    "goal_time_period": [1e6, 1e6],
    "task": "random",
    "enable_rsi": True,
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

    # Create callback to evaluate
    env_dump_callback = EnvDumpCallback(TENSORBOARD_LOG, verbose=0)
    eval_callback = EvalCallback(
        envs,
        callback_on_new_best=env_dump_callback,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    if FIRST_TASK:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            envs,
            verbose=2,
            tensorboard_log=TENSORBOARD_LOG,
            batch_size=128,
            n_steps=256,
            gamma=0.99,
            gae_lambda=0.9,
            n_epochs=10,
            policy_kwargs=dict(
                ortho_init=False,
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[], vf=[])],
                enable_critic_lstm=True,
                lstm_hidden_size=128,
            ),
        )
    else:
        model = RecurrentPPO.load(PATH_TO_PRETRAINED_NET, env=envs)

    # Train and save model
    model.learn(total_timesteps=10000000, callback=eval_callback)
    model.save("model_name")
