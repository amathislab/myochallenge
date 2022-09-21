import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory

env_name = "CustomMyoBaodingBallsP1"

# Path to normalized Vectorized environment (if not first task)
PATH_TO_NORMALIZED_ENV = "trained_models/normalized_env_original"  # "trained_models/normalized_env_original"

# Path to pretrained network (if not first task)
PATH_TO_PRETRAINED_NET = "output/training/best_model.zip"  # "trained_models/best_model.zip"

# Tensorboard log (will save best model during evaluation)
TENSORBOARD_LOG = "output/training"


# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    "goal_time_period": [1e6, 1e6],
    "task": "random",
    "enable_rsi": True
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

if __name__ == "__main__":
    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=16)

    # Normalize environment:
    envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)

    # Create callback to evaluate
    eval_callback = EvalCallback(
        envs,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    model = RecurrentPPO.load(PATH_TO_PRETRAINED_NET, env=envs)

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    perfs, lens, lstm_states, cum_rew, step = [], [], None, 0, 0
    obs = eval_env.reset()
    episode_starts = np.ones((1,), dtype=bool)
    for i in range(5000):
        eval_env.sim.render(mode="window")
        action, lstm_states = eval_model.predict(
            envs.normalize_obs(obs),
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, info = eval_env.step(action)
        episode_starts = dones
        cum_rew += rewards
        step += 1
        if dones:
            episode_starts = np.ones((1,), dtype=bool)
            lstm_states = None
            obs = eval_env.reset()
            lens.append(step)
            perfs.append(cum_rew)
            cum_rew, step = 0, 0

    print(("Average len:", np.mean(lens), "     ", "Average rew:", np.mean(perfs)))
