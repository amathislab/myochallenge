import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from main import PATH_TO_PRETRAINED_NET

from src.envs.environment_factory import EnvironmentFactory

env_name = "CustomMyoReorientP1"

# Path to normalized Vectorized environment (if not first task)
# PATH_TO_NORMALIZED_ENV = "output/training/2022-09-23_12-16-54/training_env.pkl"  # "trained_models/normalized_env_original"
PATH_TO_NORMALIZED_ENV = "output/training/2022-10-12/10-17-56_die_orient_random_pos_0_01_rot_0_1/training_env.pkl"


# Path to pretrained network (if not first task)
# PATH_TO_PRETRAINED_NET = "output/training/2022-09-23_12-16-54/best_model.zip"  # "trained_models/best_model.zip"
PATH_TO_PRETRAINED_NET = (
    "output/training/2022-10-12/10-17-56_die_orient_random_pos_0_01_rot_0_1/best_model.zip"
)

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist": 0,
        "rot_dist": 0,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    # "noise_palm": 0.1,
    # "noise_fingers": 0.1,
    "goal_pos": (-0.01, 0.01),  # phase 2: (-0.020, 0.020)
    "goal_rot": (-0.3, 0.3),  # phase 2: (-3.14, 3.14)
    "drop_th": 0.2,
}

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=16)

    # Normalize environment:
    envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    model = RecurrentPPO.load(PATH_TO_PRETRAINED_NET, env=envs)

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    num_episodes = 100
    perfs = []
    lens = []
    for i in range(num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            # eval_env.sim.render(mode="window")
            action, lstm_states = eval_model.predict(
                envs.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_rew += rewards
            step += 1
        lens.append(step)
        perfs.append(cum_rew)
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)
    print(("Average len:", np.mean(lens), "     ", "Average rew:", np.mean(perfs)))
