import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from main import PATH_TO_PRETRAINED_NET
from src.envs.environment_factory import EnvironmentFactory

# evaluation parameters:
render = False
num_episodes = 1_000

env_name = "CustomMyoBaodingBallsP2"

# Path to normalized Vectorized environment and best model (if not first task)
load_folder = "output/training/2022-10-31/14-06-01_final-everything_v2-2_beta-dist_score-on-final"
PATH_TO_NORMALIZED_ENV = load_folder + "/training_env.pkl"
PATH_TO_PRETRAINED_NET = load_folder + "/best_model.zip"

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
    "enable_rsi": False,
    "rsi_probability": 0,
    'balls_overlap': False,
    "overlap_probability": 0,
    "limit_init_angle": False,
    "goal_time_period": [4, 6],   # phase 2: (4, 6)
    "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    'obj_size_range': (0.018, 0.024),    #(0.018, 0.024)   # Object size range. Nominal 0.022
    'obj_mass_range': (0.030, 0.300),    #(0.030, 0.300)   # Object weight range. Nominal 43 gms
    'obj_friction_change': (0.2, 0.001, 0.00002), # (0.2, 0.001, 0.00002)
    'task_choice': 'random'
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):   # pylint: disable=redefined-outer-name
    def make_env(_):
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
    envs.training = False
    envs.norm_reward = False

    # Create model
    custom_objects = {
        "lr_schedule": 0,
        "clip_range": 0,
    }
    model = RecurrentPPO.load(PATH_TO_PRETRAINED_NET, env=envs)

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    perfs = []
    lens = []
    for i in range(num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        # eval_env.reset()
        # eval_env.step(np.zeros(39))
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if render:
                eval_env.sim.render(mode="window")
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

        if (i + 1) % 50 == 0:
            print(f"\nEpisode {i+1}/{num_episodes}")
            print(f"Average len: {np.mean(lens)} +/- {np.std(lens)}")
            print(f"Average rew: {np.mean(perfs)} +/- {np.std(perfs)}\n")
