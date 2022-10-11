import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory

env_name = "CustomMyoBaodingBallsP1"

# Path to normalized Vectorized environment (if not first task)
# PATH_TO_NORMALIZED_ENV = "trained_models/env_final_period4to6.pkl"  # "trained_models/normalized_env_original"
# PATH_TO_NORMALIZED_ENV = "output/training/2022-10-04/17-20-31_final_task_with_noise_reg_ccw/training_env.pkl"  # "trained_models/normalized_env_original"
# PATH_TO_NORMALIZED_ENV = "output/training/2022-10-04/16-11-36_fully_trained_ccw_to_random_direction/training_env.pkl"  # "trained_models/normalized_env_original"
# PATH_TO_NORMALIZED_ENV = "output/training/2022-10-04/17-21-03_final_task_with_noise_reg_random/training_env.pkl"  # "trained_models/normalized_env_original"
# PATH_TO_NORMALIZED_ENV = "output/training/2022-10-04/16-10-18_final_task_ccw_no_hand_noise/training_env.pkl"  # "trained_models/normalized_env_original"
PATH_TO_NORMALIZED_ENV = "trained_models/env_rsi_static_perfect.pkl"  # "trained_models/normalized_env_original"

# Path to pretrained network (if not first task)
# PATH_TO_PRETRAINED_NET = "trained_models/final_period4to6.zip"  # "trained_models/best_model.zip"
# PATH_TO_PRETRAINED_NET = "output/training/2022-10-04/17-20-31_final_task_with_noise_reg_ccw/best_model.zip"  # "trained_models/best_model.zip"
# PATH_TO_PRETRAINED_NET = "output/training/2022-10-04/16-11-36_fully_trained_ccw_to_random_direction/best_model.zip"  # "trained_models/best_model.zip"
# PATH_TO_PRETRAINED_NET = "output/training/2022-10-04/17-21-03_final_task_with_noise_reg_random/best_model.zip"  # "trained_models/best_model.zip"
# PATH_TO_PRETRAINED_NET = "output/training/2022-10-04/16-10-18_final_task_ccw_no_hand_noise/best_model.zip"  # "trained_models/best_model.zip"
PATH_TO_PRETRAINED_NET = "trained_models/rsi_static_perfect.zip"  # "trained_models/best_model.zip"

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 0,
        "solved": 0,
        "done": 0,
        "sparse": 0,
    },
    "task": "ccw",
    "enable_rsi": True,
    "noise_palm": 0,
    "noise_fingers": 0,
    "goal_time_period": [1e6, 1e6],   # phase 2: (4, 6)
    "goal_xrange": (0.025, 0.025),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.028, 0.028),  # phase 2: (0.022, 0.032)
    "drop_th": 1.3,
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
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        # "_last_lstm_states": None
    }

    model = RecurrentPPO.load(PATH_TO_PRETRAINED_NET, env=envs, custom_objects=custom_objects)

    # EVALUATE
    eval_model = model
    # eval_env = EnvironmentFactory.create(env_name, **config)
    
    import gym
    eval_env = gym.make("myoChallengeBaodingP1-v1")
    
    # Enjoy trained agent
    num_episodes = 10
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
