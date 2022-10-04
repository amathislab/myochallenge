from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0.1,
        "alive": 0,
        "solved": 1,
        "done": 0,
        "sparse": 0,
    },
    "task": "ccw",
    "enable_rsi": False,
    "noise_palm": 0,
    "noise_fingers": 0,
    "goal_time_period": [5, 5],     # phase 2: (4, 6)
    "goal_xrange": (0.025, 0.025),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.028, 0.028),  # phase 2: (0.022, 0.032)
    "drop_th": 1.25,
}

# Load expert PPO and envs

# Function that creates and monitors vectorized environments:
def make_parallel_envs_w_rollout_wrapper(env_name, env_config, num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.register(env_name, **env_config)
            env = RolloutInfoWrapper(env)
            # env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

# Create vectorized environments:
envs = make_parallel_envs_w_rollout_wrapper("CustomMyoBaodingBallsP1", config, num_env=16)
envs = VecNormalize.load("./trained_models/env_final_period4to6.pkl", envs)

# load expert model
custom_objects = {      # need to define this since my python version is newer
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
expert = RecurrentPPO.load("./trained_models/final_period4to6.zip", env=envs, custom_objects=custom_objects)


# collect rollouts from expert
rollouts = rollout.rollout(
    expert,
    envs,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
)
transitions = rollout.flatten_trajectories(rollouts)

print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)
