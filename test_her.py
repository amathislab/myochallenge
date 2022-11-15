from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.environment_factory import EnvironmentFactory
from stable_baselines3.sac.policies import SACPolicy
from src.feature_extractors.dmap import DmapExtractor
import gym
import numpy as np


model_class = SAC  # works also with SAC, DDPG and TD3
max_episode_length = 150

env = EnvironmentFactory.create("GoalHistoryMyoReorientP2")

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future"  # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = False
# Time limit for the episodes

dmap_params = {
    "history_observation_space": env.history_observation_space,
    "feature_convnet_params": [
        {"num_filters": 32, "kernel_size": 5, "stride": 4},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
    ],
    "feature_fcnet_hiddens": [32, 32],
    "embedding_size": 64,
}

# Initialize the model
model = model_class(
    # "MultiInputPolicy",
    SACPolicy,
    env,
    learning_starts=0,
    buffer_size=1000,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
    policy_kwargs={
        "features_extractor_class": DmapExtractor,
        "features_extractor_kwargs": dmap_params,
        "share_features_extractor": True,
    },
)

# Train the model
model.learn(1000)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load("./her_bit_env", env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
