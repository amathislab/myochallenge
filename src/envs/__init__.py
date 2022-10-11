import gym
import os
import myosuite
import numpy as np
from definitions import ROOT_DIR


myosuite_path = (
    os.path.join(ROOT_DIR, "data", "myosuite")
)

# MyoChallenge Baoding: Phase1 env
gym.envs.registration.register(
    id="CustomMyoChallengeBaodingP1-v1",
    entry_point="src.envs.baoding:CustomBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        # 'goal_time_period': (5, 5),
        # "goal_xrange": (0.025, 0.025),
        # "goal_yrange": (0.028, 0.028),
    },
)

# MyoChallenge Baoding: env with history
gym.envs.registration.register(
    id="HistoryMyoChallengeBaodingP1-v1",
    entry_point="src.envs.baoding:HistoryBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        # 'goal_time_period': (5, 5),
        # "goal_xrange": (0.025, 0.025),
        # "goal_yrange": (0.028, 0.028),
    },
)
