import collections
import numpy as np
import pandas as pd

Transition = collections.namedtuple("Transition", ["episode_id", "obs", "action", "next_obs", "reward", "done", "info"])


class DataSaver:
    def __init__(self):
        self.data = []
        self.episode_id = 0

    def begin_rollout(self):
        pass

    def append_step(self, obs, action, next_obs, reward, done, info):
        self.data.append(Transition(self.episode_id, obs, action, next_obs, reward, done, info))

    def end_rollout(self):
        self.episode_id += 1

    def reset(self):
        self.data = []
        self.episode_id = 0

    def save_df(self, out_path):
        df = pd.DataFrame(self.data)
        df.to_csv(out_path)

