import json
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import List
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class Trainer(ABC):
    """
    Protocol to train a library-independent RL algorithm on a gym environment.
    """

    envs: VecNormalize
    env_config: dict
    model_config: dict
    model_path: str
    total_timesteps: int
    log: bool

    def _init_agent(self) -> None:
        """Initialize the agent."""

    def train(self, total_timesteps: int) -> None:
        """Train agent on environment for total_timesteps episdodes."""


@dataclass
class BaodingTrainer:
    envs: VecNormalize
    env_config: dict
    load_model_path: str
    log_dir: str
    model_config: dict = None
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 10_000_000

    def __post_init__(self):
        self.dump_env_config(path=self.log_dir)
        self.agent = self._init_agent()

    def dump_env_config(self, path: str) -> None:
        with open(os.path.join(path, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(self.env_config, f)

    def _init_agent(self) -> RecurrentPPO:
        if self.load_model_path is not None:
            return RecurrentPPO.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
            )
        print("\nNo model path provided. Initializing new model.\n")
        return RecurrentPPO(
            "MlpLstmPolicy",
            self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=True,
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
