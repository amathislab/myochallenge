import collections.abc
import json

from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import APPOTrainer, PPOTrainer
from ray.rllib.agents.sac import SACTrainer

from metrics.callbacks_factory import CallbacksFactory


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ExperimentConfig:
    """Class that wraps the configuration json for a training experiment, adding some functionality."""

    def __init__(self, config_path, update_params=None):
        """Init function

        Args:
            config_path (str): path to the config file
            update_params (dict, optional): dictionary of the config parameters to be updated.
            Overwrites the ones included in the config file. Accepts nested dicts. Defaults to None.
        """
        with open(config_path, "rb") as config_file:
            self.json_config = json.load(config_file)
        if update_params is not None:
            update(self.json_config, update_params)
        self.num_trainer_workers = self.json_config.get("num_trainer_workers")
        self.env_name = self.json_config.get("env_name")
        self.env_config = self.json_config.get("env_config")
        self.policies_to_train = [
            key
            for key, value in self.json_config.get("policies_to_train").items()
            if value
        ]
        self.policy_classes = self.json_config.get("policy_classes")
        self.policy_configs = self.json_config.get("policy_configs")
        self.agent_policy_dict = self.json_config.get("agent_policy_dict")
        self.policy_mapping_fn = self.json_config.get("policy_mapping_fn")
        self.gamma = self.json_config.get("gamma")
        self.rollout_fragment_length = self.json_config.get("rollout_fragment_length")
        self.train_batch_size = self.json_config.get("train_batch_size")
        self.lr = self.json_config.get("lr")
        self.num_gpus = self.json_config.get("num_gpus")
        self.callbacks = self.json_config.get("callbacks")
        self.extra_trainer_params = self.json_config.get("extra_trainer_params")
        self.trainer_cls = self.get_trainer_class(self.json_config.get("trainer_class"))
        self.episodes_total = self.json_config.get("episodes_total")
        self.episode_reward_mean = self.json_config.get("episode_reward_mean")
        self.logdir = self.json_config.get("logdir")
        self.trial_name = self.json_config.get("trial_name")
        self.checkpoint_freq = self.json_config.get("checkpoint_freq")
        self.restore_checkpoint_path = self.json_config.get("restore_checkpoint_path")
        self.output = self.json_config.get("output")
        self.output_max_file_size = self.json_config.get("output_max_file_size")

    def save(self, path):
        """Persist the json configuration, with the updated parameters.

        Args:
            path (str): path to the output json
        """
        with open(path, "w") as file:  # pylint: disable=unspecified-encoding
            json.dump(self.json_config, file)

    def get_trainer_config(self):
        """Creates the configuration for a single agent trainer. If the configuration
        file is meant for multiple agents, throws an AssertionError

        Returns:
            dict: configuration of the trainer
        """
        assert len(self.policy_configs) == 1
        _, policy_config = next(iter(self.policy_configs.items()))
        trainer_config = {
            "num_workers": self.num_trainer_workers,
            "env": self.env_name,
            "env_config": self.env_config,
            "gamma": self.gamma,
            "rollout_fragment_length": self.rollout_fragment_length,
            "train_batch_size": self.train_batch_size,
            "lr": self.lr,
            "num_gpus": self.num_gpus,
            "callbacks": CallbacksFactory.get_callbacks_class(self.callbacks),
            "output": self.output,
            "output_max_file_size": self.output_max_file_size,
            **self.extra_trainer_params,
            **policy_config,
        }
        return trainer_config

    @staticmethod
    def get_trainer_class(trainer_name):
        if trainer_name == "ppo":
            return PPOTrainer
        elif trainer_name == "appo":
            return APPOTrainer
        elif trainer_name == "a3c":
            return A3CTrainer
        elif trainer_name == "apex":
            return ApexTrainer
        elif trainer_name == "dqn":
            return DQNTrainer
        elif trainer_name == "sac":
            return SACTrainer
        elif trainer_name == "impala":
            return ImpalaTrainer
        elif trainer_name == "ddpg":
            return DDPGTrainer
        else:
            raise ValueError(f"Unknown trainer class {trainer_name}")
