from src.models.model_factory import ModelFactory
from src.envs.environment_factory import EnvironmentFactory
from src.helpers.experiment_config import ExperimentConfig


def get_env_and_config(config_path, update_params=None):
    """Creates an environment and an EnvironmentConfig

    Args:
        config_path (str): path to the config json
        update_params (dict, optional): config parameters to overwrite. Defaults to None.

    Returns:
        tuple(gym.Env, EnvironmentConfig): environment and configuration
    """
    config = ExperimentConfig(config_path, update_params)
    env = EnvironmentFactory.register(config.env_name, **config.env_config)
    for policy in config.policy_configs.values():
        for model_params in policy.values():
            if isinstance(model_params, dict):
                model_name = model_params.get("custom_model")
                if model_name is not None:
                    ModelFactory.register(model_name)
    return env, config


def get_trainer(config, checkpoint_path=None):
    """Creates a trainer from an ExperimentConfig object

    Args:
        config (ExperimentConfig): configuration object
        checkpoint_path (str, optional): path to the model to restore. Defaults to None.

    Returns:
        ray.rllib.agents.Trainer: trainer object
    """
    trainer_config = config.get_trainer_config()
    trainer_config["num_workers"] = 0
    trainer = config.trainer_cls(config=trainer_config)
    if checkpoint_path is not None:
        trainer.restore(checkpoint_path)
    return trainer
