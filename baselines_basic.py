import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from definitions import ROOT_DIR
from src.envs.environment_factory import EnvironmentFactory
from src.helpers.experiment_config import ExperimentConfig
from src.models.model_factory import ModelFactory

task_name = "baodingBalls"
file_name = "sac_baodingBalls_P1.json"

config_path = os.path.join(ROOT_DIR, "configs", task_name, file_name)
config = ExperimentConfig(config_path)
# Static goals
env = EnvironmentFactory.register(config.env_name, **config.env_config)
# Original task
# env = EnvironmentFactory.register(config.env_name)
ModelFactory.register_models_from_config(config.policy_configs)


# Create parallel environments
def make_parallel_envs(env_name, env_config, num_env, tensorboard_log, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.register(env_name, **env_config)
            env = Monitor(env, tensorboard_log)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


envs = make_parallel_envs(
    config.env_name, config.env_config, num_env=16, tensorboard_log="./static_goal_ppo/"
)


# Train
policy_kwargs = dict(net_arch=[256, 256])

model = PPO(
    "MlpPolicy",
    envs,
    policy_kwargs=policy_kwargs,
    verbose=2,
    tensorboard_log="./static_goal_ppo/",
    batch_size=1024,
    n_steps=1280,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=6,
)
model.learn(total_timesteps=10000000)
model.save("static_goal_ppo")
