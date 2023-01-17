
import os
import shutil
import torch.nn as nn
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback, TensorboardCallback
from train.trainer import MyoTrainer


# define constants
ENV_NAME = "CustomMyoPenTwirlRandom"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", now) + "_pen_twirl_random_distance_full_from_0.5_sds_4096_resume"

# Path to normalized Vectorized environment and best model (if not first task)
PATH_TO_NORMALIZED_ENV = os.path.join(ROOT_DIR, "output/training/2023-01-13/10-09-33_pen_twirl_random_distance_full_from_0.5_sds_4096_resume/final_env.pkl")
PATH_TO_PRETRAINED_NET = os.path.join(ROOT_DIR, "output/training/2023-01-13/10-09-33_pen_twirl_random_distance_full_from_0.5_sds_4096_resume/final_model.pkl")

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_align": 0,
        "rot_align": 0,
        "pos_align_diff": 1e2,
        "rot_align_diff": 1e2,
        "alive": 0,
        "act_reg": 0,
        "drop": 0,
        "bonus": 0,
        "solved": 1,
        "done": 0,
        "sparse": 0,
    },
    "goal_orient_range": (-1, 1),  # (-1, 1)
    "enable_rsi": False,
    "rsi_distance": None,

}

model_config = dict(
    device="cuda",
    batch_size=4096,
    n_steps=4096,
    learning_rate=5e-05,
    ent_coef=0.00025,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.5,
    n_epochs=10,
    policy_kwargs=dict(
        ortho_init=False,
        log_std_init=-2,
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    ),
)

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config, num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(ENV_NAME, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config, 16)
    
    if PATH_TO_NORMALIZED_ENV is not None:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    else:
        envs = VecNormalize(envs)

    # Define callbacks for evaluation and saving the agent
    eval_callback = EvalCallback(
        eval_env=envs,
        callback_on_new_best=EnvDumpCallback(TENSORBOARD_LOG, verbose=0),
        n_eval_episodes=10,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )
    
    tensorboard_callback = TensorboardCallback(
        info_keywords=list(config["weighted_reward_keys"].keys())
   )

    # Define trainer
    trainer = MyoTrainer(
        envs=envs,
        env_config=config,
        load_model_path=PATH_TO_PRETRAINED_NET,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[eval_callback, checkpoint_callback, tensorboard_callback],
        timesteps=500_000_000,
    )

    # Train agent
    trainer.train(total_timesteps=trainer.timesteps)
    trainer.save()
