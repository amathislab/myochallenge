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
ENV_NAME = "CustomMyoReorientP2"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", now) + "reorient_2pi_rot_0_pos_static"

# Path to normalized Vectorized environment and best model (if not first task)
PATH_TO_NORMALIZED_ENV = None
PATH_TO_PRETRAINED_NET = None

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist": 0.5,
        "rot_dist": 0.02,
        "pos_dist_diff": 50,
        "rot_dist_diff": 5,
        "alive": 0.1,
        "act_reg": 0,
        "solved": 0.5,
        "done": 0,
        "sparse": 0,
    },
    "goal_pos": (-0.02, 0.02),  # (-.020, .020), +- 2 cm
    "goal_rot": (-3.14, 3.14),  # (-3.14, 3.14), +-180 degrees
    # Randomization in physical properties of the die
    "obj_size_change": 0,  # 0.007 +-7mm delta change in object size
    "obj_friction_change": (0, 0, 0),  # (0.2, 0.001, 0.00002)
    "enable_rsi": True,
    "rsi_distance_pos": 0,
    "rsi_distance_rot": 0,
    "goal_rot_x": None,
    "goal_rot_y": None,
    "goal_rot_z": None,
}

model_config = dict(
    device="cuda",
    batch_size=32,
    n_steps=128,
    learning_rate=2.55673e-05,
    ent_coef=3.62109e-06,
    clip_range=0.3,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.835671,
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
        info_keywords=("pos_dist", "rot_dist", "pos_dist_diff", "rot_dist_diff", "act_reg", "alive", "solved")
    )

    # Define trainer
    trainer = MyoTrainer(
        envs=envs,
        env_config=config,
        load_model_path=PATH_TO_PRETRAINED_NET,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[eval_callback, checkpoint_callback, tensorboard_callback],
        timesteps=10_000_000,
    )

    # Train agent
    trainer.train(total_timesteps=trainer.timesteps)
    trainer.save()
