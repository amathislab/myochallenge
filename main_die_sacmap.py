import copy
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.sac.policies import SACPolicy
import os
import shutil
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from src.envs.environment_factory import EnvironmentFactory
from src.feature_extractors.dmap import DmapExtractor


env_name = "GoalHistoryMyoReorientP2"

# whether this is the first task of the curriculum (True) or it is loading a previous task (False)
FIRST_TASK = True

# Path to normalized Vectorized environment (if not first task)
PATH_TO_NORMALIZED_ENV = None

# Path to pretrained network (if not first task)
PATH_TO_PRETRAINED_NET = None
# Tensorboard log (will save best model during evaluation)
now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S") + "_die_sacmap_online_0.5_rot_0.01_pos_small_noise"
TENSORBOARD_LOG = os.path.join("output", "training", now)
RecurrentPPO

# Reward structure and task parameters:
env_config = {
    "weighted_reward_keys": {
        "pos_dist": 1,
        "rot_dist": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    # "noise_palm": 0.1,
    # "noise_fingers": 0.1,
    "goal_pos": (-0.01, 0.01),  # phase 2: (-0.020, 0.020)
    "goal_rot": (-0.5, 0.5),  # phase 2: (-3.14, 3.14)
    "obj_size_change": 0.001,  # 0.007
    "obj_friction_change": (0.02, 0.0001, 0.000002),  # (0.2, 0.001, 0.00002)
}

dmap_config = {
    "feature_convnet_params": [
        {"num_filters": 32, "kernel_size": 5, "stride": 4},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
        {"num_filters": 32, "kernel_size": 3, "stride": 1},
    ],
    "feature_fcnet_hiddens": [32, 32],
    "embedding_size": 64,
}

if __name__ == "__main__":
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    env = EnvironmentFactory.create(env_name, **env_config)
    env = Monitor(env, TENSORBOARD_LOG)

    dmap_config.update(
        {
            "history_observation_space": env.history_observation_space,
        }
    )

    # Callback to evaluate dense rewards
    eval_callback = EvalCallback(
        env,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=TENSORBOARD_LOG
    )

    # Callbacks for score and for effort

    config_score, config_effort = copy.deepcopy(env_config), copy.deepcopy(env_config)

    config_score["weighted_reward_keys"].update(
        {
            "pos_dist": 0,
            "rot_dist": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        }
    )

    env_score = EnvironmentFactory.create(env_name, **config_score)

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    if FIRST_TASK:
        model = SAC(
            # "MultiInputPolicy",
            SACPolicy,
            env,
            learning_starts=10000,
            buffer_size=300000,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                online_sampling=True,
                max_episode_length=150,
            ),
            verbose=1,
            policy_kwargs={
                "features_extractor_class": DmapExtractor,
                "features_extractor_kwargs": dmap_config,
                "share_features_extractor": True,
            },
            tensorboard_log=TENSORBOARD_LOG,
        )
    else:
        model = SAC.load(
            PATH_TO_PRETRAINED_NET, env=env, tensorboard_log=TENSORBOARD_LOG
        )

    # Train and save model
    model.learn(
        total_timesteps=50_000_000,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=True,
    )

    model.save(os.path.join(TENSORBOARD_LOG, "final_model.zip"))
    env.save(os.path.join(TENSORBOARD_LOG, "final_env.pkl"))
