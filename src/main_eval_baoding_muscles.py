import os
import numpy as np
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from train.trainer import MyoTrainer


# evaluation parameters:
render = True
num_episodes = 2_000

env_name = "MuscleBaodingEnv"

# Path to normalized Vectorized environment and best model (if not first task)
PATH_TO_NORMALIZED_ENV = None
PATH_TO_PRETRAINED_NET = None

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    "obs_mode": "dict",
    "include_adapt_state": False,
    "num_memory_steps": 0,
    "task_choice": "random",
    # custom params for curriculum learning
    "enable_rsi": True,
    "rsi_probability": 1,
    "balls_overlap": True,
    "overlap_probability": 1,
    "noise_fingers": None,
    "limit_init_angle": 0,
    # "beta_init_angle": [0.9,0.9], # caution: doesn't work if limit_init_angle = False
    "goal_time_period": [1e6, 1e6],  # phase 2: (4, 6)
    "goal_xrange": (0.025, 0.025),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.028, 0.028),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    "obj_size_range": (0.022, 0.022),  # phase 2: (0.018, 0.024)
    "obj_mass_range": (0.043, 0.043),  # phase 2: (0.030, 0.300)
    "obj_friction_change": (0, 0, 0),  # phase 2: (0.2, 0.001, 0.00002)
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(
    env_name, env_config, num_env, start_index=0
):  # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=1)

    # Normalize environment:
    if PATH_TO_NORMALIZED_ENV is not None:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    else:
        envs = VecNormalize(envs)
    envs.training = False
    envs.norm_reward = False

    # Create model
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    if PATH_TO_PRETRAINED_NET is not None:
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET,
            env=envs,
            device="cpu",
            custom_objects=custom_objects,
        )
    else:
        model = MyoTrainer(
            envs=envs,
            env_config={},
            load_model_path=PATH_TO_PRETRAINED_NET,
            log_dir=os.path.join(ROOT_DIR, "output", "testing"),
        ).agent

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    perfs = []
    lens = []
    for i in range(num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        # eval_env.reset()
        # eval_env.step(np.zeros(39))
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        if render:
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            time.sleep(0)
        while not done:
            if render:
                eval_env.sim.render(mode="window")
                eval_env.sim.render(mode="window")
                eval_env.sim.render(mode="window")
            action, lstm_states = eval_model.predict(
                envs.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, info = eval_env.step(action)
            if step == 0:
                print(info["rwd_dict"])
            episode_starts = done
            cum_rew += rewards
            step += 1
        lens.append(step)
        perfs.append(cum_rew)
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)

        if (i + 1) % 10 == 0:
            len_error = np.std(lens) / np.sqrt(i + 1)
            perf_error = np.std(perfs) / np.sqrt(i + 1)

            print(f"\nEpisode {i+1}/{num_episodes}")
            print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")

    print(f"\nFinished evaluating {PATH_TO_PRETRAINED_NET}!")
