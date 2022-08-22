import os
import ray
from datetime import datetime
from ray.tune import tune
from definitions import ROOT_DIR
from src.envs.environment_factory import EnvironmentFactory
from src.helpers.experiment_config import ExperimentConfig
from src.models.model_factory import ModelFactory
from ray.tune.logger import UnifiedLogger


task_name = "baodingBalls"
file_name = "sac_baodingBalls_P1.json"

config_path = os.path.join(
    ROOT_DIR,
    "configs",
    task_name,
    file_name
)
config = ExperimentConfig(config_path)
env = EnvironmentFactory.register(config.env_name, **config.env_config)
ModelFactory.register_models_from_config(config.policy_configs)


trainer_config = config.get_trainer_config()

ray.init()

print("Test mode")
run_name = "testing"
run_dir = os.path.join(config.logdir, run_name)
os.makedirs(run_dir, exist_ok=True)
experiment_logdir = os.path.join(
    run_dir,
    "_".join((config.trial_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))),
)
trainer_config["explore"] = False
trainer_config["num_workers"] = 0
trainer = config.trainer_cls(
    config=trainer_config,
    logger_creator=lambda config: UnifiedLogger(config, experiment_logdir),
)

if config.restore_checkpoint_path is not None:
    trainer.restore(config.restore_checkpoint_path)

env = trainer.workers.local_worker().env

num_episodes = 100
for episode_idx in range(num_episodes):
    episode_reward = 0
    obs = env.reset()
    done = False
    step_idx = 0
    while not done:
        # frame = env.sim.render(width=5000, height=5000, mode='offscreen')
        # frames.append(frame[::-1,:,:])
        action = trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action) # take a random action
        episode_reward += reward
        step_idx += 1
        obs_dict = env.get_obs_dict(env.sim_obsd)
        print(obs_dict["target1_pos"])
    env.close()
    print(f"Episode {episode_idx}, reward: {episode_reward}, len: {step_idx + 1}")
    # skvideo.io.vwrite('data/myocontrol/rl-adapt/output/videos/myoHandKeyTurnFixed_randomPolicy.mp4', np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
