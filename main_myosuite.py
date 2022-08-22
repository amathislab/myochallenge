import os
import shutil
import ray
import sys
from datetime import datetime
from ray.tune import tune
from definitions import ROOT_DIR
from src.envs.environment_factory import EnvironmentFactory
from src.helpers.experiment_config import ExperimentConfig
from src.models.model_factory import ModelFactory
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


task_name = sys.argv[1]
file_name = sys.argv[2]
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

print("Train mode: optimizing the policy")
run_name = os.path.join("training", datetime.today().strftime("%Y-%m-%d"))
run_dir = os.path.join(ROOT_DIR, config.logdir, run_name)
trial_dirname = "_".join(
    (config.trial_name, datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
)

# Copy the configuration and the main files in the out dir for reproducibility
experiment_logdir = os.path.join(run_dir, trial_dirname)
out_path = "_".join((experiment_logdir, "data"))
os.makedirs(out_path, exist_ok=True)
shutil.copy(os.path.abspath(__file__), out_path)
shutil.copy(config_path, out_path)

# Start the training with ray tune
res = tune.run(
    config.trainer_cls,
    name=run_name,
    stop={"episodes_total": config.episodes_total},
    config=trainer_config,
    local_dir=os.path.join(ROOT_DIR, config.logdir),
    trial_dirname_creator=lambda _: trial_dirname,
    checkpoint_freq=config.checkpoint_freq,
    checkpoint_at_end=True,
    keep_checkpoints_num=100,
    restore=config.restore_checkpoint_path,
)
