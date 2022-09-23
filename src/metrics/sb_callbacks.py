import os
from stable_baselines3.common.callbacks import BaseCallback


class EnvDumpCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        
    def _on_step(self):
        env_path = os.path.join(self.save_path, "training_env.pkl")
        if self.verbose > 0:
            print("Saving the training environment to path ", env_path)
        self.training_env.save(env_path)
        return True