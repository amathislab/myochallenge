import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EvaluateLSTM(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, eval_freq, eval_env, name, num_episodes=20, verbose=0):
        super(EvaluateLSTM, self).__init__()
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.name = name
        self.num_episodes = num_episodes

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.num_timesteps%self.eval_freq == 0:

            perfs = []
            for _ in range(self.num_episodes):
                lstm_states, cum_rew, step = None , 0 , 0 
                obs = self.eval_env.reset()
                episode_starts = np.ones((1,), dtype=bool)
                done = False
                while not done:
                    action, lstm_states = self.model.predict(
                        self.training_env.normalize_obs(obs),
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    obs, rewards, done, _ = self.eval_env.step(action)
                    episode_starts = done
                    cum_rew += rewards
                    step += 1
                perfs.append(cum_rew)

            self.logger.record(self.name, np.mean(perfs))
        return True

