import gym
import numpy as np
from collections import deque


class HistoryMixin:
    """This class enhances the observation by including a history of transitions"""

    def _init_addon(self, include_adapt_state, num_memory_steps):
        """Function to augment an environment's init function by changing
        the observation space and the containers for the transition history.

        Args:
            include_adapt_state (bool): whether to include the transition history
            num_memory_steps (int): number of transitions to include in the state
        """
        obs_dict = {
            "x_t": gym.spaces.Box(-10, 10, shape=(self.obs_dim,)),
            "a_t": gym.spaces.Box(-1, 1, shape=(self.action_dim,)),
        }
        self._include_adapt_state = include_adapt_state
        if include_adapt_state:
            self.num_memory_steps = num_memory_steps
            obs_dict.update(
                {
                    "a_prev": gym.spaces.Box(
                        -1, 1, shape=(num_memory_steps * self.action_dim,)
                    ),
                    "x_prev": gym.spaces.Box(
                        -10, 10, shape=(num_memory_steps * self.obs_dim,)
                    ),
                }
            )
            self._x_prev_list = deque(
                [], maxlen=num_memory_steps
            )  # Storing states from newest to oldest
            self._a_prev_list = deque(
                [], maxlen=num_memory_steps
            )  # Storing actions from newest to oldest
        obs_space = gym.spaces.Dict(obs_dict)
        #self.robot.observation_space = obs_space
        self.observation_space = obs_space

    def create_rma_reset_state(self, x_t):
        """Function to augment the state returned by the reset of a random perturbation
        environment

        Args:
            x_t (np.array): current state

        Returns:
            dict: {"x_t": state, "a_t": action, "e_t": raw_perturbation,
            "a_prev": action_history, "x_prev": state_history}
        """
        a_t = np.zeros(self.action_dim)
        return_dict = {"x_t": x_t, "a_t": a_t}

        if self._include_adapt_state:
            self._x_prev_list.clear()
            self._a_prev_list.clear()
            return_dict.update(
                {
                    "a_prev": np.zeros((self.num_memory_steps * self.action_dim,)),
                    "x_prev": np.zeros((self.num_memory_steps * self.obs_dim,)),
                }
            )
        return return_dict

    def create_rma_step_state(self, state, action):
        """Function to augment the state returned by the step function of a random
        perturbation environment

        Args:
            state (np.array): current state
            action (np.array): last action

        Returns:
            dict: {"x_t": state, "a_t": action, "e_t": raw_perturbation,
            "a_prev": action_history, "x_prev": state_history}
        """
        x_t = state
        a_t = action
        return_dict = {"x_t": x_t, "a_t": a_t}

        if self._include_adapt_state:
            a_prev = np.zeros((self.num_memory_steps, self.action_dim))
            x_prev = np.zeros((self.num_memory_steps, self.obs_dim))
            for idx, (a, x) in enumerate(zip(self._a_prev_list, self._x_prev_list)):
                a_prev[self.num_memory_steps - 1 - idx, :] = a
                x_prev[self.num_memory_steps - 1 - idx, :] = x
            return_dict.update({"a_prev": a_prev.flatten(), "x_prev": x_prev.flatten()})
            self._a_prev_list.appendleft(a_t)
            self._x_prev_list.appendleft(x_t)
        return return_dict
