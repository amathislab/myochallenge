from collections import deque

import gym
import numpy as np


class DictObsMixin:
    """This class enhances the observation by including a history of transitions"""

    def _init_addon(self, obs_mode, include_adapt_state, num_memory_steps):
        """Function to augment an environment's init function by changing
        the observation space and the containers for the transition history.

        Args:
            include_adapt_state (bool): whether to include the transition history
            num_memory_steps (int): number of transitions to include in the state
        """
        obs_dict = self.get_obs_dict(self.sim)
        self._include_adapt_state = include_adapt_state
        self.obs_mode = obs_mode
        if include_adapt_state:
            self.num_memory_steps = num_memory_steps
            self._obs_prev_list = deque(
                [], maxlen=num_memory_steps
            )  # Storing observations from newest to oldest
        else:
            self.num_memory_steps = 0

        if obs_mode == "array":
            obs_len_no_adapt = np.prod(self.observation_space.shape)
            self.observation_space = gym.spaces.Box(
                -20, 20, shape=(obs_len_no_adapt * (1 + num_memory_steps),)
            )
        elif obs_mode == "dict":
            obs_boxes_dict = {
                key: gym.spaces.Box(
                    -20, 20, shape=(1 + self.num_memory_steps, *np.shape(value))
                )
                for key, value in obs_dict.items()
            }
            obs_space = gym.spaces.Dict(obs_boxes_dict)
            self.observation_space = obs_space
        else:
            raise ValueError("Unknown observation mode: ", obs_mode)

    def create_history_reset_state(self, obs_dict):
        """Function to augment the state returned by the reset of a myosuite environment

        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the same keys of obs dict, if dict mode, otherwise an array with
            the values of the dict
        """
        if self._include_adapt_state:
            self._obs_prev_list.clear()
            self._obs_prev_list.appendleft(obs_dict)
            
        return_dict = {}
        for key, obs_value in obs_dict.items():
            return_value = np.zeros((1 + self.num_memory_steps, *(obs_value.shape)))
            return_value[-1, :] = obs_value
            return_dict[key] = return_value
        if self.obs_mode == "array":
            obs = self.dict2vec(return_dict, self.obs_keys)
            return obs
        if self.obs_mode == "dict":
            return return_dict

    def create_history_step_state(self, obs_dict):
        """Function to augment the state returned by the step function of a myosuite environment
        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the same keys of obs dict, if dict mode, otherwise an array with
            the values of the dict
        """
        return_dict = {}
        for key, obs_value in obs_dict.items():
            obs_value = np.atleast_1d(obs_value)
            return_value = np.zeros((1 + self.num_memory_steps, *(obs_value.shape)))
            return_value[-1, :] = obs_value
            if self._include_adapt_state:
                for idx, past_obs in enumerate(self._obs_prev_list):
                    return_value[self.num_memory_steps - idx - 1, :] = past_obs[key]
            return_dict[key] = return_value
        if self._include_adapt_state:
            self._obs_prev_list.appendleft(obs_dict)
        if self.obs_mode == "array":
            obs = self.dict2vec(return_dict, self.obs_keys)
            return obs
        if self.obs_mode == "dict":
            return return_dict
    
    @staticmethod
    def dict2vec(dic, keys):
        # recover vec
        vec = np.zeros(0)
        for key in keys:
            vec = np.concatenate([vec, dic[key].ravel()]) # ravel helps with images
        return vec
