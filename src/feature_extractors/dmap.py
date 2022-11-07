import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from myosuite.utils.obj_vec_dict import ObsVecDict
import torch
from torch import nn


class DmapExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, history_observation_space, embedding_size, feature_convnet_params, feature_fcnet_hiddens):
        self.observation_space = history_observation_space
        self.history_obsvecdict = ObsVecDict()
        random_obs = self.observation_space.sample()
        random_obs.update({"t": np.zeros(1)})
        self.history_obs_keys = list(random_obs.keys())
        _, random_obs_vec = self.history_obsvecdict.obsdict2obsvec(random_obs, self.history_obs_keys)
        super().__init__(self.observation_space, random_obs_vec.size)

        self.x_space_size = random_obs["x_t"].size
        self.a_space_size = random_obs["a_t"].size
        x_prev_space_size = random_obs["x_prev"].size

        self.embedding_size = embedding_size

        # Define the network to extract features from each input channel (element of the state)
        feature_conv_layers = []
        in_channels = 1
        seq_len = np.prod(x_prev_space_size) // self.x_space_size

        for layer_params in feature_convnet_params:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_params["num_filters"],
                kernel_size=layer_params["kernel_size"],
                stride=layer_params["stride"],
            )
            feature_conv_layers.append(conv_layer)
            feature_conv_layers.append(nn.ReLU())
            in_channels = layer_params["num_filters"]
            seq_len = int(
                np.floor(
                    (seq_len - layer_params["kernel_size"]) / layer_params["stride"] + 1
                )
            )
        self._feature_conv_layers = nn.Sequential(
            *feature_conv_layers
        )  # the output has shape (batch_size * state_size, in_channels, seq_len) and needs to be flattened before the MLP

        # Define the network to extract features from the cnn output of each state element
        flatten_time_and_channels = nn.Flatten()
        feature_fcnet_hiddens = feature_fcnet_hiddens

        prev_layer_size = seq_len * in_channels
        feature_fc_layers = []
        for size in feature_fcnet_hiddens:
            linear_layer = nn.Linear(prev_layer_size, size)
            feature_fc_layers.append(linear_layer)
            feature_fc_layers.append(nn.ReLU())
            prev_layer_size = size

        self._feature_fc_layers = nn.Sequential(
            flatten_time_and_channels, *feature_fc_layers
        )

        self._key_readout = nn.Linear(
            in_features=prev_layer_size, out_features=1
        )
        self._value_readout = nn.Linear(
            in_features=prev_layer_size, out_features=self.embedding_size
        )
        
    @property
    def features_dim(self) -> int:
        return self.x_space_size + self.a_space_size + self.embedding_size + 12
    
    def forward(self, observation_dict):
        observation = observation_dict["observation"][None, :]
        input_dict = self.history_obsvecdict.obsvec2obsdict(observation)
        adapt_input, state_input = self.get_adapt_and_state_input(input_dict)
        keys, values = self.get_keys_and_values(adapt_input)
        embedding = torch.squeeze(torch.matmul(keys, values.transpose(1, 2)), 1)
        out = torch.cat((state_input, embedding, observation_dict["achieved_goal"], observation_dict["desired_goal"]), 1)
        return out
    
    def get_adapt_and_state_input(self, input_dict):
        """Processes the input dict to assemble the state history and the current state

        Args:
            input_dict (dict): input state. It requires it to be compatible with the implementation
            of get_obs_components

        Returns:
            tuple(torch.Tensor, torch.Tensor): (state history, current state)
        """
        x_t = torch.squeeze(input_dict["x_t"], 0)
        a_t = torch.squeeze(input_dict["a_t"], 0)
        x_prev = input_dict["x_prev"]
        a_prev = input_dict["a_prev"]
        
        adapt_input = (
            torch.cat((x_prev, a_prev), 2)
            .transpose(1, 2)
            .reshape(np.prod(x_t.shape) + np.prod(a_t.shape), 1, -1)
        )
        state_input = torch.cat((x_t, a_t), 1)

        return adapt_input, state_input

    def get_keys_and_values(self, adapt_input):
        """Processes the state history to generate the matrices K and V (see paper for details)

        Args:
            adapt_input (torch.Tensor): (K, V)

        Returns:
            tuple(torch.Tensor, torch.Tensor): _description_
        """
        cnn_out = self._feature_conv_layers(adapt_input)
        features_out = self._feature_fc_layers(cnn_out)
        flat_keys = self._key_readout(features_out)
        flat_values = self._value_readout(features_out)
        keys = torch.reshape(
            flat_keys,
            (-1, 1, self.a_space_size + self.x_space_size),
        )
        values = torch.reshape(
            flat_values,
            (-1, self.embedding_size, self.a_space_size + self.x_space_size),
        )
        softmax_keys = nn.functional.softmax(keys, dim=2)
        return softmax_keys, values
