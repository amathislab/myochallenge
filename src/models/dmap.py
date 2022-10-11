from re import L
from turtle import forward
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.models.torch.misc import SlimFC


class DMAPModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """Init function

        Args:
            obs_space (gym.spaces.Space): observation space
            action_space (gym.spaces.Space): action space
            num_outputs (int): twice the action size
            model_config (ModelConfigDict): definition of the hyperparameters of the model
            (see default configs for examples)
            name (str): name of the model
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.activation_fn = model_config["fcnet_activation"]

        self.x_space_size, self.a_space_size, x_prev_space_size = self.get_space_sizes(
            obs_space
        )

        self.embedding_size = model_config["embedding_size"]

        # Define the network to extract features from each input channel (element of the state)
        feature_convnet_params = model_config["feature_convnet_params"]
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
            activation = get_activation_fn(self.activation_fn, framework="torch")
            feature_conv_layers.append(activation())
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
        feature_fcnet_hiddens = model_config["feature_fcnet_hiddens"]

        prev_layer_size = seq_len * in_channels
        feature_fc_layers = []
        for size in feature_fcnet_hiddens:
            linear_layer = nn.Linear(prev_layer_size, size)
            feature_fc_layers.append(linear_layer)
            activation = get_activation_fn(self.activation_fn, framework="torch")
            feature_fc_layers.append(activation())
            prev_layer_size = size

        self._feature_fc_layers = nn.Sequential(
            flatten_time_and_channels, *feature_fc_layers
        )

        self._key_readout = nn.Linear(
            in_features=prev_layer_size, out_features=self.num_output_networks
        )
        self._value_readout = nn.Linear(
            in_features=prev_layer_size, out_features=self.embedding_size
        )

        # Define the policy networks
        policy_hiddens = model_config.get("policy_hiddens")
        q_hiddens = model_config.get("q_hiddens")
        if policy_hiddens is not None:
            assert q_hiddens is None, "Is this a policy network or a q network?"
            self.init_output_networks(policy_hiddens)
        elif q_hiddens is not None:
            assert policy_hiddens is None, "Is this a policy network or a q network?"
            self.init_output_networks(q_hiddens)
        else:
            raise ValueError(
                "The config must either specify policy_hiddens or q_hiddens"
            )

    def get_adapt_and_state_input(self, input_dict):
        """Processes the input dict to assemble the state history and the current state

        Args:
            input_dict (dict): input state. It requires it to be compatible with the implementation
            of get_obs_components

        Returns:
            tuple(torch.Tensor, torch.Tensor): (state history, current state)
        """
        x_t, a_t, x_prev, a_prev, a_next = self.get_obs_components(input_dict)
        adapt_input = (
            torch.cat((x_prev, a_prev), 2)
            .transpose(1, 2)
            .reshape(np.prod(x_t.shape) + np.prod(a_t.shape), 1, -1)
        )
        if a_next is None:
            state_input = torch.cat((x_t, a_t), 1)
        else:
            state_input = torch.cat((x_t, a_t, a_next), 1)
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
            (-1, self.num_output_networks, self.a_space_size + self.x_space_size),
        )
        values = torch.reshape(
            flat_values,
            (-1, self.embedding_size, self.a_space_size + self.x_space_size),
        )
        softmax_keys = F.softmax(keys, dim=2)
        return softmax_keys, values

    @staticmethod
    def get_space_sizes(obs_space):
        raise NotImplementedError()

    def init_output_networks(self, hiddens):
        raise NotImplementedError()

    @staticmethod
    def get_obs_components(input_dict):
        raise NotImplementedError

    @property
    def num_output_networks(self):
        raise NotImplementedError()


class DMAPPolicyModel(DMAPModel):
    """Policy model of DMAP. It defines one policy network for each action component. Each policy
    network can focus on a different part of the proprioceptive state history to generate an
    embedding for decision making.
    """

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Applies the network on an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "x_prev": matrix of N previous states,
                "a_prev": matrix of N previous actioons
            }}
            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (action mean and std, input state)
        """

        adapt_input, state_input = self.get_adapt_and_state_input(input_dict)
        keys, values = self.get_keys_and_values(adapt_input)
        embedding = torch.matmul(
            keys, values.transpose(1, 2)
        )  # Shape: (batch, a_size, v_size)
        action_elements_list = []
        for i in range(self.a_space_size):
            e_i = embedding[:, i, :]
            policy_input = torch.cat(
                (state_input, e_i), 1
            )  # TODO: incorporate reach_err
            action_net = getattr(
                self, f"_policy_fcnet_{i}"
            )  # TODO: change input dims of policy_fcnet
            action_elements_list.append(action_net(policy_input))
        action_mean_list = [el[:, 0:1] for el in action_elements_list]
        action_std_list = [el[:, 1:2] for el in action_elements_list]

        return torch.cat((*action_mean_list, *action_std_list), 1), state

    @staticmethod
    def get_space_sizes(obs_space):
        x_prev_space = obs_space.original_space[
            "x_prev"
        ]  # Matrix with the states in the last seconds
        x_prev_space_size = np.prod(x_prev_space.shape)
        a_space = obs_space.original_space["a_t"]  # Last action
        a_space_size = np.prod(a_space.shape)
        x_space = obs_space.original_space["x_t"]  # Current state
        x_space_size = np.prod(x_space.shape)
        return x_space_size, a_space_size, x_prev_space_size

    def init_output_networks(self, hiddens):
        for i in range(self.num_output_networks):
            policy_layers = []
            prev_layer_size = (
                self.embedding_size + self.x_space_size + self.a_space_size
            )
            for size in hiddens:
                policy_layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        activation_fn=self.activation_fn,
                    )
                )
                prev_layer_size = size
            policy_layers.append(nn.Linear(in_features=prev_layer_size, out_features=2))
            setattr(self, f"_policy_fcnet_{i}", nn.Sequential(*policy_layers))

    @staticmethod
    def get_obs_components(input_dict):
        """Processes the input dict to return the current state, the previous action and the
        history of states and actions

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "x_prev": matrix of N previous states,
                "a_prev": matrix of N previous actioons
            }}

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            (current state, previous action, state history, action history)
        """
        x_t = input_dict["obs"]["x_t"]
        a_t = input_dict["obs"]["a_t"]
        x_prev = input_dict["obs"]["x_prev"].reshape((x_t.shape[0], -1, x_t.shape[1]))
        a_prev = input_dict["obs"]["a_prev"].reshape((a_t.shape[0], -1, a_t.shape[1]))
        return x_t, a_t, x_prev, a_prev, None

    @property
    def num_output_networks(self):
        return self.a_space_size


class LocalDMAPPolicyModel(DMAPModel):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """Init function

        Args:
            obs_space (gym.spaces.Space): observation space
            action_space (gym.spaces.Space): action space
            num_outputs (int): twice the action size
            model_config (ModelConfigDict): definition of the hyperparameters of the model
            (see default configs for examples)
            name (str): name of the model
        """
        self.connectivity_map = model_config["connectivity_map"]
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Applies the network on an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "x_prev": matrix of N previous states,
                "a_prev": m1sed parameter

        Returns:
            tuple(torch.Tensor, object): (action mean and std, input state)
        """
        adapt_input, state_input = self.get_adapt_and_state_input(input_dict)
        keys, values = self.get_keys_and_values(adapt_input)
        embedding = torch.matmul(
            keys, values.transpose(1, 2)
        )  # Shape: (batch, a_size, v_size)
        action_elements_list = []
        for i in range(self.a_space_size):
            e_i = embedding[:, i, :]
            local_state = self.get_local_state(state_input, i)
            policy_input = torch.cat((local_state, e_i), 1)
            action_net = getattr(self, f"_policy_fcnet_{i}")
            action_elements_list.append(action_net(policy_input))
        action_mean_list = [el[:, 0:1] for el in action_elements_list]
        action_std_list = [el[:, 1:2] for el in action_elements_list]

        return torch.cat((*action_mean_list, *action_std_list), 1), state

    def get_local_state(self, state_input, idx):
        """Gets the local input state of a policy network by index

        Args:
            state_input (torch.tensor): tensor concatenating the current state and the previous action
            idx (int): index of the policy network

        Returns:
            torch.tensor: local slice of the input state
        """
        # The coordinate of the state are specified in the connectivity map, we add also the local action
        # local_state_idx_list = self.connectivity_map[idx] + [self.x_space_size + idx]
        local_state_idx_list = self.connectivity_map[idx]
        local_state = state_input[:, local_state_idx_list]
        return local_state

    @staticmethod
    def get_space_sizes(obs_space):
        return DMAPPolicyModel.get_space_sizes(obs_space)

    def init_output_networks(self, hiddens):
        for i in range(self.num_output_networks):
            policy_layers = []
            local_state_size = len(self.connectivity_map[i])
            prev_layer_size = (
                self.embedding_size + local_state_size + 1
            )  # +1 for the action component
            for size in hiddens:
                policy_layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        activation_fn=self.activation_fn,
                    )
                )
                prev_layer_size = size
            policy_layers.append(nn.Linear(in_features=prev_layer_size, out_features=2))
            setattr(self, f"_policy_fcnet_{i}", nn.Sequential(*policy_layers))

    @staticmethod
    def get_obs_components(input_dict):
        return DMAPPolicyModel.get_obs_components(input_dict)

    @property
    def num_output_networks(self):
        return self.a_space_size


class DMAPQModel(DMAPModel):
    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        adapt_input, state_input = self.get_adapt_and_state_input(input_dict)
        keys, values = self.get_keys_and_values(adapt_input)
        embedding = torch.matmul(keys, values.transpose(1, 2))
        embedding = torch.squeeze(embedding)
        q_fcnet_input = torch.cat((state_input, embedding), 1)
        q_value = self._q_fcnet(q_fcnet_input)
        return q_value, state

    @staticmethod
    def get_space_sizes(obs_space):
        x_prev_space = obs_space[2]  # Matrix with the states in the last 0.5 sec
        x_prev_space_size = np.prod(x_prev_space.shape)
        a_space = obs_space[1]  # Last action
        a_space_size = np.prod(a_space.shape)
        x_space = obs_space[3]  # Current state
        x_space_size = np.prod(x_space.shape)
        return x_space_size, a_space_size, x_prev_space_size

    def init_output_networks(self, hiddens):
        q_layers = []
        prev_layer_size = (
            self.embedding_size + self.x_space_size + 2 * self.a_space_size
        )
        for size in hiddens:
            q_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    activation_fn=self.activation_fn,
                )
            )
            prev_layer_size = size
        q_layers.append(nn.Linear(in_features=prev_layer_size, out_features=1))
        self._q_fcnet = nn.Sequential(*q_layers)

    @staticmethod
    def get_obs_components(input_dict):
        x_t = input_dict["obs"][3]
        a_t = input_dict["obs"][1]
        x_prev = input_dict["obs"][2].reshape((x_t.shape[0], -1, x_t.shape[1]))
        a_prev = input_dict["obs"][0].reshape((a_t.shape[0], -1, a_t.shape[1]))
        a_next = input_dict["obs"][4]
        return x_t, a_t, x_prev, a_prev, a_next

    @property
    def num_output_networks(self):
        return 1


class NonDMAPPolicyModel(DMAPModel):
    """Policy model of Non-Distribuited MAP. It defines one single policy network.
    """

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Applies the network on an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "x_prev": matrix of N previous states,
                "a_prev": matrix of N previous actioons
            }}
            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (action mean and std, input state)
        """
        adapt_input, state_input = self.get_adapt_and_state_input(input_dict)
        keys, values = self.get_keys_and_values(adapt_input)
        embedding = torch.matmul(keys, values.transpose(1, 2))
        embedding = torch.squeeze(embedding, dim=1)
        policy_input = torch.cat((state_input, embedding), 1)
        action = self._policy_fcnet(policy_input)

        return action, state

    @staticmethod
    def get_space_sizes(obs_space):
        x_prev_space = obs_space.original_space[
            "x_prev"
        ]  # Matrix with the states in the last seconds
        x_prev_space_size = np.prod(x_prev_space.shape)
        a_space = obs_space.original_space["a_t"]  # Last action
        a_space_size = np.prod(a_space.shape)
        x_space = obs_space.original_space["x_t"]  # Current state
        x_space_size = np.prod(x_space.shape)
        return x_space_size, a_space_size, x_prev_space_size

    def init_output_networks(self, hiddens):
        policy_layers = []
        prev_layer_size = self.embedding_size + self.x_space_size + self.a_space_size
        for size in hiddens:
            policy_layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    activation_fn=self.activation_fn,
                )
            )
            prev_layer_size = size
        policy_layers.append(
            nn.Linear(in_features=prev_layer_size, out_features=2 * self.a_space_size)
        )
        self._policy_fcnet = nn.Sequential(*policy_layers)

    @staticmethod
    def get_obs_components(input_dict):
        """Processes the input dict to return the current state, the previous action and the
        history of states and actions

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "x_prev": matrix of N previous states,
                "a_prev": matrix of N previous actioons
            }}

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            (current state, previous action, state history, action history)
        """
        x_t = input_dict["obs"]["x_t"]
        a_t = input_dict["obs"]["a_t"]
        x_prev = input_dict["obs"]["x_prev"].reshape((x_t.shape[0], -1, x_t.shape[1]))
        a_prev = input_dict["obs"]["a_prev"].reshape((a_t.shape[0], -1, a_t.shape[1]))
        return x_t, a_t, x_prev, a_prev, None

    @property
    def num_output_networks(self):
        return 1


class SimpleQModel(TorchModelV2, nn.Module):
    """Simple Q network which does not look at the past states or at the raw perturbation"""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        assert num_outputs == 1
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation_fn = model_config["fcnet_activation"]
        fcnet_hiddens = model_config["fcnet_hiddens"]
        a_space = obs_space[1]  # Last action
        self.a_space_size = np.prod(a_space.shape)
        x_space = obs_space[3]  # Current state
        self.x_space_size = np.prod(x_space.shape)

        # Define the fcnet
        prev_layer_size = 2 * self.a_space_size + self.x_space_size
        q_layers = []
        for size in fcnet_hiddens:
            linear_layer = nn.Linear(prev_layer_size, size)
            q_layers.append(linear_layer)
            activation = get_activation_fn(activation_fn, framework="torch")
            q_layers.append(activation())
            prev_layer_size = size
        q_layers.append(nn.Linear(in_features=prev_layer_size, out_features=1))
        self._q_fcnet = nn.Sequential(*q_layers)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_len):
        state_input = self.get_state_input(input_dict)
        q_value = self._q_fcnet(state_input)
        return q_value, state

    def get_state_input(self, input_dict):
        x_t = input_dict["obs"][3]
        a_t = input_dict["obs"][1]
        a_next = input_dict["obs"][4]
        state_input = torch.cat((x_t, a_t, a_next), 1)
        return state_input
