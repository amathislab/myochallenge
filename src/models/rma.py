from gym.spaces.tuple import Tuple
import gym
import numpy as np
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC


class OraclePolicyModel(TorchModelV2, nn.Module):
    """
    Policy model with an environment encoder (Oracle), as described in the Rapid Motor
    Adaptation paper (https://arxiv.org/abs/2107.04034), but adapte_ted for the SAC algorithm.
    """

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
            model_config (ModelConfigDict): hyperparameters of the policy model
            (see examples in the configs folder)
            name (str): name of the model
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation = model_config["fcnet_activation"]

        a_space = obs_space.original_space["a_t"]
        e_space = obs_space.original_space["e_t"]
        x_space = obs_space.original_space["x_t"]

        # Define the encoder
        encoder_hiddens = model_config["encoder_hiddens"]
        encoder_layers = []

        prev_layer_size = np.prod(e_space.shape)
        for size in encoder_hiddens:
            encoder_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._encoder_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=model_config["encoding_size"],
            activation_fn="tanh",
        )
        self._encoder_hidden_layers = nn.Sequential(*encoder_layers)

        # Define the policy
        policy_hiddens = model_config["policy_hiddens"]
        policy_layers = []

        prev_layer_size = (
            model_config["encoding_size"]
            + np.prod(x_space.shape)
            + np.prod(a_space.shape)
        )
        for size in policy_hiddens:
            policy_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._policy_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            activation_fn=None,  # TODO: check whether it is better to go through a tanh
        )
        self._policy_hidden_layers = nn.Sequential(*policy_layers)

        self._encoding = None
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Applies the network on an input state dict

        Args:
            input_dict (dict): input state. It requires the structue_tre
            {"obs": {
                "x_t": current state,
                "a_t": previous action,
                "e_t": raw perturbation,
            }}
            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (action mean and std, input state)
        """
        x_t = input_dict["obs"]["x_t"]
        a_t = input_dict["obs"]["a_t"]
        e_t = input_dict["obs"]["e_t"]
        self._encoding = self._encoder_logits(self._encoder_hidden_layers(e_t))
        policy_input = torch.cat([x_t, a_t, self._encoding], 1)
        self._features = self._policy_hidden_layers(policy_input)
        logits = self._policy_logits(self._features)
        return logits, state


class OracleQModel(TorchModelV2, nn.Module):
    """
    Q model with an environment encoder (Oracle), as described in the Rapid Motor Adaptation
    paper (https://arxiv.org/abs/2107.04034), but adapted for the SAC algorithm.
    """

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
            num_outputs (int): number of output q values (1)
            model_config (ModelConfigDict): hyperparameters of the q model
            (see examples in the configs folder)
            name (str): name of the model
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation = model_config["fcnet_activation"]
        a_space, e_space, x_space = self.unpack_obs_space(obs_space)
        
        # Define the encoder
        encoder_hiddens = model_config["encoder_hiddens"]
        encoder_layers = []

        prev_layer_size = np.prod(e_space.shape)
        for size in encoder_hiddens:
            encoder_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._encoder_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=model_config["encoding_size"],
            activation_fn="tanh",
        )
        self._encoder_hidden_layers = nn.Sequential(*encoder_layers)

        # Define the q network
        q_hiddens = model_config["q_hiddens"]
        q_layers = []

        prev_layer_size = (
            model_config["encoding_size"]
            + np.prod(x_space.shape)
            + 2 * np.prod(a_space.shape)
        )
        for size in q_hiddens:
            q_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._q_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            activation_fn=None,
        )
        self._q_hidden_layers = nn.Sequential(*q_layers)

        self._encoding = None
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Computes the q value from an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": (a_t, e_t, x_t, a_next)}
            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (q_value, input state)
        """
        x_t = input_dict["obs"][2]
        a_t = input_dict["obs"][0]
        e_t = input_dict["obs"][1]
        a_next = input_dict["obs"][3]
        self._encoding = self._encoder_logits(self._encoder_hidden_layers(e_t))
        policy_input = torch.cat([x_t, a_t, self._encoding, a_next], 1)
        self._features = self._q_hidden_layers(policy_input)
        logits = self._q_logits(self._features)
        return logits, state

    def unpack_obs_space(self, obs_space):
        a_space = obs_space[0]
        e_space = obs_space[1]
        x_space = obs_space[2]
        return a_space, e_space, x_space


class TCNPolicyModel(TorchModelV2, nn.Module):
    """
    Policy model with a TCN network working as an adaptation module, as described in the
    Rapid Motor Adaptation paper (https://arxiv.org/abs/2107.04034), but adapted for the
    SAC algorithm.
    """

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
            model_config (ModelConfigDict): hyperparameters of the policy model
            (see examples in the configs folder)
            name (str): name of the model
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation_fn = model_config["fcnet_activation"]

        x_prev_space = obs_space.original_space[
            "x_prev"
        ]  # Matrix with the states in the last seconds
        a_space = obs_space.original_space["a_t"]  # Last action
        x_space = obs_space.original_space["x_t"]  # Current state

        # Define the adaptation module phi
        adapt_fcnet_hiddens = model_config["adapt_fcnet_hiddens"]
        adapt_convnet_params = model_config["adapt_convnet_params"]

        adapt_fc_layers = []
        prev_layer_size = np.prod(a_space.shape) + np.prod(x_space.shape)
        for size in adapt_fcnet_hiddens:
            linear_layer = nn.Linear(prev_layer_size, size)
            adapt_fc_layers.append(linear_layer)
            activation = get_activation_fn(activation_fn, framework="torch")
            adapt_fc_layers.append(activation())
            prev_layer_size = size

        adapt_conv_layers = []

        in_channels = prev_layer_size
        seq_len = np.prod(x_prev_space.shape) // np.prod(x_space.shape)
        for layer_params in adapt_convnet_params:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_params["num_filters"],
                kernel_size=layer_params["kernel_size"],
                stride=layer_params["stride"],
            )
            adapt_conv_layers.append(conv_layer)
            activation = get_activation_fn(activation_fn, framework="torch")
            adapt_conv_layers.append(activation())
            in_channels = layer_params["num_filters"]
            seq_len = int(
                np.floor(
                    (seq_len - layer_params["kernel_size"]) / layer_params["stride"] + 1
                )
            )

        flatten = nn.Flatten()
        output_layer = nn.Linear(
            in_features=seq_len * in_channels,
            out_features=model_config["encoding_size"],
        )
        self._adapt_fc_layers = nn.Sequential(*adapt_fc_layers)
        self._adapt_conv_layers = nn.Sequential(*adapt_conv_layers)
        self._encoder_logits = nn.Sequential(flatten, output_layer)

        # Define the policy
        policy_hiddens = model_config["policy_hiddens"]
        policy_layers = []

        prev_layer_size = (
            model_config["encoding_size"]
            + np.prod(x_space.shape)
            + np.prod(a_space.shape)
        )
        for size in policy_hiddens:
            policy_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._policy_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            activation_fn=None,  # TODO: check whether it is better to go through a tanh
        )
        self._policy_hidden_layers = nn.Sequential(*policy_layers)

        self._encoding = None
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Computes the action from an input state dict

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
        x_t = input_dict["obs"]["x_t"]
        a_t = input_dict["obs"]["a_t"]
        x_prev = input_dict["obs"]["x_prev"].reshape((x_t.shape[0], -1, x_t.shape[1]))
        a_prev = input_dict["obs"]["a_prev"].reshape((a_t.shape[0], -1, a_t.shape[1]))
        adapt_input = torch.cat((x_prev, a_prev), 2)
        adapt_input = self._adapt_fc_layers(adapt_input).transpose(1, 2)
        adapt_input = self._adapt_conv_layers(adapt_input)
        self._encoding = self._encoder_logits(adapt_input)
        policy_input = torch.cat((x_t, a_t, self._encoding), 1)
        self._features = self._policy_hidden_layers(policy_input)
        logits = self._policy_logits(self._features)
        return logits, state


class OracleQAdaptModel(OracleQModel):
    """
    Q model with an environment encoder (Oracle), as described in the Rapid Motor Adaptation
    paper (https://arxiv.org/abs/2107.04034). It is the same as OracleQModel, but it works
    with the state dictionary returned by the RMA environments when it also includes the
    transition history. In fact, SAC processes such dictionary returning a tuple, so the
    positions of the state components (x_t, a_t, x_prev, ...) change, as they are sorted
    differently. Use this model to have an Oracle Q function, but with a TCN or DMAP policy.
    """

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Computes the q value from an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": (_, a_t, e_t, _, x_t, a_next)}
            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (q_value, input state)
        """
        x_t = input_dict["obs"][4]
        a_t = input_dict["obs"][1]
        e_t = input_dict["obs"][2]
        a_next = input_dict["obs"][5]
        self._encoding = self._encoder_logits(self._encoder_hidden_layers(e_t))
        policy_input = torch.cat([x_t, a_t, self._encoding, a_next], 1)
        self._features = self._q_hidden_layers(policy_input)
        logits = self._q_logits(self._features)
        return logits, state

    def unpack_obs_space(self, obs_space):
        #print("========================rma.py line392==========================")
        #print(obs_space)
        a_space = obs_space[1]
        e_space = obs_space[2]
        x_space = obs_space[4]
        '''
        a_space = obs_space[0]["a_t"]
        e_space = obs_space[0]["e_t"]
        x_space = obs_space[0]["x_t"]
        '''
        return a_space, e_space, x_space


class TCNQModel(TorchModelV2, nn.Module):
    """
    Q model with the TCN adaptation module, following the same architecture as the policy
    model, which attempts to replace the environment representation
    """

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
            num_outputs (int): number of output q values (1)
            model_config (ModelConfigDict): hyperparameters of the q network
            (see examples in the configs folder)
            name (str): name of the model
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        activation_fn = model_config["fcnet_activation"]

        x_prev_space = obs_space[3]  # Matrix with the states in the last 0.5 sec
        a_space = obs_space[1]  # Last action
        x_space = obs_space[4]  # Current state

        # Define the adaptation module phi
        adapt_fcnet_hiddens = model_config["adapt_fcnet_hiddens"]
        adapt_convnet_params = model_config["adapt_convnet_params"]

        adapt_fc_layers = []
        prev_layer_size = np.prod(a_space.shape) + np.prod(x_space.shape)
        for size in adapt_fcnet_hiddens:
            linear_layer = nn.Linear(prev_layer_size, size)
            adapt_fc_layers.append(linear_layer)
            activation = get_activation_fn(activation_fn, framework="torch")
            adapt_fc_layers.append(activation())
            prev_layer_size = size

        adapt_conv_layers = []

        in_channels = prev_layer_size
        seq_len = np.prod(x_prev_space.shape) // np.prod(x_space.shape)
        for layer_params in adapt_convnet_params:
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_params["num_filters"],
                kernel_size=layer_params["kernel_size"],
                stride=layer_params["stride"],
            )
            adapt_conv_layers.append(conv_layer)
            activation = get_activation_fn(activation_fn, framework="torch")
            adapt_conv_layers.append(activation())
            in_channels = layer_params["num_filters"]
            seq_len = int(
                np.floor(
                    (seq_len - layer_params["kernel_size"]) / layer_params["stride"] + 1
                )
            )

        flatten = nn.Flatten()
        output_layer = nn.Linear(
            in_features=seq_len * in_channels,
            out_features=model_config["encoding_size"],
        )
        self._adapt_fc_layers = nn.Sequential(*adapt_fc_layers)
        self._adapt_conv_layers = nn.Sequential(*adapt_conv_layers)
        self._encoder_logits = nn.Sequential(flatten, output_layer)

        # Define the q network
        q_hiddens = model_config["q_hiddens"]
        q_layers = []

        prev_layer_size = (
            model_config["encoding_size"]
            + np.prod(x_space.shape)
            + 2 * np.prod(a_space.shape)
        )
        for size in q_hiddens:
            q_layers.append(
                SlimFC(in_size=prev_layer_size, out_size=size, activation_fn=activation)
            )
            prev_layer_size = size

        self._q_logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            activation_fn=None,  # TODO: check whether it is better to go through a tanh
        )
        self._q_hidden_layers = nn.Sequential(*q_layers)

        self._encoding = None
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, _):
        """Computes the q value from an input state dict

        Args:
            input_dict (dict): input state. It requires the structure
            {"obs": (a_prev, a_t, _, x_prev, x_t, a_next)}

            state (object): unused parameter, forwarded as a return value
            _ (object): unused parameter

        Returns:
            tuple(torch.Tensor, object): (q_value, input state)
        """
        x_t = input_dict["obs"][4]
        a_t = input_dict["obs"][1]
        x_prev = input_dict["obs"][3].reshape((x_t.shape[0], -1, x_t.shape[1]))
        a_prev = input_dict["obs"][0].reshape((a_t.shape[0], -1, a_t.shape[1]))
        a_next = input_dict["obs"][5]
        adapt_input = torch.cat((x_prev, a_prev), 2)
        adapt_input = self._adapt_fc_layers(adapt_input).transpose(1, 2)
        adapt_input = self._adapt_conv_layers(adapt_input)
        self._encoding = self._encoder_logits(adapt_input)
        policy_input = torch.cat([x_t, a_t, self._encoding, a_next], 1)
        self._features = self._q_hidden_layers(policy_input)
        logits = self._q_logits(self._features)
        return logits, state


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
        x_space = obs_space[4]  # Current state
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
        x_t = input_dict["obs"][4]
        a_t = input_dict["obs"][1]
        a_next = input_dict["obs"][5]
        state_input = torch.cat((x_t, a_t, a_next), 1)
        return state_input
    
    
    
class SensoryTargetSimpleQModel(TorchModelV2, nn.Module):
    """Simple Q network which does not look at the past states or at the raw perturbation
    Used for the sensory-target environments.
    """

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
        x_space = obs_space[5]  # Current state
        self.x_space_size = np.prod(x_space.shape)
        self.reach_err_size = 3

        # Define the fcnet
        prev_layer_size = 2 * self.a_space_size + self.x_space_size + self.reach_err_size
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
        x_t = input_dict["obs"][5]
        a_t = input_dict["obs"][1]
        a_next = input_dict["obs"][6]
        reach_err = input_dict["obs"][3]
        state_input = torch.cat((x_t, a_t, a_next, reach_err), 1)
        return state_input
    
    
class DecisionSimpleQModel(TorchModelV2, nn.Module):

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

        a_space = obs_space[0]  # Last action
        self.a_space_size = np.prod(a_space.shape)
        x_space = obs_space[3]  # Current state
        self.x_space_size = np.prod(x_space.shape)
        self.reach_err_size = 3
        self.tip_pos_size = 3

        # Define the fcnet
        prev_layer_size = 2 * self.a_space_size + self.x_space_size + self.reach_err_size + self.tip_pos_size
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
        x_t = input_dict["obs"][7]
        a_t = input_dict["obs"][2]
        a_next = input_dict["obs"][8]
        reach_err = input_dict["obs"][4]
        tip_pos = input_dict["obs"][5]
        state_input = torch.cat((x_t, a_t, a_next, reach_err, tip_pos), 1)
        return state_input