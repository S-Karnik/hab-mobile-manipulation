from typing import Dict, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

# from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CustomFixedCategorical, CustomNormal
from torch.distributions import Distribution

from mobile_manipulation.utils.nn_utils import MLP
from habitat_baselines.il.models.models import build_mlp


class Net(nn.Module):
    """Base class for backbone to extract features."""

    output_size: int
    rnn_hidden_size = 0
    num_recurrent_layers = 0

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class CategoricalNet(nn.Module):
    def __init__(
        self, num_inputs: int, num_outputs: int, hidden_sizes: Sequence[int]
    ) -> None:
        super().__init__()
        self.mlp = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.linear = nn.Linear(self.mlp.output_size, num_outputs)
        self.num_outputs = num_outputs
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)
    
import torch.cuda

class MultiLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        outputs = torch.stack([layer(x) for layer in self.layers], dim=-1)
        return outputs

class BilinearCategoricalNet(nn.Module):
    def __init__(
        self, num_inputs: int, num_outputs: int, hidden_sizes: Sequence[int], pre_bilinear_size
    ) -> None:
        super().__init__()
        self.mlp_1 = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.mlp_2 = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.num_outputs = num_outputs
        self.mll_1 = MultiLinearLayer(self.mlp_1.output_size, pre_bilinear_size, num_outputs)
        self.mll_2 = MultiLinearLayer(self.mlp_2.output_size, pre_bilinear_size, num_outputs)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()

        with torch.cuda.stream(stream_1):
            x_1 = self.mlp_1(x_1)
            output_1 = self.mll_1(x_1)

        with torch.cuda.stream(stream_2):
            x_2 = self.mlp_2(x_2)
            output_2 = self.mll_2(x_2)

        torch.cuda.synchronize()  # Synchronize both streams

        y = output_1 * output_2
        y = torch.sum(y, dim=-2)
        return CustomFixedCategorical(logits=y)

class ClassifierNet(nn.Module):
    def __init__(self, num_inputs, hidden_sizes) -> None:
        super().__init__()
        self.mlp = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.linear = nn.Linear(self.mlp.output_size, 1)
        self.lin_seq = self.linear
        self.sigmoid = nn.Sigmoid()
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor):
        # x = self.mlp(x)
        x = self.lin_seq(x)
        return self.sigmoid(x)

class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_sizes: Sequence[int],
        action_activation: str,
        std_transform: str,
        min_std: float,
        max_std: float,
        conditioned_std: bool,
        std_init_bias: float,
        ddpg_init = None,
        # TODO(jigu): remove deprecated keys in ckpt
        **kwargs
    ) -> None:
        super().__init__()

        assert action_activation in ["", "tanh", "sigmoid"], action_activation
        self.action_activation = action_activation
        assert std_transform in ["log", "softplus"], std_transform
        self.std_transform = std_transform

        self.min_std = min_std
        self.max_std = max_std
        self.conditioned_std = conditioned_std

        self.mlp = MLP(num_inputs, hidden_sizes).orthogonal_()

        self.mu = nn.Linear(self.mlp.output_size, num_outputs)
        if ddpg_init is None:
            nn.init.orthogonal_(self.mu.weight, gain=0.01)
        else:
            self.mu.weight.data.uniform_(-ddpg_init, ddpg_init)
        nn.init.constant_(self.mu.bias, 0)

        if conditioned_std:
            self.std = nn.Linear(self.mlp.output_size, num_outputs)
            nn.init.orthogonal_(self.std.weight, gain=0.01)
            nn.init.constant_(self.std.bias, std_init_bias)
        else:
            self.std = nn.Parameter(
                torch.zeros([num_outputs]), requires_grad=True
            )
            nn.init.constant_(self.std.data, std_init_bias)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)

        mu = self.mu(x)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)
        elif self.action_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        std = self.std(x) if self.conditioned_std else self.std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        if self.std_transform == "log":
            std = torch.exp(std)
        elif self.std_transform == "softplus":
            std = F.softplus(std)

        return CustomNormal(mu, std)

class BilinearGaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_sizes: Sequence[int],
        action_activation: str,
        std_transform: str,
        min_std: float,
        max_std: float,
        conditioned_std: bool,
        std_init_bias: float = -1,
        pre_bilinear_size: int = 64,
        # TODO(jigu): remove deprecated keys in ckpt
        **kwargs
    ) -> None:
        super().__init__()

        assert action_activation in ["", "tanh", "sigmoid"], action_activation
        self.action_activation = action_activation
        assert std_transform in ["log", "softplus"], std_transform
        self.std_transform = std_transform

        self.min_std = min_std
        self.max_std = max_std
        self.conditioned_std = conditioned_std

        self.mlp_1 = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.mlp_2 = MLP(num_inputs, hidden_sizes).orthogonal_()
        self.mll_1 = MultiLinearLayer(self.mlp_1.output_size, pre_bilinear_size, num_outputs)
        self.mll_2 = MultiLinearLayer(self.mlp_2.output_size, pre_bilinear_size, num_outputs)
        self.num_outputs = num_outputs
        self.pre_bilinear_size = pre_bilinear_size

        if conditioned_std:
            self.std = nn.Linear(pre_bilinear_size, num_outputs)
            nn.init.orthogonal_(self.std.weight, gain=0.01)
            nn.init.constant_(self.std.bias, std_init_bias)
        else:
            self.std = nn.Parameter(
                torch.zeros([num_outputs]), requires_grad=True
            )
            nn.init.constant_(self.std.data, std_init_bias)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor):
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()

        with torch.cuda.stream(stream_1):
            x_1 = self.mlp_1(x_1)
            output_1 = self.mll_1(x_1)

        with torch.cuda.stream(stream_2):
            x_2 = self.mlp_2(x_2)
            output_2 = self.mll_2(x_2)

        torch.cuda.synchronize()  # Synchronize both streams
        
        # TODO: assert that dim 0 of output_1 is equal to the batch size
        y = output_1 * output_2
        y = torch.sum(y, dim=-2)
        mu = y.to(x_1.device)
        if self.action_activation == "tanh":
            mu = torch.tanh(mu)
        elif self.action_activation == "sigmoid":
            mu = torch.sigmoid(mu)

        std = self.std(x_1*x_2) if self.conditioned_std else self.std
        std = torch.clamp(std, min=self.min_std, max=self.max_std)
        if self.std_transform == "log":
            std = torch.exp(std)
        elif self.std_transform == "softplus":
            std = F.softplus(std)

        return CustomNormal(mu, std)


class CriticHead(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = ()):
        super().__init__()

        self.mlp = MLP(input_size, hidden_sizes).orthogonal_()

        self.fc = nn.Linear(self.mlp.output_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return self.fc(x)


class BilinearCriticHead(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (), pre_bilinear_size: int = 64):
        super().__init__()

        self.mlp_1 = MLP(input_size, hidden_sizes).orthogonal_()
        self.mlp_2 = MLP(input_size, hidden_sizes).orthogonal_()

        self.fc_1 = nn.Linear(self.mlp_1.output_size, pre_bilinear_size)
        self.fc_2 = nn.Linear(self.mlp_2.output_size, pre_bilinear_size)
        nn.init.orthogonal_(self.fc_1.weight)
        nn.init.constant_(self.fc_1.bias, 0)
        nn.init.orthogonal_(self.fc_2.weight)
        nn.init.constant_(self.fc_2.bias, 0)

    def forward(self, x_1, x_2):
        x_1 = self.mlp_1(x_1)
        x_1 = self.fc_1(x_1)

        x_2 = self.mlp_2(x_2)
        x_2 = self.fc_2(x_2)

        return torch.sum(x_1 * x_2, dim=-1,keepdim=True)


class ActorCritic(nn.Module):
    r"""Base class for actor-critic policy."""

    def __init__(self, net: Net, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.net = net
        self.actor = actor
        self.critic = critic

    def act(self, batch: Dict[str, torch.Tensor], deterministic=False, include_entropy=False):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        distribution: Distribution = self.actor(features)
        value: torch.Tensor = self.critic(features)

        if deterministic:
            if isinstance(distribution, CustomFixedCategorical):
                action = distribution.mode()
            elif isinstance(distribution, CustomNormal):
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        if isinstance(distribution, CustomFixedCategorical):
            all_action_log_probs = torch.log(distribution.probs).flatten()
        else:
            all_action_log_probs = torch.tensor(0).to(device=action.device)
        
        output = dict(
                action=action,
                action_log_probs=action_log_probs,
                all_action_log_probs=all_action_log_probs,
                entropy=distribution.entropy().cpu(),
                value=value,
                rnn_hidden_states=rnn_hidden_states,
            )
        
        if include_entropy:
            output["entropy"] = distribution.entropy().cpu()
                    
        return output

    def get_value(self, batch: Dict[str, torch.Tensor]):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        return self.critic(features)

    def evaluate_actions(
        self, batch: Dict[str, torch.Tensor], action: torch.Tensor
    ):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        distribution: Distribution = self.actor(features)
        value: torch.Tensor = self.critic(features)

        action_log_probs = distribution.log_probs(action)  # [B, 1]
        dist_entropy = distribution.entropy()  # [B, 1]

        return dict(
            action_log_probs=action_log_probs,
            dist_entropy=dist_entropy,
            value=value,
            rnn_hidden_states=rnn_hidden_states,
        )

    @classmethod
    def build_gaussian_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Box), action_space
        assert len(action_space.shape) == 1, action_space.shape
        actor = GaussianNet(num_inputs, action_space.shape[0], **kwargs)
        return actor

    @classmethod
    def build_categorical_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Discrete), action_space
        actor = CategoricalNet(num_inputs, action_space.n, **kwargs)
        return actor
    
class BilinearActorCritic(nn.Module):
    r"""Base class for bilinear actor-critic policy."""

    def __init__(self, net: Net, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.net = net
        self.actor = actor
        self.critic = critic

    def act(self, batch: Dict[str, torch.Tensor], deterministic=False, include_entropy=False):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        goal_dep_features = net_outputs["goal_dep_features"]
        goal_ind_features = net_outputs["goal_ind_features"]
        goal_dep_rnn_hidden_states = net_outputs["goal_dep_rnn_hidden_states"]
        goal_ind_rnn_hidden_states = net_outputs["goal_ind_rnn_hidden_states"]

        distribution: Distribution = self.actor(goal_dep_features, goal_ind_features)
        value: torch.Tensor = self.critic(goal_dep_features, goal_ind_features)

        if deterministic:
            if isinstance(distribution, CustomFixedCategorical):
                action = distribution.mode()
            elif isinstance(distribution, CustomNormal):
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        
        output = dict(
                action=action,
                action_log_probs=action_log_probs,
                entropy=distribution.entropy().cpu(),
                value=value,
                goal_dep_rnn_hidden_states=goal_dep_rnn_hidden_states,
                goal_ind_rnn_hidden_states=goal_ind_rnn_hidden_states,
            )
        
        if include_entropy:
            output["entropy"] = distribution.entropy().cpu()
                    
        return output

    def get_value(self, batch: Dict[str, torch.Tensor]):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        goal_dep_features = net_outputs["goal_dep_features"]
        goal_ind_features = net_outputs["goal_ind_features"]
        return self.critic(goal_dep_features, goal_ind_features)

    def evaluate_actions(
        self, batch: Dict[str, torch.Tensor], action: torch.Tensor
    ):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        goal_dep_features = net_outputs["goal_dep_features"]
        goal_ind_features = net_outputs["goal_ind_features"]        
        goal_dep_rnn_hidden_states = net_outputs["goal_dep_rnn_hidden_states"]
        goal_ind_rnn_hidden_states = net_outputs["goal_ind_rnn_hidden_states"]

        distribution: Distribution = self.actor(goal_dep_features, goal_ind_features)
        value: torch.Tensor = self.critic(goal_dep_features, goal_ind_features)

        action_log_probs = distribution.log_probs(action)  # [B, 1]
        dist_entropy = distribution.entropy()  # [B, 1]

        return dict(
            action_log_probs=action_log_probs,
            dist_entropy=dist_entropy,
            value=value,
            goal_dep_rnn_hidden_states=goal_dep_rnn_hidden_states,
            goal_ind_rnn_hidden_states=goal_ind_rnn_hidden_states,
        )

    @classmethod
    def build_gaussian_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Box), action_space
        assert len(action_space.shape) == 1, action_space.shape
        actor = BilinearGaussianNet(num_inputs, action_space.shape[0], **kwargs)
        return actor

    @classmethod
    def build_categorical_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Discrete), action_space
        actor = BilinearCategoricalNet(num_inputs, action_space.n, **kwargs)
        return actor

class MultiHeadActorCritic(nn.Module):
    r"""Base class for actor-critic policy."""

    def __init__(self, net: Net, gaussian_actor: nn.Module, categorical_actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.net = net
        self.gaussian_actor = gaussian_actor
        self.categorical_actor = categorical_actor
        self.critic = critic

    def act(self, batch: Dict[str, torch.Tensor], deterministic=False):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        gaussian_distribution: Distribution = self.gaussian_actor(features)
        categorical_distribution: Distribution = self.categorical_actor(features)
        value: torch.Tensor = self.critic(features)

        if deterministic:
            categorical_actions = categorical_distribution.mode()
            gaussian_actions = gaussian_distribution.mean
        else:
            categorical_actions = categorical_distribution.sample()
            gaussian_actions = gaussian_distribution.sample()

        gaussian_action_log_probs = gaussian_distribution.log_probs(gaussian_actions)
        categorical_action_log_probs = categorical_distribution.log_probs(categorical_actions)
        categorical_all_log_probs = torch.log(categorical_distribution.probs)

        return dict(
            gaussian_action=gaussian_actions,
            categorical_action=categorical_actions,
            gaussian_action_log_probs=gaussian_action_log_probs,
            categorical_action_log_probs=categorical_action_log_probs,
            categorical_all_log_probs=categorical_all_log_probs,
            value=value,
            rnn_hidden_states=rnn_hidden_states,
        )

    def get_value(self, batch: Dict[str, torch.Tensor]):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        return self.critic(features)

    def evaluate_actions(
        self, batch: Dict[str, torch.Tensor], gaussian_actions: torch.Tensor, categorical_actions: torch.Tensor
    ):
        net_outputs: Dict[str, torch.Tensor] = self.net(batch)
        features = net_outputs["features"]
        rnn_hidden_states = net_outputs["rnn_hidden_states"]

        gaussian_distribution: Distribution = self.gaussian_actor(features)
        categorical_distribution: Distribution = self.categorical_actor(features)
        value: torch.Tensor = self.critic(features)

        gaussian_action_log_probs = gaussian_distribution.log_probs(gaussian_actions)  # [B, 1]
        categorical_action_log_probs = categorical_distribution.log_probs(categorical_actions)  # [B, 1]
        gaussian_dist_entropy = gaussian_distribution.entropy()  # [B, 1]
        categorical_dist_entropy = categorical_distribution.entropy()  # [B, 1]
        return dict(
            gaussian_action_log_probs=gaussian_action_log_probs,
            categorical_action_log_probs=categorical_action_log_probs,
            gaussian_dist_entropy=gaussian_dist_entropy,
            categorical_dist_entropy=categorical_dist_entropy,
            value=value,
            rnn_hidden_states=rnn_hidden_states,
        )

    @classmethod
    def build_gaussian_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Box), action_space
        assert len(action_space.shape) == 1, action_space.shape
        actor = GaussianNet(num_inputs, action_space.shape[0], **kwargs)
        return actor

    @classmethod
    def build_categorical_actor(self, num_inputs, action_space, **kwargs):
        assert isinstance(action_space, spaces.Discrete), action_space
        actor = CategoricalNet(num_inputs, action_space.n, **kwargs)
        return actor
