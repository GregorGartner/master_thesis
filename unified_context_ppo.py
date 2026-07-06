from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from collections import deque
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, safe_mean
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    BernoulliDistribution,
)

LOGVAR_CLAMP_MIN = -50.0
LOGVAR_CLAMP_MAX = 20.0
MIN_UNCERTAINTY_STD = math.exp(0.5 * LOGVAR_CLAMP_MIN)
MAX_UNCERTAINTY_STD = math.exp(0.5 * LOGVAR_CLAMP_MAX)


def _flatten_obs(
    obs: Union[np.ndarray, th.Tensor, Mapping[str, Any]],
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Takes whatever the env returns as observation (nunpy array, tensor, or Dict of arrays)
    and flattens it into a single numpy array of shape (n_envs, obs_dim).
    n_envs is the number of parallel envs in the VecEnv, which is usually 1.
    """

    if isinstance(obs, th.Tensor):
        obs = obs.detach().cpu().numpy()

    if isinstance(obs, np.ndarray) or not isinstance(obs, Mapping):
        arr = np.asarray(obs, dtype=dtype)
        if arr.ndim == 1:
            return arr.reshape((1, -1))
        return arr

    obs_keys = list(obs.keys())

    # Infer n_envs from the first key.
    first = np.asarray(obs[obs_keys[0]])
    if first.ndim == 0:
        n_envs = 1
    elif first.ndim == 1:
        n_envs = 1
    else:
        n_envs = int(first.shape[0])

    parts: List[np.ndarray] = []
    for key in obs_keys:
        value = np.asarray(obs[key], dtype=dtype)
        if value.ndim == 0:
            value = value.reshape((n_envs, 1))
        elif value.ndim == 1:
            value = value.reshape((n_envs, -1))
        else:
            value = value.reshape((n_envs, -1))
        parts.append(value)

    return np.concatenate(parts, axis=1)


def _flat_obs_dim_from_space(space: gym.Space) -> Tuple[int, List[str]]:
    """Return flattened observation dimension and the key order used (for Dict spaces)."""

    if isinstance(space, gym.spaces.Dict):
        keys = list(space.spaces.keys())
        dim = 0
        for k in keys:
            sub = space.spaces[k]
            if not isinstance(sub, gym.spaces.Box):
                raise NotImplementedError(f"Only Box subspaces supported in Dict obs, got {type(sub)} for key '{k}'")
            dim += int(np.prod(sub.shape))
        return dim, keys

    if isinstance(space, gym.spaces.Box):
        dim = int(np.prod(space.shape))
        return dim, []

    raise NotImplementedError(f"Unsupported observation space: {type(space)}")


# ---------- Samples that also include trajectory windows ----------

@dataclass
class ContextRolloutBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor 
    advantages: th.Tensor 
    returns: th.Tensor
    traj: th.Tensor  # shape: (batch, window_length, obs_dim + act_dim_encoded)
    true_params: th.Tensor  # shape: (batch, true_param_dim)
    policy_context: th.Tensor  # shape: (batch, context_dim)
    use_nominal_context: th.Tensor  # shape: (batch,)


# ---------- Rollout buffer that stores trajectory windows ----------

class ContextRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
        window_length: int = 8,
        traj_dim: int = 0,
        true_param_dim: int = 0,
        policy_context_dim: int = 0,
    ):
        self.window_length = int(window_length)
        self.traj_dim = int(traj_dim)
        self.true_param_dim = int(true_param_dim)
        self.policy_context_dim = int(policy_context_dim)
        self.traj = None  # allocated in reset()
        self.true_params = None  # allocated in reset()
        self.policy_context = None  # allocated in reset()
        self.use_nominal_context = None  # allocated in reset()

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        super().reset()
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.traj = np.zeros((self.buffer_size, self.n_envs, self.window_length, self.traj_dim), dtype=np.float32)
        self.true_params = np.zeros((self.buffer_size, self.n_envs, self.true_param_dim), dtype=np.float32)
        self.policy_context = np.zeros((self.buffer_size, self.n_envs, self.policy_context_dim), dtype=np.float32)
        self.use_nominal_context = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        traj_window: np.ndarray,
        true_params: np.ndarray,
        policy_context: np.ndarray,
        use_nominal_context: np.ndarray,
    ) -> None:
        # Standard SB3 add
        super().add(obs, action, reward, episode_start, value, log_prob)
        # inside supper().add(), self.pos has already been incremented, so the current transition is at self.pos - 1
        self.traj[self.pos - 1] = traj_window
        self.true_params[self.pos - 1] = true_params
        self.policy_context[self.pos - 1] = policy_context
        self.use_nominal_context[self.pos - 1] = np.asarray(use_nominal_context, dtype=np.bool_)

    def get(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling from it."

        observations = th.as_tensor(self.swap_and_flatten(self.observations).astype(np.float32, copy=False), device=self.device, dtype=th.float32)
        actions = th.as_tensor(self.swap_and_flatten(self.actions).astype(np.float32, copy=False), device=self.device, dtype=th.float32)
        old_values = th.as_tensor(self.swap_and_flatten(self.values).astype(np.float32, copy=False), device=self.device, dtype=th.float32).flatten()
        old_log_prob = th.as_tensor(self.swap_and_flatten(self.log_probs).astype(np.float32, copy=False), device=self.device, dtype=th.float32).flatten()
        advantages = th.as_tensor(self.swap_and_flatten(self.advantages).astype(np.float32, copy=False), device=self.device, dtype=th.float32).flatten()
        returns = th.as_tensor(self.swap_and_flatten(self.returns).astype(np.float32, copy=False), device=self.device, dtype=th.float32).flatten()
        traj = th.as_tensor(self.swap_and_flatten(self.traj).astype(np.float32, copy=False), device=self.device, dtype=th.float32)
        true_params = th.as_tensor(self.swap_and_flatten(self.true_params).astype(np.float32, copy=False), device=self.device, dtype=th.float32)
        policy_context = th.as_tensor(self.swap_and_flatten(self.policy_context).astype(np.float32, copy=False), device=self.device, dtype=th.float32)
        use_nominal_context = th.as_tensor(self.swap_and_flatten(self.use_nominal_context), device=self.device, dtype=th.bool).flatten()

        n_samples = observations.shape[0]
        if batch_size is None:
            batch_size = n_samples

        indices = np.random.permutation(n_samples)
        start_idx = 0
        while start_idx < n_samples:
            batch_inds = indices[start_idx:start_idx + batch_size]
            start_idx += batch_size

            yield ContextRolloutBufferSamples(
                observations=observations[batch_inds],
                actions=actions[batch_inds].to(th.float32),
                old_values=old_values[batch_inds],
                old_log_prob=old_log_prob[batch_inds],
                advantages=advantages[batch_inds],
                returns=returns[batch_inds],
                traj=traj[batch_inds],
                true_params=true_params[batch_inds],
                policy_context=policy_context[batch_inds],
                use_nominal_context=use_nominal_context[batch_inds],
            )


# ---------- Policy with a pluggable context encoder (trajectory -> z) ----------

def build_mlp(input_dim: int, layer_sizes: List[int], output_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = input_dim
    for h in layer_sizes:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers += [nn.Linear(last, output_dim)]
    return nn.Sequential(*layers)


class TemporalCNNEncoder(nn.Module):
    def __init__(
        self,
        step_dim: int,
        output_dim: int,
        window_length: int,
        activation=nn.Tanh,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(step_dim, 16, kernel_size=7, stride=2, padding=3),
            activation(),
            nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
            activation(),
        )
        temporal_dim = self._conv1d_out_length(window_length, kernel_size=7, stride=2, padding=3)
        temporal_dim = self._conv1d_out_length(temporal_dim, kernel_size=5, stride=1, padding=2)
        if temporal_dim <= 0:
            raise ValueError(f"TemporalCNNEncoder produced invalid temporal_dim={temporal_dim} for window_length={window_length}")
        self.head = nn.Linear(16 * temporal_dim, output_dim)

    @staticmethod
    def _conv1d_out_length(length: int, *, kernel_size: int, stride: int, padding: int) -> int:
        return (length + 2 * padding - kernel_size) // stride + 1

    def forward(self, traj: th.Tensor) -> th.Tensor:
        # traj: (batch, time, step_dim)
        x = traj.transpose(1, 2)
        x = self.temporal(x).reshape(traj.shape[0], -1)
        return self.head(x)


class TransformerHistoryEncoder(nn.Module):
    def __init__(
            self,
            step_dim: int,
            output_dim: int,
            window_length: int,
            d_model: int = 32,
            n_heads: int = 4,
            ff_dim: int = 64,
            n_layers: int = 1,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.window_length = window_length
        self.token_embedding = nn.Linear(step_dim, d_model)
        self.position_embedding = nn.Embedding(window_length, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, traj: th.Tensor) -> th.Tensor:
        # traj: (batch, time, step_dim)
        valid = traj.abs().sum(dim=-1) > 0
        empty_history = ~valid.any(dim=1)
        valid = valid.clone()
        valid[empty_history, -1] = True
        positions = th.arange(self.window_length, device=traj.device)

        x = self.token_embedding(traj)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = self.transformer(x, src_key_padding_mask=~valid)

        weights = valid.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)



class ContextActorCriticPolicy(nn.Module):
    """
    A minimal actor-critic with:
      - context_encoder: (traj_window) -> z
      - actor, critic: (obs, z) -> pi, V

    It exposes:
      - forward_with_context(obs, traj) -> actions, values, log_prob
      - evaluate_actions_with_context(obs, traj, actions) -> values, log_prob, entropy
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        window_length: int,
        traj_dim: int,
        latent_dim: int = 8,
        use_transition_features: bool = False,
        transition_type: str = "s_next",
        encoder_type: Literal["mlp", "temporal_cnn", "transformer"] = "mlp",
        encoder_net_arch: List[int] = [64, 64],
        actor_net_arch: List[int] = [64, 64],
        critic_net_arch: List[int] = [64, 64],
        detach_context_for_rl: bool = False,
        condition_on_uncertainty: bool = False,
        uncertainty_context_dim: Optional[int] = None,
        device: Union[str, th.device] = "cpu",
        transformer_d_model: int = 32,
        transformer_n_heads: int = 4,
        transformer_ff_dim: int = 64,
        transformer_n_layers: int = 1,
        transformer_dropout: float = 0.0,
    ):
        super().__init__()
        self.device = th.device(device)
        self.observation_space = observation_space
        self.action_space = action_space

        obs_shape = get_obs_shape(observation_space)
        assert len(obs_shape) == 1, "This template assumes 1D obs vector."
        self.obs_dim = int(obs_shape[0])

        self.window_length = int(window_length)
        self.traj_dim = int(traj_dim)
        self.latent_dim = int(latent_dim)
        self.use_transition_features = bool(use_transition_features)
        self.transition_type = str(transition_type)
        self.encoder_type = str(encoder_type)
        self.encoder_net_arch = list(encoder_net_arch)
        self.detach_context_for_rl = bool(detach_context_for_rl)
        self.condition_on_uncertainty = bool(condition_on_uncertainty)
        self.uncertainty_context_dim = (
            int(uncertainty_context_dim)
            if uncertainty_context_dim is not None
            else self.latent_dim
        )
        if self.uncertainty_context_dim < 0:
            raise ValueError("uncertainty_context_dim must be nonnegative.")
        self.transformer_d_model = int(transformer_d_model)
        self.transformer_n_heads = int(transformer_n_heads)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.transformer_n_layers = int(transformer_n_layers)
        self.transformer_dropout = float(transformer_dropout)

        if self.use_transition_features and self.transition_type not in {"s_next", "delta"}:
            raise ValueError(
                "transition_type must be 's_next' or 'delta' when use_transition_features=True, "
                f"got: {self.transition_type!r}"
            )
        if self.encoder_type not in {"mlp", "temporal_cnn", "transformer"}:
            raise ValueError(f"encoder_type must be 'mlp', 'temporal_cnn', or 'transformer', got: {self.encoder_type!r}")

        # Build distributions depending on action space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = int(action_space.n)
            self.action_dist = CategoricalDistribution(self.action_dim)
            action_head_dim = self.action_dim
            self.log_std = None
        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = int(get_action_dim(action_space))
            self.action_dist = DiagGaussianDistribution(self.action_dim)
            action_head_dim = 64  # unused for Gaussian; kept for consistent init flow
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_dist = MultiCategoricalDistribution(action_space.nvec)
            self.action_dim = int(np.sum(action_space.nvec))
            action_head_dim = self.action_dim
            self.log_std = None
        elif isinstance(action_space, gym.spaces.MultiBinary):
            self.action_dist = BernoulliDistribution(action_space.n)
            self.action_dim = int(action_space.n)
            action_head_dim = self.action_dim
            self.log_std = None
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")

        self.context_encoder = self._build_context_encoder(self.latent_dim)

        # Actor and critic take [obs, context], where context can be z alone
        # or [z, uncertainty] in encoder_nll mode.
        self.context_dim = self.latent_dim + (
            self.uncertainty_context_dim if self.condition_on_uncertainty else 0
        )
        actor_in = self.obs_dim + self.context_dim
        critic_in = self.obs_dim + self.context_dim

        self.actor_mlp = build_mlp(actor_in, actor_net_arch, 64)
        self.critic_mlp = build_mlp(critic_in, critic_net_arch, 64)

        # Heads
        if isinstance(action_space, gym.spaces.Box):
            # SB3 Gaussian policy requires mean net + a learnable log_std
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=64)
        else:
            self.action_net = nn.Linear(64, action_head_dim)
        self.value_net = nn.Linear(64, 1)

        self._init_ortho_weights()

        self.to(self.device)

    @staticmethod
    def _ortho_init_module(module: nn.Module, gain: float) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _build_context_encoder(self, output_dim: int) -> nn.Module:
        if self.encoder_type == "mlp":
            enc_in = self.window_length * self.traj_dim
            return build_mlp(enc_in, self.encoder_net_arch, output_dim)

        if self.encoder_type == "temporal_cnn":
            return TemporalCNNEncoder(
                step_dim=self.traj_dim,
                output_dim=output_dim,
                window_length=self.window_length,
        )

        return TransformerHistoryEncoder(
            step_dim=self.traj_dim,
            output_dim=output_dim,
            window_length=self.window_length,
            d_model=self.transformer_d_model,
            n_heads=self.transformer_n_heads,
            ff_dim=self.transformer_ff_dim,
            n_layers=self.transformer_n_layers,
            dropout=self.transformer_dropout,
    )

    def _init_ortho_weights(self) -> None:
        """Initialize weights like SB3 ActorCriticPolicy (orthogonal init + head gains)."""

        # Hidden layers
        hidden_gain = math.sqrt(2)
        self.context_encoder.apply(lambda m: self._ortho_init_module(m, hidden_gain))
        self.actor_mlp.apply(lambda m: self._ortho_init_module(m, hidden_gain))
        self.critic_mlp.apply(lambda m: self._ortho_init_module(m, hidden_gain))

        # Heads
        # SB3 uses small init for action head and 1.0 for value head
        self._ortho_init_module(self.action_net, gain=0.01)
        self._ortho_init_module(self.value_net, gain=1.0)

    def encode_context(self, traj: th.Tensor, return_logvar: bool = False) -> th.Tensor:
        # traj: (batch, window_length, traj_dim)
        if self.encoder_type == "mlp":
            enc_input = traj.reshape(traj.shape[0], -1)
        else:
            enc_input = traj

        z = self.context_encoder(enc_input)
        
        if getattr(self, "nll_mode", False):
            mu, logvar = z.chunk(2, dim=-1)
            if return_logvar:
                return mu, logvar
            return mu

        if return_logvar:
            raise RuntimeError("return_logvar=True requires encoder_nll mode.")
        
        return z

    @staticmethod
    def clamp_logvar(logvar: th.Tensor) -> th.Tensor:
        return th.clamp(logvar, min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)

    def build_context_features(self, traj: th.Tensor) -> th.Tensor:
        if not self.condition_on_uncertainty:
            return self.encode_context(traj)

        mu, logvar = self.encode_context(traj, return_logvar=True)
        std = th.exp(0.5 * self.clamp_logvar(logvar))
        return th.cat([mu, std], dim=1)

    def _distribution_from_latent(self, latent_pi: th.Tensor):
        if isinstance(self.action_space, gym.spaces.Box):
            mean_actions = self.action_net(latent_pi)
            assert self.log_std is not None
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        params = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(params)

    def forward_with_context(self, obs: th.Tensor, traj: th.Tensor, deterministic: bool = False):
        context = self.build_context_features(traj)
        if self.detach_context_for_rl:
            context = context.detach()
        
        # context[:, :] = 0.0
        x = th.cat([obs, context], dim=1)

        latent_pi = self.actor_mlp(x)
        latent_v = self.critic_mlp(x)

        dist = self._distribution_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        # Important: SB3's CategoricalDistribution expects actions shaped (batch,)
        # (not (batch, 1)), otherwise log_prob can broadcast incorrectly.
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions_for_logprob = actions.long().flatten()
        else:
            actions_for_logprob = actions
        log_prob = dist.log_prob(actions_for_logprob)
        values = self.value_net(latent_v).flatten()
        return actions, values, log_prob

    def forward_with_z(self, obs: th.Tensor, z: th.Tensor, deterministic: bool = False):
        """Same as forward_with_context, but uses a provided context vector z.

        This is used for "privileged" conditioning where z is the ground-truth
        parameter vector rather than the encoder output.
        """
        # z[:, :] = 0.0
        x = th.cat([obs, z], dim=1)

        latent_pi = self.actor_mlp(x)
        latent_v = self.critic_mlp(x)

        dist = self._distribution_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions_for_logprob = actions.long().flatten()
        else:
            actions_for_logprob = actions
        log_prob = dist.log_prob(actions_for_logprob)
        values = self.value_net(latent_v).flatten()
        return actions, values, log_prob

    def evaluate_actions_with_context(self, obs: th.Tensor, traj: th.Tensor, actions: th.Tensor):
        context = self.build_context_features(traj)
        if self.detach_context_for_rl:
            context = context.detach()
        
        # context[:, :] = 0.0
        x = th.cat([obs, context], dim=1)

        latent_pi = self.actor_mlp(x)
        latent_v = self.critic_mlp(x)

        dist = self._distribution_from_latent(latent_pi)
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.long().flatten()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_v).flatten()
        return values, log_prob, entropy

    def evaluate_actions_with_z(self, obs: th.Tensor, z: th.Tensor, actions: th.Tensor):
        """Same as evaluate_actions_with_context, but uses a provided context vector z."""

        # z[:, :] = 0.0
        x = th.cat([obs, z], dim=1)

        latent_pi = self.actor_mlp(x)
        latent_v = self.critic_mlp(x)

        dist = self._distribution_from_latent(latent_pi)
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.long().flatten()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_v).flatten()
        return values, log_prob, entropy

    def predict_values_with_context(self, obs: th.Tensor, traj: th.Tensor):
        context = self.build_context_features(traj)
        if self.detach_context_for_rl:
            context = context.detach()
        
        # context[:, :] = 0.0
        x = th.cat([obs, context], dim=1)
        
        latent_v = self.critic_mlp(x)
        return self.value_net(latent_v).flatten()

    def predict_values_with_z(self, obs: th.Tensor, z: th.Tensor):
        
        # z[:, :] = 0.0
        x = th.cat([obs, z], dim=1)
        
        latent_v = self.critic_mlp(x)
        return self.value_net(latent_v).flatten()


# ---------- PPO variant that feeds trajectory windows into policy ----------

class UnifiedContextPPO(PPO):
    """
    PPO that:
      - maintains per-env history deque during rollout
      - stores the window in a ContextRolloutBuffer
      - uses the window for both acting and training, so gradients flow into the encoder
      - can use a closed-form LQR identifier (theta + variance) instead of a learned encoder
      - can train a privileged policy with an optional masked uncertainty channel
      - can mix privileged and encoder context at the episode level during encoder training
      - can penalize encoder uncertainty either as a loss regularizer or as a reward-shaping term
    """

    def __init__(
        self,
        policy,  # ignored: we build ContextActorCriticPolicy directly
        env: GymEnv | str,
        context_mode: Literal["encoder_mle", "encoder_nll", "closed_form", "privileged"] = "encoder_mle",
        window_length: int = 8,
        latent_dim: int = 8,
        use_transition_features: bool = False,
        transition_type: str = "s_next",
        encoder_type: Literal["mlp", "temporal_cnn", "transformer"] = "mlp",
        regression_param_names: list[str] | None = None,
        regression_coef: float = 1.0,
        encoder_net_arch: list[int] = [64, 64],
        actor_net_arch: list[int] = [64, 64],
        critic_net_arch: list[int] = [64, 64],
        detach_context_for_rl: bool = True,
        condition_on_uncertainty: bool = False,
        uncertainty_context_dim: int | None = None,
        z_scale: float = 1.0,
        freeze_ppo: bool = False,
        deterministic_actions: bool = False,
        policy_loss_coef: float = 1.0,
        uncertainty_regularization_coef: float = 0.0,
        uncertainty_reward_penalty_coef: float = 0.0,
        uncertainty_penalty_metric: Literal["variance", "std"] = "variance",
        privileged_uncertainty_mode: Literal["zeros", "max_std", "random_uniform", "constant", "predicted"] = "zeros",
        privileged_uncertainty_value: float = 0.0,
        privileged_context_probability: float = 0.0,
        id_update_interval: int = 1,
        nominal_warmup_steps: int = 0,
        initial_context: float | list[float] | None = None,
        closed_form_prior_mean: float | list[float] = 0.0,
        closed_form_prior_var: float | list[float] | None = None,
        closed_form_system: Literal["auto", "lqr", "stopping_car", "emergency_brake", "scalar_ab_lqr"] = "auto",
        closed_form_obs_noise_var_floor: float = 1e-4,
        naive_action_noise_std: float | list[float] = 0.0,
        naive_action_noise_dist: str | list[str] = "gaussian",
        action_log_std_init: float | None = None,
        transformer_d_model: int = 32,
        transformer_n_heads: int = 4,
        transformer_ff_dim: int = 64,
        transformer_n_layers: int = 1,
        transformer_dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        self.freeze_ppo = bool(freeze_ppo)
        self.deterministic_actions = bool(deterministic_actions)
        self.context_mode = str(context_mode)
        if self.context_mode not in {"encoder_mle", "encoder_nll", "closed_form", "privileged"}:
            raise ValueError(f"context_mode must be 'encoder_mle', 'encoder_nll', 'closed_form', or 'privileged', got: {context_mode!r}")
        if self.context_mode == "privileged" and self.freeze_ppo:
            raise ValueError("freeze_ppo=True is invalid in privileged mode because privileged mode bypasses the encoder and leaves no trainable parameters.")
        if self.context_mode == "closed_form" and self.freeze_ppo:
            raise ValueError("freeze_ppo=True is invalid in closed_form mode because closed-form ID bypasses encoder optimization.")
        
        self.condition_on_uncertainty = bool(condition_on_uncertainty)
        if self.condition_on_uncertainty and self.context_mode not in {"encoder_nll", "closed_form", "privileged"}:
            raise ValueError("condition_on_uncertainty=True requires context_mode='encoder_nll', 'closed_form', or 'privileged'.")
        self.uncertainty_context_dim = (
            int(uncertainty_context_dim)
            if uncertainty_context_dim is not None
            else self._latent_dim if hasattr(self, "_latent_dim") else int(latent_dim)
        )
        if self.uncertainty_context_dim < 0:
            raise ValueError("uncertainty_context_dim must be nonnegative.")
        if "uncertainty_penalty_coef" in kwargs:
            raise ValueError("`uncertainty_penalty_coef` was removed. Remove it from the config and use `uncertainty_regularization_coef` and/or `uncertainty_reward_penalty_coef` instead.")

        self.window_length = int(window_length)
        self._latent_dim = int(latent_dim)
        self.policy_loss_coef = float(policy_loss_coef)
        if self.policy_loss_coef < 0.0:
            raise ValueError("policy_loss_coef must be nonnegative.")
        self.use_transition_features = bool(use_transition_features)
        self.transition_type = str(transition_type)
        self._enc_type = str(encoder_type)
        self._transformer_d_model = int(transformer_d_model)
        self._transformer_n_heads = int(transformer_n_heads)
        self._transformer_ff_dim = int(transformer_ff_dim)
        self._transformer_n_layers = int(transformer_n_layers)
        self._transformer_dropout = float(transformer_dropout)
        init_setup_model = bool(kwargs.get("_init_setup_model", True))

        if regression_param_names is None:
            # If regression_param_names is not provided but we're initializing a UnifiedContextPPO, we create placeholder names based on latent_dim.
            if init_setup_model:
                raise ValueError("regression_param_names must be provided explicitly.")
            self.regression_param_names = [f"_placeholder_{i}" for i in range(self._latent_dim)]
        else:
            self.regression_param_names = [str(name) for name in regression_param_names]

        if self.context_mode == "closed_form":
            if self.regression_param_names not in (["theta"], ["a", "b"]):
                raise ValueError("closed_form mode supports regression_param_names=['theta'] or ['a', 'b'].")
            if not self.use_transition_features:
                raise ValueError("closed_form mode requires use_transition_features=True so transitions are available.")

        # Regression only makes sense when the encoder is used.
        self.regression_coef = float(regression_coef) if self.context_mode.startswith("encoder") else 0.0
        self.uncertainty_regularization_coef = (float(uncertainty_regularization_coef) if self.context_mode == "encoder_nll" else 0.0)
        self.uncertainty_reward_penalty_coef = (float(uncertainty_reward_penalty_coef) if self.context_mode in {"encoder_nll", "closed_form"} else 0.0)
        if (
            self.context_mode == "closed_form"
            and self.uncertainty_reward_penalty_coef != 0.0
            and not self.condition_on_uncertainty
        ):
            raise ValueError("closed_form mode with uncertainty_reward_penalty_coef != 0 requires condition_on_uncertainty=True.")
        self.uncertainty_penalty_metric = str(uncertainty_penalty_metric)
        if self.uncertainty_penalty_metric not in {"variance", "std"}:
            raise ValueError(f"uncertainty_penalty_metric must be 'variance' or 'std', got: {self.uncertainty_penalty_metric!r}")
        
        self._true_param_dim = int(len(self.regression_param_names))
        self._enc_arch = encoder_net_arch
        self._actor_arch = actor_net_arch
        self._critic_arch = critic_net_arch
        self._detach_context_for_rl = bool(detach_context_for_rl)
        self.z_scale = float(z_scale)
        self.privileged_context_probability = float(privileged_context_probability)
        if not 0.0 <= self.privileged_context_probability <= 1.0:
            raise ValueError(
                "privileged_context_probability must be between 0 and 1, "
                f"got {self.privileged_context_probability}"
            )
        if self.privileged_context_probability > 0.0:
            if self.context_mode not in {"encoder_mle", "encoder_nll", "closed_form"}:
                raise ValueError(
                    "privileged_context_probability > 0 requires context_mode='encoder_mle' "
                    "or context_mode='encoder_nll' or context_mode='closed_form'."
                )
            if not self._detach_context_for_rl:
                raise ValueError(
                    "privileged_context_probability > 0 requires detach_context_for_rl=True "
                    "so PPO updates use the exact mixed context stored during rollout."
                )
            if self.uncertainty_reward_penalty_coef != 0.0:
                raise ValueError(
                    "privileged_context_probability > 0 requires uncertainty_reward_penalty_coef=0."
                )
        self.privileged_uncertainty_mode = str(privileged_uncertainty_mode)
        if self.privileged_uncertainty_mode not in {"zeros", "max_std", "random_uniform", "constant", "predicted"}:
            raise ValueError("privileged_uncertainty_mode must be one of {'zeros', 'max_std', 'random_uniform', 'constant', 'predicted'}, "f"got: {self.privileged_uncertainty_mode!r}")
        if self.context_mode == "privileged" and self.privileged_uncertainty_mode == "predicted":
            raise ValueError("privileged_uncertainty_mode='predicted' is only valid when context_mode='encoder_nll'.")
        
        self.privileged_uncertainty_value = float(privileged_uncertainty_value)
        self.nominal_warmup_steps = int(nominal_warmup_steps)
        self.id_update_interval = int(id_update_interval)
        self.initial_context = initial_context
        self.closed_form_prior_mean = closed_form_prior_mean
        self.closed_form_prior_var = closed_form_prior_var
        if self.closed_form_prior_var is not None:
            prior_var_arr = np.asarray(self.closed_form_prior_var, dtype=np.float32).reshape(-1)
            if np.any(prior_var_arr <= 0.0):
                raise ValueError(
                    f"closed_form_prior_var must be positive or None, got {self.closed_form_prior_var}."
                )
        self.closed_form_system = str(closed_form_system)
        if self.closed_form_system not in {"auto", "lqr", "stopping_car", "emergency_brake", "scalar_ab_lqr"}:
            raise ValueError(
                "closed_form_system must be one of {'auto', 'lqr', 'stopping_car', 'emergency_brake', 'scalar_ab_lqr'}, "
                f"got {self.closed_form_system!r}."
            )
        if (
            self.context_mode == "closed_form"
            and self.regression_param_names == ["a", "b"]
            and self.condition_on_uncertainty
            and self.uncertainty_context_dim != 3
        ):
            raise ValueError("scalar_ab_lqr uncertainty-aware context requires uncertainty_context_dim=3.")
        self.closed_form_obs_noise_var_floor = float(closed_form_obs_noise_var_floor)
        if self.closed_form_obs_noise_var_floor <= 0.0:
            raise ValueError(
                "closed_form_obs_noise_var_floor must be positive, "
                f"got {self.closed_form_obs_noise_var_floor}."
            )
        if self.id_update_interval < 1:
            raise ValueError(f"id_update_interval must be >= 1, got {self.id_update_interval}")
        if self.nominal_warmup_steps < 0:
            raise ValueError(f"nominal_warmup_steps must be >= 0, got {self.nominal_warmup_steps}")
        if self.nominal_warmup_steps >= self.window_length:
            raise ValueError(
                f"nominal_warmup_steps ({self.nominal_warmup_steps}) must be < window_length ({self.window_length}) with history-based warmup."
            )
        
        self.naive_action_noise_std = naive_action_noise_std
        self.naive_action_noise_dist = naive_action_noise_dist
        self._ep_noise_std = naive_action_noise_std if isinstance(naive_action_noise_std, (int, float)) else naive_action_noise_std[-1]
        self._ep_noise_dist = "gaussian"
        self.action_log_std_init = None if action_log_std_init is None else float(action_log_std_init)
        
        # averaged uncertainty penalty over one rollout (policy update) used for logging
        # (gradient steps a done with individual values)
        self._last_uncertainty_reward_penalty = None
        # running cumulative return (stage cost + potential uncertainty penalty) for the current episode
        self._shaped_ep_returns: Optional[np.ndarray] = None
        # stored value for parameter estimation for asynchronous id intervals
        self._predict_cached_context: Optional[np.ndarray] = None
        self._predict_steps_since_update: Optional[np.ndarray] = None
        self._episode_uses_privileged_context: Optional[np.ndarray] = None
        self._closed_form_reg_eps = 1e-8
        # A0, B0, dA, dB tensors stored for closed-form GLS
        self._closed_form_lqr_tensors: Optional[Dict[str, th.Tensor]] = None

        # Dict-observation handling (shimmy dm_control returns Dict obs)
        self._flat_observation_space: Optional[gym.spaces.Box] = None

        # Normalization denominators for true params (set on first extraction)
        self._true_param_denoms: Optional[np.ndarray] = None

        # SB3 enforces MultiInputPolicy when env.observation_space is Dict.
        # We flatten Dict observations internally later, but must satisfy this check.
        try:
            obs_space = env.observation_space
        except Exception:
            obs_space = None
        if isinstance(obs_space, gym.spaces.Dict):
            policy = "MultiInputPolicy"

        super().__init__(policy=policy, env=env, *args, **kwargs)

    def _setup_model(self) -> None:
        self._closed_form_lqr_tensors = None
        super()._setup_model()

        # determine obs dimension and keys, if observation space is Dict.
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs_dim, obs_keys = _flat_obs_dim_from_space(self.observation_space)
        else:
            obs_dim, _ = _flat_obs_dim_from_space(self.observation_space)

        # create lower and upper bounds for the determined obs dimensions
        self._flat_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # determine action dimension
        if isinstance(self.action_space, gym.spaces.Discrete):
            act_enc_dim = int(self.action_space.n)
        else:
            act_enc_dim = int(get_action_dim(self.action_space))
        if self.context_mode == "closed_form" and not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("closed_form mode currently supports continuous Box action spaces only.")

        self._pair_traj_dim = int(obs_dim + act_enc_dim)
        self.traj_dim = int(self._pair_traj_dim + (obs_dim if self.use_transition_features else 0))

        # Supervised encoder training uses the embedding as the regression output.
        # If regression is disabled, the latent can be a small free bottleneck.
        if self._latent_dim != self._true_param_dim and self.regression_coef != 0.0:
            raise ValueError(
                f"latent_dim ({self._latent_dim}) must equal number of regression params "
                f"({self._true_param_dim}) when using the embedding as the regression output."
            )

        self.policy = ContextActorCriticPolicy(
            observation_space=self._flat_observation_space,
            action_space=self.action_space,
            window_length=self.window_length,
            traj_dim=self.traj_dim,
            latent_dim=self._latent_dim,
            use_transition_features=self.use_transition_features,
            transition_type=self.transition_type,
            encoder_type=self._enc_type,
            encoder_net_arch=self._enc_arch,
            actor_net_arch=self._actor_arch,
            critic_net_arch=self._critic_arch,
            detach_context_for_rl=self._detach_context_for_rl,
            condition_on_uncertainty=self.condition_on_uncertainty,
            uncertainty_context_dim=self.uncertainty_context_dim,
            device=self.device,
            transformer_d_model=self._transformer_d_model,
            transformer_n_heads=self._transformer_n_heads,
            transformer_ff_dim=self._transformer_ff_dim,
            transformer_n_layers=self._transformer_n_layers,
            transformer_dropout=self._transformer_dropout,
        )
        if self.action_log_std_init is not None and getattr(self.policy, "log_std", None) is not None:
            with th.no_grad():
                self.policy.log_std.fill_(float(self.action_log_std_init))

        # if nll, we override the encoder of the policy
        if self.context_mode == "encoder_nll":
            self.policy.nll_mode = True
            context_encoder = self.policy._build_context_encoder(self._latent_dim * 2)
            context_encoder.apply(lambda m: self.policy._ortho_init_module(m, math.sqrt(2)))
            self.policy.context_encoder = context_encoder.to(self.device)

        # Recreate optimizer and optionally freeze everything except the encoder
        if self.freeze_ppo:
            # Freeze all non-encoder parameters
            for name, p in self.policy.named_parameters():
                if not name.startswith("context_encoder."):
                    p.requires_grad_(False)
            # Only optimize the encoder
            self.policy.optimizer = th.optim.Adam(
                self.policy.context_encoder.parameters(),
                lr=self.lr_schedule(1.0),
                eps=1e-5,
            )
        elif self.context_mode == "closed_form":
            # Closed-form identification bypasses the neural context encoder entirely.
            for p in self.policy.context_encoder.parameters():
                p.requires_grad_(False)
            self.policy.optimizer = th.optim.Adam(
                [p for p in self.policy.parameters() if p.requires_grad],
                lr=self.lr_schedule(1.0),
                eps=1e-5,
            )
        else:
            self.policy.optimizer = th.optim.Adam(self.policy.parameters(), lr=self.lr_schedule(1.0), eps=1e-5)

        # Use our rollout buffer that stores traj windows
        self.rollout_buffer = ContextRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self._flat_observation_space,
            action_space=self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            window_length=self.window_length,
            traj_dim=self.traj_dim,
            true_param_dim=self._true_param_dim,
            policy_context_dim=self.policy.context_dim,
        )

    def _get_true_params(self, env: VecEnv) -> np.ndarray:
        """Extract per-env true dynamics parameters."""

        raw = np.zeros((env.num_envs, self._true_param_dim), dtype=np.float32)
        try:
            vector_getters = env.get_attr("get_true_params")
            denom_getters = env.get_attr("get_param_denoms")
        except Exception:
            vector_getters = None
            denom_getters = None

        if vector_getters is None or denom_getters is None:
            raise AttributeError(
                "UnifiedContextPPO requires the env wrapper stack to expose "
                "'get_true_params' and 'get_param_denoms'."
            )

        for i, (vector_getter, denom_getter) in enumerate(zip(vector_getters, denom_getters)):
            raw[i, :] = np.asarray(vector_getter(self.regression_param_names), dtype=np.float32)
            if self._true_param_denoms is None:
                self._true_param_denoms = np.asarray(
                    denom_getter(self.regression_param_names),
                    dtype=np.float32,
                )

        #return np.log(np.maximum(raw / self._true_param_denoms, 1e-8)) * self.z_scale
        return raw / self._true_param_denoms * self.z_scale

    def _ensure_true_param_denoms(self, env: VecEnv) -> None:
        if self._true_param_denoms is not None:
            return
        try:
            denom_getters = env.get_attr("get_param_denoms")
        except Exception as exc:
            raise AttributeError(
                "UnifiedContextPPO closed_form mode requires env.get_param_denoms(...) in the wrapper stack."
            ) from exc
        if len(denom_getters) == 0:
            raise AttributeError("Could not fetch parameter denominator getter from VecEnv.")
        self._true_param_denoms = np.asarray(
            denom_getters[0](self.regression_param_names),
            dtype=np.float32,
        )

    def _get_vec_env_attr(self, env: VecEnv, attr: str) -> List[Any]:
        try:
            values = env.get_attr(attr)
        except Exception as exc:
            raise AttributeError(
                f"UnifiedContextPPO closed_form mode requires VecEnv attribute {attr!r} in the env wrapper stack."
            ) from exc
        if len(values) == 0:
            raise AttributeError(f"VecEnv returned no values for attribute {attr!r}.")
        return list(values)

    def _get_closed_form_lqr_tensors(self, env: VecEnv) -> Dict[str, th.Tensor]:
        if self._closed_form_lqr_tensors is not None:
            return self._closed_form_lqr_tensors

        A_nominal_vals = self._get_vec_env_attr(env, "A_nominal")
        B_nominal_vals = self._get_vec_env_attr(env, "B_nominal")
        delta_B_vals = self._get_vec_env_attr(env, "delta_B")

        try:
            delta_A_vals = self._get_vec_env_attr(env, "delta_A")
        except AttributeError:
            # delta_A is created in set_dynamics_scales(). If it is missing, recover it from delta_B and K_nominal.
            K_nominal_vals = self._get_vec_env_attr(env, "K_nominal")
            delta_A_vals = [
                np.asarray(delta_B, dtype=np.float32) @ np.asarray(K, dtype=np.float32)
                for delta_B, K in zip(delta_B_vals, K_nominal_vals)
            ]

        A0 = np.asarray(A_nominal_vals[0], dtype=np.float32)
        B0 = np.asarray(B_nominal_vals[0], dtype=np.float32)
        dA = np.asarray(delta_A_vals[0], dtype=np.float32)
        dB = np.asarray(delta_B_vals[0], dtype=np.float32)

        obs_dim = int(np.prod(self._flat_observation_space.shape))
        if not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("closed_form mode currently supports Box action spaces only.")
        act_dim = int(np.prod(self.action_space.shape))

        if A0.shape != (obs_dim, obs_dim):
            raise ValueError(
                f"closed_form mode expected A_nominal shape {(obs_dim, obs_dim)}, got {A0.shape}."
            )
        if B0.shape != (obs_dim, act_dim):
            raise ValueError(
                f"closed_form mode expected B_nominal shape {(obs_dim, act_dim)}, got {B0.shape}."
            )
        if dA.shape != A0.shape:
            raise ValueError(f"closed_form mode expected delta_A shape {A0.shape}, got {dA.shape}.")
        if dB.shape != B0.shape:
            raise ValueError(f"closed_form mode expected delta_B shape {B0.shape}, got {dB.shape}.")

        self._closed_form_lqr_tensors = {
            "A0": th.as_tensor(A0, device=self.device, dtype=th.float32),
            "B0": th.as_tensor(B0, device=self.device, dtype=th.float32),
            "dA": th.as_tensor(dA, device=self.device, dtype=th.float32),
            "dB": th.as_tensor(dB, device=self.device, dtype=th.float32),
        }
        return self._closed_form_lqr_tensors

    def _get_closed_form_noise_inv_diag(
        self,
        env: VecEnv,
        n_envs: int,
        obs_dim: int,
        env_indices: Optional[Sequence[int]] = None,
    ) -> th.Tensor:
        process_noise_std_vals = self._get_vec_env_attr(env, "process_noise_std")
        process_noise_scale_vals = self._get_vec_env_attr(env, "process_noise_scale")
        try:
            velocity_index_vals = self._get_vec_env_attr(env, "velocity_obs_index")
        except AttributeError:
            velocity_index_vals = None
        if env_indices is None:
            env_indices = list(range(n_envs))
        if len(env_indices) != n_envs:
            raise ValueError(
                f"closed_form mode expected {n_envs} env indices, got {len(env_indices)}."
            )

        max_env_index = max(int(i) for i in env_indices) if len(env_indices) > 0 else -1
        if len(process_noise_std_vals) <= max_env_index or len(process_noise_scale_vals) <= max_env_index:
            raise ValueError(
                "closed_form mode could not fetch per-env process noise statistics for all environments."
            )

        inv_vars = np.zeros((n_envs, obs_dim), dtype=np.float32)
        for local_env_idx, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            std_raw = np.asarray(process_noise_std_vals[env_idx], dtype=np.float32).reshape(-1)
            if std_raw.size == 1:
                std_vec = np.full((obs_dim,), float(std_raw[0]), dtype=np.float32)
            elif std_raw.size == obs_dim:
                std_vec = std_raw
            else:
                raise ValueError(
                    f"closed_form mode expected process_noise_std size 1 or {obs_dim}, got {std_raw.size}."
                )
            scale_raw = np.asarray(process_noise_scale_vals[env_idx], dtype=np.float32).reshape(-1)
            if scale_raw.size == 0:
                raise ValueError("closed_form mode received empty process_noise_scale.")
            scale = float(scale_raw[0])
            var_diag = np.maximum((std_vec * scale) ** 2, self._closed_form_reg_eps)
            inv_vars[local_env_idx, :] = 1.0 / var_diag

        return th.as_tensor(inv_vars, device=self.device, dtype=th.float32)

    def _resolve_closed_form_system(self, env: VecEnv) -> str:
        closed_form_system = getattr(self, "closed_form_system", "lqr")
        if closed_form_system != "auto":
            return closed_form_system
        if self.regression_param_names == ["a", "b"]:
            return "scalar_ab_lqr"
        if env is None:
            return "lqr"
        try:
            self._get_vec_env_attr(env, "A_nominal")
            return "lqr"
        except AttributeError:
            pass
        try:
            self._get_vec_env_attr(env, "m_nominal")
            self._get_vec_env_attr(env, "dt")
            return "stopping_car"
        except AttributeError as exc:
            raise AttributeError(
                "closed_form_system='auto' could not identify either LQR matrices "
                "or stopping-car attributes in the env wrapper stack."
            ) from exc

    def _closed_form_prior_vectors(self, dim: int) -> tuple[th.Tensor, th.Tensor | None]:
        mean_arr = np.asarray(self.closed_form_prior_mean, dtype=np.float32).reshape(-1)
        if mean_arr.size == 1:
            mean_arr = np.full((dim,), float(mean_arr[0]), dtype=np.float32)
        if mean_arr.size != dim:
            raise ValueError(
                f"closed_form_prior_mean must be scalar or length {dim}, got length {mean_arr.size}."
            )
        mean = th.as_tensor(mean_arr, device=self.device, dtype=th.float32)

        if self.closed_form_prior_var is None:
            return mean, None
        var_arr = np.asarray(self.closed_form_prior_var, dtype=np.float32).reshape(-1)
        if var_arr.size == 1:
            var_arr = np.full((dim,), float(var_arr[0]), dtype=np.float32)
        if var_arr.size != dim:
            raise ValueError(
                f"closed_form_prior_var must be scalar or length {dim}, got length {var_arr.size}."
            )
        var = th.as_tensor(var_arr, device=self.device, dtype=th.float32)
        return mean, var

    def _estimate_closed_form_scalar_ab_posterior(
        self,
        traj: th.Tensor,
        env: VecEnv,
        env_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        if traj.ndim != 3:
            raise ValueError(f"Expected traj tensor of shape (batch, window, dim), got shape {tuple(traj.shape)}.")
        if self.regression_param_names != ["a", "b"] or self._latent_dim != 2:
            raise ValueError("scalar_ab_lqr closed_form mode requires regression_param_names=['a', 'b'] and latent_dim=2.")
        if self._flat_observation_space is None:
            raise RuntimeError("closed_form scalar_ab_lqr mode requires initialized flattened observation space.")

        obs_dim = int(np.prod(self._flat_observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        pair_dim = obs_dim + act_dim
        if obs_dim != 1 or act_dim != 1:
            raise ValueError(
                f"scalar_ab_lqr closed_form mode expected obs_dim=1 and act_dim=1, got obs_dim={obs_dim}, act_dim={act_dim}."
            )
        if traj.shape[-1] < pair_dim + obs_dim:
            raise ValueError(
                "scalar_ab_lqr closed_form mode requires transition features in trajectory windows. "
                f"Expected trailing transition dim {obs_dim}, got traj_dim={traj.shape[-1]}."
            )

        s = traj[:, :, :obs_dim]
        a = traj[:, :, obs_dim:pair_dim]
        trans = traj[:, :, pair_dim: pair_dim + obs_dim]
        s_next = s + trans if self.transition_type == "delta" else trans
        design = th.cat([s, a], dim=2)  # (batch, window, 2)
        y = s_next[:, :, 0]

        inv_noise_diag = self._get_closed_form_noise_inv_diag(
            env,
            n_envs=traj.shape[0],
            obs_dim=obs_dim,
            env_indices=env_indices,
        )
        inv_var = inv_noise_diag[:, 0].clamp_min(self._closed_form_reg_eps)
        weighted_design = design * inv_var[:, None, None]
        precision = th.matmul(weighted_design.transpose(1, 2), design)
        rhs = (weighted_design * y[:, :, None]).sum(dim=1)

        prior_mean, prior_var = self._closed_form_prior_vectors(2)
        if prior_var is not None:
            prior_precision = 1.0 / prior_var.clamp_min(self._closed_form_reg_eps)
            eye = th.eye(2, device=self.device, dtype=th.float32).unsqueeze(0)
            precision = precision + eye * prior_precision.view(1, 2)
            rhs = rhs + prior_precision.view(1, 2) * prior_mean.view(1, 2)

        reg_eye = self._closed_form_reg_eps * th.eye(2, device=self.device, dtype=th.float32).unsqueeze(0)
        safe_precision = precision + reg_eye
        cov = th.linalg.inv(safe_precision)
        mean = th.linalg.solve(safe_precision, rhs.unsqueeze(2)).squeeze(2)
        if prior_var is None:
            info_trace = precision.diagonal(dim1=1, dim2=2).sum(dim=1)
            low_info_mask = info_trace <= self._closed_form_reg_eps
            if bool(low_info_mask.any().item()):
                mean = th.where(low_info_mask[:, None], th.zeros_like(mean), mean)
        return mean, cov

    def _get_closed_form_stopping_car_inv_var(
        self,
        env: VecEnv,
        n_envs: int,
        env_indices: Optional[Sequence[int]] = None,
    ) -> th.Tensor:
        process_noise_std_vals = self._get_vec_env_attr(env, "process_noise_std")
        process_noise_scale_vals = self._get_vec_env_attr(env, "process_noise_scale")
        if env_indices is None:
            env_indices = list(range(n_envs))
        if len(env_indices) != n_envs:
            raise ValueError(
                f"closed_form stopping-car mode expected {n_envs} env indices, got {len(env_indices)}."
            )

        inv_vars = np.zeros((n_envs,), dtype=np.float32)
        for local_env_idx, env_idx in enumerate(env_indices):
            env_idx = int(env_idx)
            std_raw = np.asarray(process_noise_std_vals[env_idx], dtype=np.float32).reshape(-1)
            if std_raw.size == 0:
                vel_std = 0.0
            elif std_raw.size == 1:
                vel_std = float(std_raw[0])
            else:
                vel_idx = 1 if velocity_index_vals is None else int(velocity_index_vals[env_idx])
                vel_std = float(std_raw[vel_idx])
            scale_raw = np.asarray(process_noise_scale_vals[env_idx], dtype=np.float32).reshape(-1)
            scale = float(scale_raw[0]) if scale_raw.size > 0 else 1.0
            var = max((vel_std * scale) ** 2, self.closed_form_obs_noise_var_floor)
            inv_vars[local_env_idx] = 1.0 / var
        return th.as_tensor(inv_vars, device=self.device, dtype=th.float32)

    def _estimate_closed_form_stopping_car_theta_posterior(
        self,
        traj: th.Tensor,
        env: VecEnv,
        env_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        if traj.ndim != 3:
            raise ValueError(f"Expected traj tensor of shape (batch, window, dim), got shape {tuple(traj.shape)}.")
        if self.regression_param_names != ["theta"] or self._latent_dim != 1:
            raise ValueError("closed_form stopping-car mode supports regression_param_names=['theta'] only.")
        if self._flat_observation_space is None:
            raise RuntimeError("closed_form stopping-car mode requires initialized flattened observation space.")

        obs_dim = int(np.prod(self._flat_observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        pair_dim = obs_dim + act_dim
        if obs_dim < 1 or act_dim != 1:
            raise ValueError(
                f"closed_form stopping-car mode expected obs_dim>=1 and act_dim=1, got obs_dim={obs_dim}, act_dim={act_dim}."
            )
        if traj.shape[-1] < pair_dim + obs_dim:
            raise ValueError(
                "closed_form stopping-car mode requires transition features in trajectory windows. "
                f"Expected trailing transition dim {obs_dim}, got traj_dim={traj.shape[-1]}."
            )

        dt_vals = self._get_vec_env_attr(env, "dt")
        m_nominal_vals = self._get_vec_env_attr(env, "m_nominal")
        try:
            velocity_index_vals = self._get_vec_env_attr(env, "velocity_obs_index")
        except AttributeError:
            velocity_index_vals = [1 for _ in range(traj.shape[0])]
        if env_indices is None:
            env_indices = list(range(traj.shape[0]))
        if len(env_indices) != traj.shape[0]:
            raise ValueError(
                f"closed_form stopping-car mode expected {traj.shape[0]} env indices, got {len(env_indices)}."
            )
        dt = th.as_tensor(
            [float(dt_vals[int(env_idx)]) for env_idx in env_indices],
            device=self.device,
            dtype=th.float32,
        )
        m_nominal = th.as_tensor(
            [float(m_nominal_vals[int(env_idx)]) for env_idx in env_indices],
            device=self.device,
            dtype=th.float32,
        )
        velocity_indices = [int(velocity_index_vals[int(env_idx)]) for env_idx in env_indices]
        if any(idx < 0 or idx >= obs_dim for idx in velocity_indices):
            raise ValueError(
                f"closed_form stopping-car mode received invalid velocity_obs_index values "
                f"{velocity_indices} for obs_dim={obs_dim}."
            )

        s = traj[:, :, :obs_dim]
        a = traj[:, :, obs_dim:pair_dim]
        trans = traj[:, :, pair_dim: pair_dim + obs_dim]
        s_next = s + trans if self.transition_type == "delta" else trans
        row_indices = th.arange(traj.shape[0], device=self.device)
        vel_idx = th.as_tensor(velocity_indices, device=self.device, dtype=th.long)
        dv = s_next[row_indices[:, None], th.arange(traj.shape[1], device=self.device)[None, :], vel_idx[:, None]] - s[
            row_indices[:, None],
            th.arange(traj.shape[1], device=self.device)[None, :],
            vel_idx[:, None],
        ]
        u = a[:, :, 0]

        # dv = dt * (m_nominal + theta) * u + noise_v
        residual = dv - dt[:, None] * m_nominal[:, None] * u
        design = dt[:, None] * u
        inv_var = self._get_closed_form_stopping_car_inv_var(
            env,
            n_envs=traj.shape[0],
            env_indices=env_indices,
        )

        numerator = ((design * inv_var[:, None]) * residual).sum(dim=1)
        denominator = ((design * inv_var[:, None]) * design).sum(dim=1)
        if self.closed_form_prior_var is not None:
            prior_precision = 1.0 / self.closed_form_prior_var
            numerator = numerator + prior_precision * self.closed_form_prior_mean
            denominator = denominator + prior_precision
        safe_denominator = th.clamp(denominator, min=self._closed_form_reg_eps)

        theta_hat = numerator / safe_denominator
        if self.closed_form_prior_var is None:
            low_info_mask = denominator <= self._closed_form_reg_eps
            if bool(low_info_mask.any().item()):
                theta_hat = th.where(low_info_mask, th.zeros_like(theta_hat), theta_hat)
        var_theta = 1.0 / safe_denominator
        return theta_hat.unsqueeze(1), var_theta.unsqueeze(1)

    def _estimate_closed_form_theta_posterior(
        self,
        traj: th.Tensor,
        env: VecEnv,
        env_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        if self._resolve_closed_form_system(env) in {"stopping_car", "emergency_brake"}:
            return self._estimate_closed_form_stopping_car_theta_posterior(
                traj,
                env=env,
                env_indices=env_indices,
            )

        if traj.ndim != 3:
            raise ValueError(f"Expected traj tensor of shape (batch, window, dim), got shape {tuple(traj.shape)}.")
        if self.regression_param_names != ["theta"] or self._latent_dim != 1:
            raise ValueError("closed_form mode currently supports single-parameter theta estimation only.")
        if self._flat_observation_space is None:
            raise RuntimeError("closed_form mode requires initialized flattened observation space.")

        tensors = self._get_closed_form_lqr_tensors(env)
        A0 = tensors["A0"]
        B0 = tensors["B0"]
        dA = tensors["dA"]
        dB = tensors["dB"]

        obs_dim = int(A0.shape[0])
        pair_dim = self._pair_traj_dim
        if traj.shape[-1] < pair_dim + obs_dim:
            raise ValueError(
                "closed_form mode requires transition features in trajectory windows. "
                f"Expected trailing transition dim {obs_dim}, got traj_dim={traj.shape[-1]}."
            )

        s = traj[:, :, :obs_dim]
        a = traj[:, :, obs_dim:pair_dim]
        trans = traj[:, :, pair_dim: pair_dim + obs_dim]
        s_next = s + trans if self.transition_type == "delta" else trans

        # r_t = x_{t+1} - A0 x_t - B0 u_t
        r = s_next - (s @ A0.T) - (a @ B0.T)
        # g_t = dA x_t + dB u_t
        g = (s @ dA.T) + (a @ dB.T)

        inv_noise_diag = self._get_closed_form_noise_inv_diag(
            env,
            n_envs=traj.shape[0],
            obs_dim=obs_dim,
            env_indices=env_indices,
        )
        weighted_gr = (g * inv_noise_diag[:, None, :]) * r
        weighted_gg = (g * inv_noise_diag[:, None, :]) * g

        numerator = weighted_gr.sum(dim=(1, 2))
        denominator = weighted_gg.sum(dim=(1, 2))
        if self.closed_form_prior_var is not None:
            prior_precision = 1.0 / self.closed_form_prior_var
            numerator = numerator + prior_precision * self.closed_form_prior_mean
            denominator = denominator + prior_precision
        safe_denominator = th.clamp(denominator, min=self._closed_form_reg_eps)

        theta_hat = numerator / safe_denominator
        if self.closed_form_prior_var is None:
            low_info_mask = denominator <= self._closed_form_reg_eps
            if bool(low_info_mask.any().item()):
                theta_hat = th.where(low_info_mask, th.zeros_like(theta_hat), theta_hat)

        var_theta = 1.0 / safe_denominator
        return theta_hat.unsqueeze(1), var_theta.unsqueeze(1)

    def _closed_form_context_from_traj(
        self,
        traj: th.Tensor,
        env: VecEnv,
        env_indices: Optional[Sequence[int]] = None,
    ) -> th.Tensor:
        self._ensure_true_param_denoms(env)
        if self._resolve_closed_form_system(env) == "scalar_ab_lqr":
            mean_raw, cov_raw = self._estimate_closed_form_scalar_ab_posterior(
                traj,
                env,
                env_indices=env_indices,
            )
            denoms = th.as_tensor(
                np.maximum(np.abs(self._true_param_denoms[:2]), self._closed_form_reg_eps),
                device=self.device,
                dtype=th.float32,
            )
            mu_scaled = mean_raw / denoms.view(1, 2) * float(self.z_scale)
            if not self.condition_on_uncertainty:
                return mu_scaled

            var_a = th.clamp(cov_raw[:, 0, 0], min=0.0)
            var_b = th.clamp(cov_raw[:, 1, 1], min=0.0)
            std_a = th.sqrt(var_a) / denoms[0] * float(self.z_scale)
            std_b = th.sqrt(var_b) / denoms[1] * float(self.z_scale)
            corr = cov_raw[:, 0, 1] / th.sqrt(th.clamp(var_a * var_b, min=self._closed_form_reg_eps))
            corr = th.clamp(corr, min=-1.0, max=1.0)
            unc = th.stack([std_a, std_b, corr], dim=1)
            unc[:, :2] = th.clamp(
                unc[:, :2],
                min=MIN_UNCERTAINTY_STD,
                max=MAX_UNCERTAINTY_STD,
            )
            return th.cat([mu_scaled, unc], dim=1)

        theta_hat_raw, var_theta_raw = self._estimate_closed_form_theta_posterior(
            traj,
            env,
            env_indices=env_indices,
        )

        denom = float(abs(self._true_param_denoms[0]))
        denom = max(denom, self._closed_form_reg_eps)
        mu_scaled = theta_hat_raw / denom * float(self.z_scale)
        if not self.condition_on_uncertainty:
            return mu_scaled

        std_theta_raw = th.sqrt(th.clamp(var_theta_raw, min=0.0))
        std_scaled = std_theta_raw / denom * float(self.z_scale)
        std_scaled = th.clamp(std_scaled, min=MIN_UNCERTAINTY_STD, max=MAX_UNCERTAINTY_STD)
        return th.cat([mu_scaled, std_scaled], dim=1)

    def _encode_action_for_traj(self, actions: np.ndarray) -> np.ndarray:
        """
        actions from SB3 are usually shape (n_envs, 1) for Discrete or (n_envs, act_dim) for Box.
        Returns action encoding of shape (n_envs, act_enc_dim).
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            a = actions.reshape(-1).astype(int)
            one_hot = np.zeros((a.shape[0], self.action_space.n), dtype=np.float32)
            one_hot[np.arange(a.shape[0]), a] = 1.0
            return one_hot
        else:
            return actions.astype(np.float32)

    def _init_history(self) -> Tuple[List[Deque[np.ndarray]], List[Deque[np.ndarray]]]:
        # initialize empty history deques to build trajectory windows during the rollout
        obs_hist: List[Deque[np.ndarray]] = [deque(maxlen=self.window_length) for _ in range(self.n_envs)]
        act_hist: List[Deque[np.ndarray]] = [deque(maxlen=self.window_length) for _ in range(self.n_envs)]
        return obs_hist, act_hist

    def _build_traj_window(
        self,
        obs_hist: List[Deque[np.ndarray]],
        act_hist: List[Deque[np.ndarray]],
        current_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build window for each env: (n_envs, window_length, traj_dim)
        Pads with zeros at the beginning.
        """
        n_envs = len(obs_hist)
        traj = np.zeros((n_envs, self.window_length, self.traj_dim), dtype=np.float32)
        
        # Reshape to (1, obs_dim) if not None
        if current_obs is not None:
            # current_obs = np.asarray(current_obs, dtype=np.float32)
            if current_obs.ndim == 1:
                current_obs = current_obs.reshape((1, -1))
        
        for e in range(n_envs):
            # number of collected (obs, act) pairs
            n_pairs = len(obs_hist[e])
            if n_pairs == 0:
                continue
            
            if self.use_transition_features:
                # Only keep fully specified transitions: the most recent pair needs current_obs as s_{t+1}.
                n_full_transitions = min(n_pairs if current_obs is not None else max(n_pairs - 1, 0), self.window_length)
                if n_full_transitions == 0:
                    continue
                
                # concatenate state, action and transition depending on whether s_next is in obs_hist or current_obs
                for idx in range(n_full_transitions):
                    s = obs_hist[e][idx].astype(np.float32)
                    a = act_hist[e][idx].astype(np.float32)
                    if idx + 1 < n_pairs:
                        s_next = obs_hist[e][idx + 1].astype(np.float32)
                    else:
                        s_next = current_obs[e]
                    trans = s_next - s if self.transition_type == "delta" else s_next
                    traj[e, self.window_length - n_full_transitions + idx, :] = np.concatenate([s, a, trans], axis=0)
            
            else:
                # only state-action pairs without transitions
                for idx in range(n_pairs):
                    s = obs_hist[e][idx].astype(np.float32)
                    a = act_hist[e][idx].astype(np.float32)
                    traj[e, self.window_length - n_pairs + idx, :] = np.concatenate([s, a], axis=0)
        return traj

    def _use_nominal_context_from_history(self, obs_hist: List[Deque[np.ndarray]]) -> np.ndarray:
        # returns a per-env boolean mask saying whether to ignore the estimated context and use the nominal / zero context instead
        if self.context_mode == "privileged" or self.nominal_warmup_steps <= 0:
            return np.zeros((len(obs_hist),), dtype=np.bool_)
        return np.asarray([len(hist) < self.nominal_warmup_steps for hist in obs_hist], dtype=np.bool_)

    def _init_context_cache(self, n_envs: int) -> Tuple[np.ndarray, np.ndarray]:
        # Set cached context to the configured initial value and steps since update
        # to trigger immediate context computation when the ID interval allows it.
        cached_context = self._initial_context_cache(n_envs)
        initial_due_step = 0 if self.initial_context is not None else self.id_update_interval
        steps_since_update = np.full((n_envs,), initial_due_step, dtype=np.int64)
        return cached_context, steps_since_update

    def _initial_context_cache(self, n_envs: int) -> np.ndarray:
        context_dim = int(self.policy.context_dim)
        if self.initial_context is None:
            row = np.zeros((context_dim,), dtype=np.float32)
        else:
            arr = np.asarray(self.initial_context, dtype=np.float32).reshape(-1)
            if arr.size == 1 and context_dim != 1:
                row = np.full((context_dim,), float(arr[0]), dtype=np.float32)
            elif arr.size == context_dim:
                row = arr.astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"initial_context must be scalar or length {context_dim}, got length {arr.size}."
                )
        return np.tile(row.reshape(1, -1), (int(n_envs), 1)).astype(np.float32, copy=False)

    def _context_features_from_traj(
        self,
        traj: th.Tensor,
        use_nominal_context: Optional[Union[np.ndarray, th.Tensor]] = None,
        env: Optional[VecEnv] = None,
        env_indices: Optional[Sequence[int]] = None,
    ) -> th.Tensor:
        if self.context_mode == "closed_form":
            if env is None:
                if self.env is None:
                    raise RuntimeError("closed_form mode requires a bound environment to compute context from trajectories.")
                env = self.env
            context = self._closed_form_context_from_traj(
                traj,
                env=env,
                env_indices=env_indices,
            )
        else:
            context = self.policy.build_context_features(traj)
        if (
            self.context_mode in {"encoder_nll", "closed_form"}
            and self.condition_on_uncertainty
            and self.privileged_uncertainty_mode != "predicted"
        ):
            mu = context[:, : self._latent_dim]
            if self.privileged_uncertainty_mode == "reflected":
                predicted_uncertainty = context[:, self._latent_dim : 2 * self._latent_dim]
                uncertainty = th.clamp(
                    2.0 * float(self.privileged_uncertainty_value) - predicted_uncertainty,
                    min=MIN_UNCERTAINTY_STD,
                    max=MAX_UNCERTAINTY_STD,
                )
            else:
                uncertainty = self._masked_uncertainty_features(mu)
            context = th.cat([mu, uncertainty], dim=1)
        mean_override = getattr(self, "_eval_mean_context_override", None)
        if mean_override is not None:
            context = context.clone()
            if mean_override == "zeros":
                context[:, : self._latent_dim] = 0.0
            elif mean_override == "quantize":
                steps = th.as_tensor(
                    self._eval_mean_context_quantization_steps,
                    device=context.device,
                    dtype=context.dtype,
                )
                context[:, : self._latent_dim] = (
                    th.round(context[:, : self._latent_dim] / steps) * steps
                )
            else:
                raise RuntimeError(f"Unsupported evaluation mean-context override: {mean_override!r}")
        if self._detach_context_for_rl:
            context = context.detach()
        if use_nominal_context is None:
            return context
        warm_mask = th.as_tensor(use_nominal_context, device=context.device, dtype=th.bool).flatten()
        if bool(warm_mask.any().item()):
            initial_context = th.as_tensor(
                self._initial_context_cache(context.shape[0]),
                device=context.device,
                dtype=context.dtype,
            )
            context = context.clone()
            context[warm_mask] = initial_context[warm_mask]
        return context

    def _async_context_from_traj(
        self,
        traj: th.Tensor,
        use_nominal_context: Optional[Union[np.ndarray, th.Tensor]],
        cached_context_np: np.ndarray,
        steps_since_update: np.ndarray,
        env: Optional[VecEnv] = None,
        env_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[th.Tensor, np.ndarray, np.ndarray]:
        # If use_context was provided: flatten and convert to numpy boolean array.
        # If not provided, create a dummy array of all False (no warm up).
        if use_nominal_context is None:
            warm_mask_np = np.zeros((traj.shape[0],), dtype=np.bool_)
        else:
            warm_mask_np = np.asarray(use_nominal_context, dtype=np.bool_).reshape(-1)

        context_th = th.as_tensor(cached_context_np, device=self.device, dtype=th.float32)
        # due_mask_np marks envs where the context should be refreshed now
        due_mask_np = (~warm_mask_np) & (steps_since_update >= self.id_update_interval)

        # update context for envs that are due for an update
        if bool(np.any(due_mask_np)):
            fresh_context_th = self._context_features_from_traj(
                traj,
                use_nominal_context=None,
                env=env,
                env_indices=env_indices,
            )
            due_mask_th = th.as_tensor(due_mask_np, device=self.device, dtype=th.bool)
            context_th = context_th.clone()
            context_th[due_mask_th] = fresh_context_th[due_mask_th]
            steps_since_update = steps_since_update.copy()
            steps_since_update[due_mask_np] = 0

        # override context with nominal context for envs that are still in the warmup phase
        if bool(np.any(warm_mask_np)):
            warm_mask_th = th.as_tensor(warm_mask_np, device=self.device, dtype=th.bool)
            initial_context_th = th.as_tensor(
                self._initial_context_cache(traj.shape[0]),
                device=self.device,
                dtype=th.float32,
            )
            context_th = context_th.clone()
            context_th[warm_mask_th] = initial_context_th[warm_mask_th]

        cached_context_np = context_th.detach().cpu().numpy().astype(np.float32, copy=False)
        return context_th, cached_context_np, steps_since_update

    def _masked_uncertainty_features(self, reference: th.Tensor) -> th.Tensor:
        shape = (reference.shape[0], int(self.uncertainty_context_dim))
        if self.privileged_uncertainty_mode == "zeros":
            return th.zeros(shape, device=reference.device, dtype=reference.dtype)
        if self.privileged_uncertainty_mode == "max_std":
            return th.full(shape, fill_value=MAX_UNCERTAINTY_STD, device=reference.device, dtype=reference.dtype)
        if self.privileged_uncertainty_mode == "constant":
            return th.full(shape, fill_value=self.privileged_uncertainty_value, device=reference.device, dtype=reference.dtype)
        if self.privileged_uncertainty_mode == "random_uniform":
            low = float(getattr(self, "_eval_uncertainty_uniform_low", MIN_UNCERTAINTY_STD))
            high = float(getattr(self, "_eval_uncertainty_uniform_high", MAX_UNCERTAINTY_STD))
            return low + (high - low) * th.rand(shape, device=reference.device, dtype=reference.dtype)
        if self.privileged_uncertainty_mode == "predicted":
            raise RuntimeError(
                "privileged_uncertainty_mode='predicted' requires encoder_nll context, "
                "not a masked uncertainty placeholder."
            )
        raise RuntimeError(f"Unsupported privileged_uncertainty_mode={self.privileged_uncertainty_mode!r}")

    def _build_privileged_context(self, true_params: th.Tensor) -> th.Tensor:
        if not self.condition_on_uncertainty:
            return true_params
        uncertainty = self._masked_uncertainty_features(true_params)
        return th.cat([true_params, uncertainty], dim=1)

    def _build_curriculum_privileged_context(self, true_params: th.Tensor) -> th.Tensor:
        if not self.condition_on_uncertainty:
            return true_params
        zeros = th.zeros(
            (true_params.shape[0], int(self.uncertainty_context_dim)),
            device=true_params.device,
            dtype=true_params.dtype,
        )
        return th.cat([true_params, zeros], dim=1)

    def _sample_privileged_episode_sources(self, n_envs: int) -> np.ndarray:
        return np.random.random(size=n_envs) < self.privileged_context_probability

    def _uncertainty_metric_from_logvar(self, logvar: th.Tensor) -> th.Tensor:
        logvar = self.policy.clamp_logvar(logvar)
        if self.uncertainty_penalty_metric == "std":
            return th.exp(0.5 * logvar).mean(dim=1)
        return th.exp(logvar).mean(dim=1)

    def _uncertainty_metric_from_std(self, std: th.Tensor) -> th.Tensor:
        std = th.clamp(std, min=0.0)
        if self.uncertainty_penalty_metric == "std":
            return std.mean(dim=1)
        return (std ** 2).mean(dim=1)

    def predict(
        self,
        observation: Union[np.ndarray, th.Tensor, Mapping[str, Any]],
        state: Optional[Tuple[List[Deque[np.ndarray]], List[Deque[np.ndarray]]]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """
        Compatible with SB3's BaseAlgorithm.predict().

        We treat the returned `state` like a recurrent policy state: it stores
        per-env history deques that are used to build the trajectory window.
        """

        obs_np = _flatten_obs(observation)

        n_envs = obs_np.shape[0]

        if state is None:
            obs_hist: List[Deque[np.ndarray]] = [deque(maxlen=self.window_length) for _ in range(n_envs)]
            act_hist: List[Deque[np.ndarray]] = [deque(maxlen=self.window_length) for _ in range(n_envs)]
            if self.context_mode != "privileged":
                self._predict_cached_context, self._predict_steps_since_update = self._init_context_cache(n_envs)
        else:
            obs_hist, act_hist = state

        if self.context_mode != "privileged":
            if (
                self._predict_cached_context is None
                or self._predict_steps_since_update is None
                or self._predict_cached_context.shape[0] != n_envs
            ):
                self._predict_cached_context, self._predict_steps_since_update = self._init_context_cache(n_envs)

        if episode_start is not None:
            ep = np.asarray(episode_start).reshape(-1)
            for e in range(min(n_envs, ep.shape[0])):
                if bool(ep[e]):
                    obs_hist[e].clear()
                    act_hist[e].clear()

                    # sample action noise std and distribution
                    if isinstance(self.naive_action_noise_std, list):
                        self._ep_noise_std = float(np.random.uniform(self.naive_action_noise_std[0], self.naive_action_noise_std[1]))
                    else:
                        self._ep_noise_std = self.naive_action_noise_std
                    
                    if isinstance(self.naive_action_noise_dist, list):
                        self._ep_noise_dist = np.random.choice(self.naive_action_noise_dist)
                    else:                        
                        self._ep_noise_dist = self.naive_action_noise_dist

                    if self.context_mode != "privileged":
                        assert self._predict_cached_context is not None
                        assert self._predict_steps_since_update is not None
                        self._predict_cached_context[e] = self._initial_context_cache(n_envs)[e]
                        self._predict_steps_since_update[e] = 0 if self.initial_context is not None else self.id_update_interval

        obs_th = th.as_tensor(obs_np, device=self.device)

        self.policy.eval()
        with th.no_grad():
            if self.context_mode == "privileged":
                # In privileged mode, use the ground-truth params from the env
                # instead of running the (untrained) encoder.
                if self.env is None:
                    raise RuntimeError(
                        "In privileged mode, predict() needs access to the env "
                        "to extract true params. Call model.set_env(env) first."
                    )
                true_params_np = self._get_true_params(self.env)
                true_params_th = th.as_tensor(true_params_np, device=self.device, dtype=th.float32)
                privileged_context_th = self._build_privileged_context(true_params_th)
                actions_th, _, _ = self.policy.forward_with_z(
                    obs_th, privileged_context_th, deterministic=deterministic
                )
            else:
                traj_window_np = self._build_traj_window(
                    obs_hist,
                    act_hist,
                    current_obs=obs_np if self.use_transition_features else None,
                )
                traj_th = th.as_tensor(traj_window_np, device=self.device)
                use_nominal_context = self._use_nominal_context_from_history(obs_hist)
                assert self._predict_cached_context is not None
                assert self._predict_steps_since_update is not None
                context_th, self._predict_cached_context, self._predict_steps_since_update = (
                    self._async_context_from_traj(
                        traj_th,
                        use_nominal_context,
                        self._predict_cached_context,
                        self._predict_steps_since_update,
                        env=self.env,
                    )
                )
                actions_th, _, _ = self.policy.forward_with_z(obs_th, context_th, deterministic=deterministic)

        actions_np = actions_th.cpu().numpy()

        if self._ep_noise_dist == "gaussian":
            actions_np = actions_np + np.random.normal(0.0, self._ep_noise_std, size=actions_np.shape).astype(actions_np.dtype)
        elif self._ep_noise_dist == "uniform":
            bounds = self._ep_noise_std * np.sqrt(3)
            actions_np = actions_np + np.random.uniform(-bounds, bounds, size=actions_np.shape).astype(actions_np.dtype)
        else:
            raise ValueError(f"Unsupported noise distribution {self._ep_noise_dist!r}.")

        # Match SB3 behavior: clip actions for Box spaces before returning.
        if isinstance(self.action_space, gym.spaces.Box):
            actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)

        # Update history with the new (s_t, a_t) pair for future steps
        act_enc = self._encode_action_for_traj(actions_np)
        for e in range(n_envs):
            obs_hist[e].append(obs_np[e].copy())
            act_hist[e].append(act_enc[e].copy())
        if self.context_mode != "privileged" and self._predict_steps_since_update is not None:
            self._predict_steps_since_update = self._predict_steps_since_update + 1

        return actions_np, (obs_hist, act_hist)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback,
        rollout_buffer: ContextRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        
        # _self_obs is set in env.reset() if not reset wasn't called, it throws an error
        assert self._last_obs is not None
        if isinstance(self._last_obs, dict):
            self._last_obs = _flatten_obs(self._last_obs)

        # reset saved observations, trajectories, predicted contexts, true parameters all to zero for the new rollout
        rollout_buffer.reset()
        # initialize empty history deques to build trajectory windows during the rollout
        obs_hist, act_hist = self._init_history()
        # set cached context to zero and steps since update to trigger immediate context computation
        cached_context_np, steps_since_update = self._init_context_cache(self.n_envs)
        if (
            self._episode_uses_privileged_context is None
            or self._episode_uses_privileged_context.shape != (self.n_envs,)
        ):
            self._episode_uses_privileged_context = self._sample_privileged_episode_sources(self.n_envs)

        callback.on_rollout_start()
        uncertainty_reward_penalties: List[float] = []
        privileged_episode_returns: List[float] = []
        encoder_episode_returns: List[float] = []
        encoder_context_steps = 0
        
        # Initialize or reset the shaped episode returns if this is the first rollout
        # or if the number of environments has changed.
        if self._shaped_ep_returns is None or self._shaped_ep_returns.shape != (self.n_envs,):
            self._shaped_ep_returns = np.zeros(self.n_envs, dtype=np.float64)
        
        # Reset shaped-return accumulator for envs that are at an episode start on this rollout step.
        # This looks at the "last" i.e. most recent "episode_start" mask
        if self._last_episode_starts is not None:
            self._shaped_ep_returns[np.asarray(self._last_episode_starts, dtype=bool)] = 0.0

        n_steps = 0
        while n_steps < n_rollout_steps:
            # Build traj window BEFORE acting at this step (used in encoder mode)
            traj_window_np = self._build_traj_window(
                obs_hist,
                act_hist,
                current_obs=self._last_obs if self.use_transition_features else None,
            )
            traj_window_th = th.as_tensor(traj_window_np, device=self.device)
            # per env mask for whether to use nominal context instead of encoder context
            use_nominal_context = self._use_nominal_context_from_history(obs_hist)

            # Capture true params for this transition BEFORE stepping.
            true_params_np = self._get_true_params(env)
            true_params_th = th.as_tensor(true_params_np, device=self.device, dtype=th.float32)

            obs_th = th.as_tensor(self._last_obs, device=self.device)

            with th.no_grad():
                uncertainty_metric_th: Optional[th.Tensor] = None
                policy_context_th: th.Tensor
                if self.context_mode == "privileged":
                    privileged_context_th = self._build_privileged_context(true_params_th)
                    policy_context_th = privileged_context_th
                    actions_th, values_th, log_prob_th = self.policy.forward_with_z(
                        obs_th, privileged_context_th, deterministic=self.deterministic_actions
                    )
                else:
                    context_th, cached_context_np, steps_since_update = self._async_context_from_traj(
                        traj_window_th,
                        use_nominal_context,
                        cached_context_np,
                        steps_since_update,
                        env=env,
                    )
                    policy_context_th = context_th
                    if self.privileged_context_probability > 0.0:
                        privileged_mask_th = th.as_tensor(
                            self._episode_uses_privileged_context,
                            device=self.device,
                            dtype=th.bool,
                        )
                        if bool(privileged_mask_th.any().item()):
                            policy_context_th = policy_context_th.clone()
                            privileged_context_th = self._build_curriculum_privileged_context(true_params_th)
                            policy_context_th[privileged_mask_th] = privileged_context_th[privileged_mask_th]
                    actions_th, values_th, log_prob_th = self.policy.forward_with_z(
                        obs_th, policy_context_th, deterministic=self.deterministic_actions
                    )
                    # recompute uncertainty metric for penalties in case the true uncertainty wasn't fed to policy 
                    # but some masked version of it (e.g. zeros, max_std or random)
                    if self.context_mode == "encoder_nll" and self.uncertainty_reward_penalty_coef != 0.0:
                        _, logvar_th = self.policy.encode_context(traj_window_th, return_logvar=True)
                        uncertainty_metric_th = self._uncertainty_metric_from_logvar(logvar_th)
                    elif self.context_mode == "closed_form" and self.uncertainty_reward_penalty_coef != 0.0:
                        if self.condition_on_uncertainty:
                            std_th = context_th[:, self._latent_dim:]
                            uncertainty_metric_th = self._uncertainty_metric_from_std(std_th)

            actions_np = actions_th.cpu().numpy()
            # Note that this is adding dither / naive exploration noise to the actions sampled from the policy distribution
            # therefore the applied action differs from the one that was used to record log-probs and values. If policy is not 
            # frozen, this can lead to inaccuracies during traning. However, it provides additional signal to the encoder 
            # already during training, which is why we already apply it in the rollouts.
            if self._ep_noise_dist == "gaussian":
                actions_np = actions_np + np.random.normal(0.0, self._ep_noise_std, size=actions_np.shape).astype(actions_np.dtype)
            elif self._ep_noise_dist == "uniform":
                bounds = self._ep_noise_std * np.sqrt(3)
                actions_np = actions_np + np.random.uniform(-bounds, bounds, size=actions_np.shape).astype(actions_np.dtype)
            else:
                raise ValueError(f"Unsupported noise distribution {self._ep_noise_dist!r}.")

            # Match SB3 behavior: clip actions for Box spaces before stepping the env.
            # Important: keep the *unclipped* actions for PPO loss/log-prob consistency,
            # but use the clipped actions for the actual environment transition.
            actions_env = actions_np
            if isinstance(self.action_space, gym.spaces.Box):
                actions_env = np.clip(actions_np, self.action_space.low, self.action_space.high)
            act_enc = self._encode_action_for_traj(actions_env)

            # Step the env
            new_obs, rewards, dones, infos = env.step(actions_env)
            if self.context_mode != "privileged":
                steps_since_update = steps_since_update + 1
            encoder_context_steps += int(np.count_nonzero(~self._episode_uses_privileged_context))

            if uncertainty_metric_th is not None:
                uncertainty_penalty_np = (
                    self.uncertainty_reward_penalty_coef
                    * uncertainty_metric_th.cpu().numpy().astype(np.float32, copy=False)
                )
                rewards = rewards - uncertainty_penalty_np
                for idx, info in enumerate(infos):
                    info["uncertainty_reward_penalty"] = float(uncertainty_penalty_np[idx])
                    info["shaped_reward"] = float(rewards[idx])
                uncertainty_reward_penalties.extend(float(x) for x in uncertainty_penalty_np)

            self._shaped_ep_returns += rewards.astype(np.float64, copy=False)

            #####################################################
            # Handle timeout by bootstrapping with value function
            # (mirrors SB3's on_policy_algorithm.py, see GH#633)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs_np = _flatten_obs(infos[idx]["terminal_observation"])
                    terminal_obs_th = th.as_tensor(terminal_obs_np, device=self.device)
                    with th.no_grad():
                        uses_privileged_context = (
                            self.context_mode == "privileged"
                            or bool(self._episode_uses_privileged_context[idx])
                        )
                        if uses_privileged_context:
                            if self.context_mode == "privileged":
                                terminal_privileged_context = self._build_privileged_context(
                                    true_params_th[idx : idx + 1]
                                )
                            else:
                                terminal_privileged_context = self._build_curriculum_privileged_context(
                                    true_params_th[idx : idx + 1]
                                )
                            terminal_value = self.policy.predict_values_with_z(
                                terminal_obs_th, terminal_privileged_context
                            )[0]
                        else:
                            # For time-limit truncation, bootstrap the terminal observation with the
                            # history that includes the just-executed (s_t, a_t) pair.
                            bootstrap_obs_hist = deque(obs_hist[idx], maxlen=self.window_length)
                            bootstrap_act_hist = deque(act_hist[idx], maxlen=self.window_length)
                            bootstrap_obs_hist.append(self._last_obs[idx].copy())
                            bootstrap_act_hist.append(act_enc[idx].copy())
                            bootstrap_traj_np = self._build_traj_window(
                                [bootstrap_obs_hist],
                                [bootstrap_act_hist],
                                current_obs=terminal_obs_np if self.use_transition_features else None,
                            )
                            bootstrap_traj_th = th.as_tensor(bootstrap_traj_np, device=self.device)
                            bootstrap_use_nominal_context = self._use_nominal_context_from_history([bootstrap_obs_hist])
                            bootstrap_context_th, _, _ = self._async_context_from_traj(
                                bootstrap_traj_th,
                                bootstrap_use_nominal_context,
                                cached_context_np[idx : idx + 1].copy(),
                                steps_since_update[idx : idx + 1].copy(),
                                env=env,
                                env_indices=[idx],
                            )
                            terminal_value = self.policy.predict_values_with_z(terminal_obs_th, bootstrap_context_th)[0]
                    rewards[idx] += self.gamma * terminal_value.cpu().numpy()
                if done:
                    if "episode" in infos[idx]:
                        infos[idx]["episode"]["r_shaped"] = float(self._shaped_ep_returns[idx])
                        episode_return = float(infos[idx]["episode"]["r"])
                        if bool(self._episode_uses_privileged_context[idx]):
                            privileged_episode_returns.append(episode_return)
                        else:
                            encoder_episode_returns.append(episode_return)
                    self._shaped_ep_returns[idx] = 0.0
            #####################################################

            self._update_info_buffer(infos, dones)

            # Flatten new obs (Dict -> vector)
            if isinstance(new_obs, dict):
                new_obs = _flatten_obs(new_obs)

            # episode_starts: SB3 convention (True when a new episode starts at this step)
            episode_starts = self._last_episode_starts

            # Store in buffer with the traj window used to produce the action
            rollout_buffer.add(
                obs=self._last_obs,
                action=actions_np,
                reward=rewards,
                episode_start=episode_starts,
                value=values_th,
                log_prob=log_prob_th,
                traj_window=traj_window_np,
                true_params=true_params_np,
                policy_context=policy_context_th.detach().cpu().numpy().astype(np.float32, copy=False),
                use_nominal_context=use_nominal_context,
            )

            # Update history with the pair (s_t, a_t) 
            for e in range(self.n_envs):
                obs_hist[e].append(self._last_obs[e].copy())
                act_hist[e].append(act_enc[e].copy())
                if dones[e]:
                    obs_hist[e].clear()
                    act_hist[e].clear()
                    cached_context_np[e] = self._initial_context_cache(self.n_envs)[e]
                    steps_since_update[e] = 0 if self.initial_context is not None else self.id_update_interval
                    self._episode_uses_privileged_context[e] = self._sample_privileged_episode_sources(1)[0]

                    # sample action noise std and distribution
                    if isinstance(self.naive_action_noise_std, list):
                        self._ep_noise_std = float(np.random.uniform(self.naive_action_noise_std[0], self.naive_action_noise_std[1]))
                    else:
                        self._ep_noise_std = self.naive_action_noise_std
                    if isinstance(self.naive_action_noise_dist, list):
                        self._ep_noise_dist = np.random.choice(self.naive_action_noise_dist)
                    else:                        
                        self._ep_noise_dist = self.naive_action_noise_dist

            self._last_obs = new_obs
            self._last_episode_starts = dones

            n_steps += 1
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

        # Outside of loop, compute value for last obs using last context window
        traj_last_np = self._build_traj_window(
            obs_hist,
            act_hist,
            current_obs=self._last_obs if self.use_transition_features else None,
        )
        traj_last_th = th.as_tensor(traj_last_np, device=self.device)
        obs_last_th = th.as_tensor(self._last_obs, device=self.device)
        use_nominal_context_last = self._use_nominal_context_from_history(obs_hist)

        # True params at the end of the rollout for privileged value bootstrap
        true_params_last_np = self._get_true_params(env)
        true_params_last_th = th.as_tensor(true_params_last_np, device=self.device, dtype=th.float32)

        with th.no_grad():
            if self.context_mode == "privileged":
                last_privileged_context = self._build_privileged_context(true_params_last_th)
                last_values = self.policy.predict_values_with_z(obs_last_th, last_privileged_context)
            else:
                context_last_th, _, _ = self._async_context_from_traj(
                    traj_last_th,
                    use_nominal_context_last,
                    cached_context_np.copy(),
                    steps_since_update.copy(),
                    env=env,
                )
                if self.privileged_context_probability > 0.0:
                    privileged_mask_th = th.as_tensor(
                        self._episode_uses_privileged_context,
                        device=self.device,
                        dtype=th.bool,
                    )
                    if bool(privileged_mask_th.any().item()):
                        context_last_th = context_last_th.clone()
                        privileged_context_last_th = self._build_curriculum_privileged_context(true_params_last_th)
                        context_last_th[privileged_mask_th] = privileged_context_last_th[privileged_mask_th]
                last_values = self.policy.predict_values_with_z(obs_last_th, context_last_th)

        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)

        self._last_uncertainty_reward_penalty = (
            float(np.mean(uncertainty_reward_penalties)) if uncertainty_reward_penalties else 0.0
        )
        if len(self.ep_info_buffer) > 0 and "r_shaped" in self.ep_info_buffer[0]:
            self.logger.record(
                "rollout/ep_rew_shaped_mean",
                safe_mean([ep_info["r_shaped"] for ep_info in self.ep_info_buffer]),
            )
        if self.context_mode.startswith("encoder"):
            self.logger.record(
                "rollout/encoder_context_fraction",
                encoder_context_steps / float(n_rollout_steps * self.n_envs),
            )
            completed_episodes = len(privileged_episode_returns) + len(encoder_episode_returns)
            if completed_episodes > 0:
                self.logger.record(
                    "rollout/encoder_episode_fraction",
                    len(encoder_episode_returns) / float(completed_episodes),
                )
            if privileged_episode_returns:
                self.logger.record("rollout/ep_rew_mean_privileged", safe_mean(privileged_episode_returns))
            if encoder_episode_returns:
                self.logger.record("rollout/ep_rew_mean_encoder", safe_mean(encoder_episode_returns))

        callback.on_rollout_end()
        return True

    def train(self) -> None:
        # swtiching policy from .eval() to .train() for enabling gradients
        self.policy.train()

        # Update optimizer LR
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        regression_losses = []
        regression_mse_losses = []
        uncertainty_losses = []
        approx_kls = []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                obs = rollout_data.observations
                actions = rollout_data.actions
                old_log_prob = rollout_data.old_log_prob
                old_values = rollout_data.old_values
                advantages = rollout_data.advantages
                returns = rollout_data.returns
                traj = rollout_data.traj
                true_params = rollout_data.true_params
                policy_context = rollout_data.policy_context
                use_nominal_context = rollout_data.use_nominal_context

                # Advantage normalization (SB3 default)
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                if self.context_mode.startswith("encoder") and not self._detach_context_for_rl:
                    policy_context_for_update = self._context_features_from_traj(
                        traj,
                        use_nominal_context=use_nominal_context,
                        env=self.env,
                    )
                else:
                    policy_context_for_update = policy_context

                values, log_prob, entropy = self.policy.evaluate_actions_with_z(obs, policy_context_for_update, actions)

                # Policy loss
                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2))

                # Value loss (with optional clipping)
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                value_loss = th.mean((returns - values_pred) ** 2)

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                uncertainty_loss = th.zeros((), device=self.device)
                regression_mse = th.zeros((), device=self.device)
                if self.context_mode == "encoder_mle" and self.regression_coef != 0.0:
                    pred_params = self.policy.encode_context(traj)
                    regression_loss = F.mse_loss(pred_params, true_params)
                    regression_mse = regression_loss
                elif self.context_mode == "encoder_nll":
                    mu, logvar = self.policy.encode_context(traj, return_logvar=True)
                    logvar = self.policy.clamp_logvar(logvar)
                    uncertainty_loss = logvar.exp().mean()
                    if self.regression_coef != 0.0:
                        regression_mse = F.mse_loss(mu, true_params)
                        nll = 0.5 * (logvar + (true_params - mu) ** 2 / logvar.exp())
                        regression_loss = nll.mean()
                    else:
                        regression_loss = th.zeros((), device=self.device)
                elif self.context_mode == "closed_form":
                    # In closed-form mode, policy_context already contains [theta_hat] or [theta_hat, std_hat].
                    pred_params = policy_context[:, : self._latent_dim]
                    regression_mse = F.mse_loss(pred_params, true_params)
                    if self.condition_on_uncertainty:
                        pred_std = th.clamp(policy_context[:, self._latent_dim:], min=0.0)
                        uncertainty_loss = (pred_std ** 2).mean()
                    regression_loss = th.zeros((), device=self.device)
                else:
                    regression_loss = th.zeros((), device=self.device)

                loss = (
                    self.policy_loss_coef * policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.regression_coef * regression_loss
                    + self.uncertainty_regularization_coef * uncertainty_loss
                )

                # Approx KL for monitoring / early stopping if you want
                with th.no_grad():
                    approx_kl = th.mean(old_log_prob - log_prob).cpu().item()
                approx_kls.append(approx_kl)

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(entropy_loss.detach().cpu().item())
                pg_losses.append(policy_loss.detach().cpu().item())
                value_losses.append(value_loss.detach().cpu().item())
                regression_losses.append(regression_loss.detach().cpu().item())
                regression_mse_losses.append(regression_mse.detach().cpu().item())
                uncertainty_losses.append(uncertainty_loss.detach().cpu().item())

            # Optional: early stop on KL like SB3
            if self.target_kl is not None and np.mean(approx_kls) > 1.5 * self.target_kl:
                break

        self._n_updates += self.n_epochs

        # Logging (minimal)
        mean_entropy_loss = float(np.mean(entropy_losses))
        mean_pg_loss = float(np.mean(pg_losses))
        mean_value_loss = float(np.mean(value_losses))
        mean_regression_loss = float(np.mean(regression_losses)) if len(regression_losses) > 0 else 0.0
        mean_regression_mse = float(np.mean(regression_mse_losses)) if len(regression_mse_losses) > 0 else 0.0
        mean_uncertainty_loss = float(np.mean(uncertainty_losses)) if len(uncertainty_losses) > 0 else 0.0
        mean_approx_kl = float(np.mean(approx_kls))

        # Expose for callbacks (avoid reading progress.csv for live plotting)
        # Meaningful in encoder modes and in closed-form mode.
        if self.context_mode.startswith("encoder") or self.context_mode == "closed_form":
            self._last_regression_loss = mean_regression_loss
            self._last_regression_mse = mean_regression_mse
        else:
            self._last_regression_loss = None
            self._last_regression_mse = None

        if self.context_mode == "encoder_nll" or (
            self.context_mode == "closed_form" and self.condition_on_uncertainty
        ):
            self._last_regression_uncertainty = mean_uncertainty_loss
            self._last_uncertainty_loss = mean_uncertainty_loss
        else:
            self._last_regression_uncertainty = None
            self._last_uncertainty_loss = None

        self.logger.record("train/entropy_loss", mean_entropy_loss)
        self.logger.record("train/policy_gradient_loss", mean_pg_loss)
        self.logger.record("train/policy_loss_coef", float(self.policy_loss_coef))
        self.logger.record("train/value_loss", mean_value_loss)
        if self.context_mode.startswith("encoder") or self.context_mode == "closed_form":
            self.logger.record("train/regression_loss", mean_regression_loss)
            self.logger.record("train/regression_mse", mean_regression_mse)
        if self.context_mode == "encoder_nll" or (
            self.context_mode == "closed_form" and self.condition_on_uncertainty
        ):
            self.logger.record("train/uncertainty_loss", mean_uncertainty_loss)
            self.logger.record(
                "train/uncertainty_reward_penalty",
                float(self._last_uncertainty_reward_penalty or 0.0),
            )
        self.logger.record("train/approx_kl", mean_approx_kl)
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        ))
