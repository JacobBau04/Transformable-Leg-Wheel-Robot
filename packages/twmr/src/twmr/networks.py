from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jp
from flax import linen as nn

from brax.training import distribution
from brax.training import networks
from brax.training import types


STUDENT_OBS_SIZE = 29
PRIV_OBS_SIZE = 9  # friction(1) + motor_strengths(8)
TEACHER_OBS_SIZE = STUDENT_OBS_SIZE + PRIV_OBS_SIZE
ENV_LATENT_SIZE = 8


class EnvFactorEncoder(nn.Module):
    latent_dim: int = ENV_LATENT_SIZE

    @nn.compact
    def __call__(self, privileged_obs: jax.Array) -> jax.Array:
        x = nn.Dense(64)(privileged_obs)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        z_t = nn.Dense(self.latent_dim)(x)
        return z_t


class TeacherPolicyNetwork(nn.Module):
    action_size: int
    hidden_layer_sizes: Sequence[int]
    latent_dim: int = ENV_LATENT_SIZE

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        student_obs = obs[..., :STUDENT_OBS_SIZE]
        privileged_obs = obs[..., STUDENT_OBS_SIZE:STUDENT_OBS_SIZE + PRIV_OBS_SIZE]
        z_t = EnvFactorEncoder(latent_dim=self.latent_dim)(privileged_obs)
        return self._mlp(student_obs, z_t)

    # Phase 2 inference path: skip the encoder and inject an externally-supplied z
    # (e.g. z_hat from the adaptation module). Param names match __call__ because
    # Flax auto-naming counts Dense layers in call order: encoder is "EnvFactorEncoder_0",
    # then the MLP Dense layers are "Dense_0..Dense_3" in both methods.
    @nn.compact
    def apply_with_z(self, student_obs: jax.Array, z: jax.Array) -> jax.Array:
        return self._mlp(student_obs, z)

    def _mlp(self, student_obs: jax.Array, z: jax.Array) -> jax.Array:
        x = jp.concatenate([student_obs, z], axis=-1)
        for hidden_size in self.hidden_layer_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.tanh(x)
        return nn.Dense(self.action_size)(x)


class TeacherValueNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int]
    latent_dim: int = ENV_LATENT_SIZE

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        student_obs = obs[..., :STUDENT_OBS_SIZE]
        privileged_obs = obs[..., STUDENT_OBS_SIZE:STUDENT_OBS_SIZE + PRIV_OBS_SIZE]

        z_t = EnvFactorEncoder(latent_dim=self.latent_dim)(privileged_obs)
        x = jp.concatenate([student_obs, z_t], axis=-1)

        for hidden_size in self.hidden_layer_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.tanh(x)

        value = nn.Dense(1)(x)
        return jp.squeeze(value, axis=-1)


@dataclass
class TeacherPPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_teacher_ppo_networks(
    observation_size,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    latent_dim: int = ENV_LATENT_SIZE,
) -> TeacherPPONetworks:
    # Brax may pass observation_size as int, tuple (e.g. (39,)), or Mapping; normalize to int.
    obs_size = jax.tree_util.tree_flatten(observation_size)[0][-1]
    if obs_size != TEACHER_OBS_SIZE:
        print(
            f"Warning: expected teacher obs size {TEACHER_OBS_SIZE}, "
            f"but got observation_size={obs_size}"
        )

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_module = TeacherPolicyNetwork(
        action_size=parametric_action_distribution.param_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        latent_dim=latent_dim,
    )

    value_module = TeacherValueNetwork(
        hidden_layer_sizes=value_hidden_layer_sizes,
        latent_dim=latent_dim,
    )

    dummy_obs = jp.zeros((1, obs_size))

    def policy_apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    def value_apply(processor_params, value_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return value_module.apply(value_params, obs)

    policy_network = networks.FeedForwardNetwork(
        init=lambda rng: policy_module.init(rng, dummy_obs),
        apply=policy_apply,
    )

    value_network = networks.FeedForwardNetwork(
        init=lambda rng: value_module.init(rng, dummy_obs),
        apply=value_apply,
    )

    return TeacherPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
