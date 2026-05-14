"""RMA Phase 2 — Adaptation module φ.

Maps a window of recent (student_obs, action) frames to an extrinsics estimate
ẑ that approximates the privileged latent z = μ(priv_obs) produced by the
EnvFactorEncoder during Phase 1.

Architecture:
    Input  : (B, HISTORY_LEN=25, PER_STEP_FEAT=37)   — student_obs(29) + action(8)
    Per-step MLP  37 → 64 → 32 (tanh)                — applied to last dim
    Temporal stack x3 (valid windowing, ReLU, 32 ch):
        kernel=4 stride=2 :  25 → 11
        kernel=3 stride=1 :  11 →  9
        kernel=3 stride=1 :   9 →  7
    Flatten 7 × 32 = 224
    Dense → ẑ of size ENV_LATENT_SIZE (8)

Two design notes:

* The CNN sizing diverges from the original RMA paper ([8/4][5/1][5/1]) because
  that stack collapses the temporal dim to ≤1 at HISTORY_LEN=25; this stack
  preserves a 7-step temporal feature map.
* The "Conv1D" layers are implemented as windowed Dense (manual im2col):
  ``nn.Dense`` over reshaped sliding windows. Numerically identical to
  Conv1D with VALID padding and the same param count, but routes through
  XLA matmul instead of cuDNN. Needed because cuDNN on V100 fails (status
  5003) for our particular Conv1D shapes.
"""
from typing import Sequence

import jax
import jax.numpy as jp
from flax import linen as nn

from .networks import ENV_LATENT_SIZE, STUDENT_OBS_SIZE


HISTORY_LEN = 25
ACTION_SIZE = 8
PER_STEP_FEAT = STUDENT_OBS_SIZE + ACTION_SIZE  # 29 + 8 = 37


def _windows(x: jax.Array, kernel: int, stride: int) -> jax.Array:
    """Build sliding windows over the time axis: (B, T, C) -> (B, T_out, kernel*C).

    Equivalent to im2col for a 1D conv with VALID padding.
    """
    B, T, C = x.shape
    T_out = (T - kernel) // stride + 1
    # Stack the kernel windows; each row i gathers x[:, i*s : i*s + k, :].
    win = jp.stack(
        [jax.lax.dynamic_slice_in_dim(x, i * stride, kernel, axis=1)
         for i in range(T_out)],
        axis=1,
    )  # (B, T_out, kernel, C)
    return win.reshape(B, T_out, kernel * C)


class AdaptationModule(nn.Module):
    """φ: per-step MLP + temporal windowed-Dense stack → ẑ."""

    latent_dim: int = ENV_LATENT_SIZE
    per_step_hidden: Sequence[int] = (64, 32)
    conv_channels: int = 32
    conv_specs: Sequence[tuple] = (
        (4, 2),  # (kernel, stride): 25 → 11
        (3, 1),  # 11 → 9
        (3, 1),  # 9 → 7
    )

    @nn.compact
    def __call__(self, history: jax.Array) -> jax.Array:
        # history: (B, HISTORY_LEN, PER_STEP_FEAT)
        x = history
        for h in self.per_step_hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        # x: (B, HISTORY_LEN, per_step_hidden[-1])

        for kernel, stride in self.conv_specs:
            x = _windows(x, kernel, stride)               # (B, T_out, k*C_in)
            x = nn.Dense(self.conv_channels)(x)           # (B, T_out, C_out)
            x = nn.relu(x)
        # x: (B, T_out, conv_channels) — T_out=7 for the default specs

        x = x.reshape(x.shape[0], -1)
        z_hat = nn.Dense(self.latent_dim)(x)
        return z_hat


def make_adaptation_module(
    latent_dim: int = ENV_LATENT_SIZE,
) -> AdaptationModule:
    return AdaptationModule(latent_dim=latent_dim)
