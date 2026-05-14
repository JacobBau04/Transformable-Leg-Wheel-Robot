"""History-buffer helpers for RMA Phase 2 rollouts.

The buffer holds the last HISTORY_LEN per-step frames `[student_obs, action]`
across B parallel envs. It is updated each env step with `update_history`,
which rolls the time axis left and writes the newest frame at the last index.

Layout: buf[:, 0, :] is the OLDEST frame; buf[:, -1, :] is the NEWEST.
This matches a 1D Conv1D's natural left-to-right temporal ordering.

Buffer lives outside the env (in the rollout `carry`) for two reasons:
  1) keeps the Phase 1 env and its observation contract untouched
  2) mirrors the Phase 3 deployment design (a process-side ring buffer,
     not env-coupled state)
"""
from __future__ import annotations

import jax
import jax.numpy as jp


def init_history(num_envs: int, history_len: int, feat: int) -> jax.Array:
    """Zero-initialised buffer of shape (num_envs, history_len, feat)."""
    return jp.zeros((num_envs, history_len, feat), dtype=jp.float32)


def update_history(buf: jax.Array, frame: jax.Array) -> jax.Array:
    """Append `frame` (B, feat) at the newest slot, dropping the oldest.

    Equivalent semantics to a ring buffer; uses roll+set so the result remains
    a contiguous (B, k, feat) tensor compatible with Conv1D over time.
    """
    rolled = jp.roll(buf, shift=-1, axis=1)
    return rolled.at[:, -1, :].set(frame)


def extract_frame(student_obs: jax.Array, prev_action: jax.Array) -> jax.Array:
    """Concatenate the per-step (student_obs, action) frame.

    student_obs : (B, STUDENT_OBS_SIZE)
    prev_action : (B, ACTION_SIZE)
    returns       (B, PER_STEP_FEAT)
    """
    return jp.concatenate([student_obs, prev_action], axis=-1)
