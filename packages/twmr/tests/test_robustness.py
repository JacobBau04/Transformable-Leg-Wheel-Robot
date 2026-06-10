"""Step 1 test gate: the sim-to-real robustness layer added to twmr.py.

Two tiers:
  * Pure-function tests (default) — exercise `add_obs_noise` / `apply_action_delay`
    directly on CPU. Fast, no env build. These guard the core math.
  * `env`-marked tests — build the actual env and assert the invariants that
    matter for training: the privileged slice stays clean while the student
    slice is noised, and the action delay reaches the motors one tick late.

Run fast tier:   pytest packages/twmr/tests -m "not env"
Run everything:  pytest packages/twmr/tests
"""
import os

# Pure-function tests don't need the GPU; force CPU so they run anywhere and
# don't contend with training jobs. (Set before jax initializes.)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import numpy as np
import pytest

from twmr.twmr import (
    add_obs_noise,
    apply_action_delay,
    default_config,
    _STUDENT_OBS_SIZE,
    _OBS_GRAVITY,
    _OBS_PREV_ACTION,
)


# ── helpers ───────────────────────────────────────────────────────────────
def _noise_cfg(**overrides):
    cfg = default_config().obs_noise   # fresh, mutable ConfigDict each call
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _zero_biases():
    return {"accel": jp.zeros(3), "gyro": jp.zeros(3), "gravity": jp.zeros(3)}


# ── add_obs_noise ───────────────────────────────────────────────────────────
def test_prev_action_is_left_exact():
    """prev_action (21:29) is a known command, never a sensor — must be untouched."""
    cfg = _noise_cfg()
    x = jp.arange(_STUDENT_OBS_SIZE, dtype=jp.float32)
    y = add_obs_noise(x, jax.random.PRNGKey(0), cfg, _zero_biases())
    np.testing.assert_array_equal(np.array(y[_OBS_PREV_ACTION]), np.array(x[_OBS_PREV_ACTION]))


def test_gravity_stays_unit_norm():
    """gravity is a direction; after perturbation it must be renormalized."""
    cfg = _noise_cfg()
    x = jp.zeros(_STUDENT_OBS_SIZE).at[_OBS_GRAVITY].set(jp.array([0.0, 0.0, -1.0]))
    y = add_obs_noise(x, jax.random.PRNGKey(1), cfg, _zero_biases())
    assert abs(float(jp.linalg.norm(y[_OBS_GRAVITY])) - 1.0) < 1e-5


def test_zero_std_zero_bias_is_identity_except_gravity_renorm():
    cfg = _noise_cfg(accel_std=0, gyro_std=0, gravity_std=0,
                     leg_pos_std=0, wheel_vel_std=0, leg_vel_std=0)
    x = jp.arange(_STUDENT_OBS_SIZE, dtype=jp.float32).at[_OBS_GRAVITY].set(
        jp.array([0.0, 0.0, -2.0]))
    y = add_obs_noise(x, jax.random.PRNGKey(2), cfg, _zero_biases())
    expect = x.at[_OBS_GRAVITY].set(x[_OBS_GRAVITY] / jp.linalg.norm(x[_OBS_GRAVITY]))
    np.testing.assert_allclose(np.array(y), np.array(expect), atol=1e-6)


def test_per_step_noise_varies_but_bias_is_the_mean():
    cfg = _noise_cfg()
    biases = {"accel": jp.array([1.0, 2.0, 3.0]), "gyro": jp.zeros(3), "gravity": jp.zeros(3)}
    x = jp.zeros(_STUDENT_OBS_SIZE)
    y1 = add_obs_noise(x, jax.random.PRNGKey(0), cfg, biases)
    y2 = add_obs_noise(x, jax.random.PRNGKey(1), cfg, biases)
    assert float(jp.max(jp.abs(y1 - y2))) > 1e-4          # different keys -> different noise

    keys = jax.random.split(jax.random.PRNGKey(5), 4000)
    ys = jax.vmap(lambda k: add_obs_noise(x, k, cfg, biases))(keys)
    np.testing.assert_allclose(np.array(jp.mean(ys[:, 0:3], axis=0)),
                               np.array([1.0, 2.0, 3.0]), atol=0.05)


def test_per_step_std_matches_config():
    cfg = _noise_cfg()
    x = jp.zeros(_STUDENT_OBS_SIZE)
    keys = jax.random.split(jax.random.PRNGKey(7), 20000)
    ys = jax.vmap(lambda k: add_obs_noise(x, k, cfg, _zero_biases()))(keys)
    assert abs(float(jp.std(ys[:, 0])) - cfg.accel_std) < 0.05   # accel
    assert abs(float(jp.std(ys[:, 3])) - cfg.gyro_std) < 0.02    # gyro
    assert float(jp.std(ys[:, 21:29])) == 0.0                    # prev_action exact


# ── apply_action_delay ────────────────────────────────────────────────────
def test_action_delay_fifo_semantics():
    """With delay=2, a command emerges exactly 2 ticks later."""
    buf = jp.zeros((2, 8))
    applied, buf = apply_action_delay(buf, jp.ones(8) * 1)
    np.testing.assert_array_equal(np.array(applied), np.zeros(8))   # tick 1: still zero
    applied, buf = apply_action_delay(buf, jp.ones(8) * 2)
    np.testing.assert_array_equal(np.array(applied), np.zeros(8))   # tick 2: still zero
    applied, buf = apply_action_delay(buf, jp.ones(8) * 3)
    np.testing.assert_array_equal(np.array(applied), np.ones(8) * 1)  # tick 3: first cmd lands
