"""Env-level gates for the Step 1 robustness layer (need the warp backend / GPU).

Kept separate from test_robustness.py because that module forces JAX onto CPU
for its pure-function tests, which would break the warp physics backend used
here. Run with:  pytest packages/twmr/tests/test_robustness_env.py
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest

pytestmark = pytest.mark.env


def test_priv_slice_clean_student_noisy():
    """Noise hits the 29-dim student slice; the 9-dim privileged label stays exact.

    This is the invariant that makes phase 2 valid: phi regresses toward
    z = mu(privileged), so the privileged slice must be identical with noise on
    vs off, while the student slice (what phi/pi actually consume) must differ.
    """
    import twmr  # noqa: F401  (registers envs)
    from mujoco_playground import registry
    from twmr.networks import STUDENT_OBS_SIZE, PRIV_OBS_SIZE

    key = jax.random.PRNGKey(0)
    env_on = registry.load("TWMRLegFlat")
    env_off = registry.load("TWMRLegFlat", config_overrides={"obs_noise": {"enable": False}})
    s_on = jax.jit(env_on.reset)(key)
    s_off = jax.jit(env_off.reset)(key)

    lo, hi = STUDENT_OBS_SIZE, STUDENT_OBS_SIZE + PRIV_OBS_SIZE
    np.testing.assert_allclose(np.array(s_on.obs[lo:hi]), np.array(s_off.obs[lo:hi]), atol=1e-6)
    assert float(jp.max(jp.abs(s_on.obs[:STUDENT_OBS_SIZE] - s_off.obs[:STUDENT_OBS_SIZE]))) > 1e-4


def test_action_delay_reaches_motors_late():
    """delay=1: a strong forward command must NOT reach the wheels on tick 1."""
    import twmr  # noqa: F401
    from mujoco_playground import registry

    key = jax.random.PRNGKey(0)
    a = jp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # full forward wheels
    env_d1 = registry.load(
        "TWMRLegFlat",
        config_overrides={"action_delay_steps": 1, "obs_noise": {"enable": False}},
    )
    env_d0 = registry.load(
        "TWMRLegFlat",
        config_overrides={"action_delay_steps": 0, "obs_noise": {"enable": False}},
    )
    s_d1 = jax.jit(env_d1.step)(jax.jit(env_d1.reset)(key), a)
    s_d0 = jax.jit(env_d0.step)(jax.jit(env_d0.reset)(key), a)

    ctrl_d1 = float(jp.sum(jp.abs(s_d1.data.ctrl[:4])))
    ctrl_d0 = float(jp.sum(jp.abs(s_d0.data.ctrl[:4])))
    assert ctrl_d0 > 1.0     # immediate command drives the wheels hard (clipped torque)
    assert ctrl_d1 < 0.2     # delayed: only the tiny reset-velocity feedback term


def test_info_pytree_consistent_reset_vs_step():
    """Auto-reset requires reset() and step() to return the same info structure."""
    import twmr  # noqa: F401
    from mujoco_playground import registry

    env = registry.load("TWMRLegFlat")
    key = jax.random.PRNGKey(0)
    s0 = jax.jit(env.reset)(key)
    s1 = jax.jit(env.step)(s0, jp.zeros(env.action_size))
    t0 = jax.tree_util.tree_structure(s0.info)
    t1 = jax.tree_util.tree_structure(s1.info)
    assert t0 == t1, f"info pytree mismatch:\n reset={t0}\n step ={t1}"
