"""Verification gate for the numpy deploy path (Steps 3-4).

Three tiers:
  * numpy<->Flax numerical parity for pi and phi, and for the normalizer
    (the < 1e-5 gate from the plan). Needs jax + a trained checkpoint pair, but
    NOT warp/GPU, so it runs on CPU.
  * controller correctness: deploy constants must equal twmr.py, plus
    hand-computed torque cases including saturation.
  * wheel-encoder unwrap: a sequence crossing 2*pi yields a smooth velocity.

Run:  pytest packages/twmr/tests/test_deploy_parity.py
"""
import os
import sys
from pathlib import Path

# Flax reference runs on CPU; we never build the warp env here.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from twmr import deploy

_REPO = Path(__file__).resolve().parents[3]
_SANDBOX = _REPO / "sandbox"


# ── shared fixture: export the latest checkpoint pair to a temp npz ─────────
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    sys.path.insert(0, str(_SANDBOX))
    import export_params

    runs = sorted((_SANDBOX / "logs").glob("TWMRLegTerr-[0-9]*/checkpoints"),
                  key=lambda p: p.stat().st_mtime)
    pair = next(((c / "ppo_final", c / "phi_final") for c in reversed(runs)
                 if (c / "ppo_final").exists() and (c / "phi_final").exists()), None)
    if pair is None:
        pytest.skip("no TWMRLegTerr ppo_final+phi_final checkpoint pair under sandbox/logs")
    ppo_path, phi_path = pair
    out = tmp_path_factory.mktemp("deploy") / "params.npz"
    export_params.export(ppo_path, phi_path, out)
    return {"npz": out, "ppo": ppo_path, "phi": phi_path}


# ── pi / phi / normalizer parity (numpy vs Flax) ────────────────────────────
def test_pi_parity(exported):
    import jax, jax.numpy as jp
    from brax.io import model
    from twmr.networks import TeacherPolicyNetwork, STUDENT_OBS_SIZE, ENV_LATENT_SIZE

    _norm, policy_params, _value = model.load_params(str(exported["ppo"]))
    pol = TeacherPolicyNetwork(action_size=2 * 8, hidden_layer_sizes=(64, 64, 64),
                               latent_dim=ENV_LATENT_SIZE)
    rma = deploy.RMADeploy(str(exported["npz"]))

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(16):
        student = rng.standard_normal(STUDENT_OBS_SIZE).astype(np.float32)
        z = rng.standard_normal(ENV_LATENT_SIZE).astype(np.float32)
        logits = pol.apply(policy_params, jp.asarray(student)[None], jp.asarray(z)[None],
                           method=TeacherPolicyNetwork.apply_with_z)
        flax_action = np.tanh(np.asarray(logits)[0, :8])
        np_action = rma.pi_forward(student.astype(np.float64), z.astype(np.float64))
        max_diff = max(max_diff, float(np.max(np.abs(flax_action - np_action))))
    assert max_diff < 1e-5, f"pi parity max|diff|={max_diff:.2e}"


def test_phi_parity(exported):
    import jax, jax.numpy as jp
    from brax.io import model
    from twmr.adaptation import AdaptationModule, HISTORY_LEN, PER_STEP_FEAT
    from twmr.networks import ENV_LATENT_SIZE

    phi_params = model.load_params(str(exported["phi"]))
    phi = AdaptationModule(latent_dim=ENV_LATENT_SIZE)
    rma = deploy.RMADeploy(str(exported["npz"]))

    rng = np.random.default_rng(1)
    max_diff = 0.0
    for _ in range(8):
        hist = rng.standard_normal((HISTORY_LEN, PER_STEP_FEAT)).astype(np.float32)
        flax_z = np.asarray(phi.apply(phi_params, jp.asarray(hist)[None]))[0]
        np_z = rma.phi_forward(hist.astype(np.float64))
        max_diff = max(max_diff, float(np.max(np.abs(flax_z - np_z))))
    assert max_diff < 1e-5, f"phi parity max|diff|={max_diff:.2e}"


def test_normalizer_parity(exported):
    """deploy's (x-mean)/std on the 29-dim slice == Flax normalize of full obs[:29]."""
    import jax.numpy as jp
    from brax.io import model
    from brax.training.acme import running_statistics
    from twmr.networks import STUDENT_OBS_SIZE

    norm, _policy, _value = model.load_params(str(exported["ppo"]))
    rma = deploy.RMADeploy(str(exported["npz"]))

    rng = np.random.default_rng(2)
    full = rng.standard_normal(38).astype(np.float32)
    flax_n = np.asarray(running_statistics.normalize(jp.asarray(full), norm))[:STUDENT_OBS_SIZE]
    np_n = (full[:STUDENT_OBS_SIZE].astype(np.float64) - rma.norm_mean) / rma.norm_std
    assert float(np.max(np.abs(flax_n - np_n))) < 1e-5


# ── controller correctness ──────────────────────────────────────────────────
def test_controller_constants_match_twmr():
    """Deploy controller constants must not drift from twmr.py."""
    from twmr import twmr as T
    assert deploy.WHEEL_MAX_SPEED == T._WHEEL_MAX_SPEED
    assert deploy.WHEEL_KP == T._WHEEL_KP
    assert deploy.WHEEL_KD == T._WHEEL_KD
    assert deploy.WHEEL_TORQUE_LIMIT == T._WHEEL_TORQUE_LIMIT
    assert deploy.LEG_CENTER == T._LEG_CENTER
    assert deploy.LEG_HALF_RANGE == T._LEG_HALF_RANGE
    assert deploy.LEG_POS_KP == T._LEG_POS_KP
    assert deploy.LEG_VEL_KP == T._LEG_VEL_KP
    assert deploy.LEG_TORQUE_LIMIT == T._LEG_TORQUE_LIMIT
    assert deploy.LEG_MAX_VEL_CMD == T._LEG_MAX_VEL_CMD


def _bare_rma():
    """An RMADeploy whose weights don't matter (we only call .controller)."""
    rma = deploy.RMADeploy.__new__(deploy.RMADeploy)
    return rma


def test_controller_wheel_saturation():
    rma = _bare_rma()
    action = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])      # full forward wheel 0
    t = rma.controller(action, wheel_vel=np.zeros(4),
                       leg_pos=np.full(4, deploy.LEG_CENTER), leg_vel=np.zeros(4))
    # 0.2 * (8 - 0) = 1.6 -> clipped to the 0.8 limit
    assert abs(t[0] - deploy.WHEEL_TORQUE_LIMIT) < 1e-9
    assert np.allclose(t[1:4], 0.0)


def test_controller_leg_saturation():
    rma = _bare_rma()
    action = np.array([0, 0, 0, 0, 1.0, 0, 0, 0])      # leg 0 to full extension
    t = rma.controller(action, wheel_vel=np.zeros(4),
                       leg_pos=np.full(4, deploy.LEG_CENTER), leg_vel=np.zeros(4))
    # desired_leg_vel = clip(5*2.237, 10)=10 -> 0.3*10 = 3.0 -> clipped to 0.6
    assert abs(t[4] - deploy.LEG_TORQUE_LIMIT) < 1e-9


def test_controller_non_saturated():
    rma = _bare_rma()
    action = np.array([0.25, 0, 0, 0, 0.1, 0, 0, 0])
    t = rma.controller(action, wheel_vel=np.zeros(4),
                       leg_pos=np.full(4, deploy.LEG_CENTER), leg_vel=np.zeros(4))
    assert abs(t[0] - 0.2 * (0.25 * 8.0)) < 1e-9          # 0.2*2 = 0.4
    leg_des_vel = 5.0 * (0.1 * deploy.LEG_HALF_RANGE)      # 5*0.2237 = 1.1185
    assert abs(t[4] - 0.3 * leg_des_vel) < 1e-9


# ── step-loop ordering / obs assembly (pure numpy, real weights) ────────────
def test_step_loop_ordering_and_assembly(exported):
    """Pins the parts the parity tests don't: obs assembly order, finite-diff
    velocities, prev_action wiring, gravity-from-quat, and that phi consumes
    STRICTLY-PAST frames (z_t depends only on history before tick t)."""
    rma = deploy.RMADeploy(str(exported["npz"]))
    rng = np.random.default_rng(3)

    def rand_inputs():
        accel = rng.standard_normal(3)
        gyro = rng.standard_normal(3)
        quat = np.array([1.0, 0, 0, 0]) + 0.01 * rng.standard_normal(4)
        leg_pos = deploy.LEG_CENTER + 0.1 * rng.standard_normal(4)
        wheel_angle = rng.standard_normal(4)
        return accel, gyro, quat, leg_pos, wheel_angle

    # ----- tick 1: zero history, zero velocities, zero prev_action -----
    a1i, g1, q1, lp1, wa1 = rand_inputs()
    student1 = np.concatenate([a1i, g1, deploy.gravity_from_quat(q1), lp1,
                               np.zeros(4), np.zeros(4), np.zeros(8)])
    sn1 = (student1 - rma.norm_mean) / rma.norm_std
    z1 = rma.phi_forward(np.zeros((rma.history_len, rma.per_step_feat)))
    a1_expected = rma.pi_forward(sn1, z1)

    rma.reset()
    a1, _ = rma.step(a1i, g1, q1, lp1, wa1)
    assert np.allclose(a1, a1_expected, atol=0)
    assert np.allclose(rma.history[-1], np.concatenate([sn1, a1]), atol=0)  # newest frame
    assert np.allclose(rma.history[:-1], 0.0)                               # rest still zero

    # ----- tick 2: velocities are finite-diffs, prev_action == a1, z from past -----
    a2i, g2, q2, lp2, wa2 = rand_inputs()
    wv2 = deploy._wrap_to_pi(wa2 - wa1) / deploy.CTRL_DT
    lv2 = (lp2 - lp1) / deploy.CTRL_DT
    student2 = np.concatenate([a2i, g2, deploy.gravity_from_quat(q2), lp2, wv2, lv2, a1])
    sn2 = (student2 - rma.norm_mean) / rma.norm_std
    hist_before = rma.history.copy()                 # what phi must see at tick 2
    a2_expected = rma.pi_forward(sn2, rma.phi_forward(hist_before))

    a2, _ = rma.step(a2i, g2, q2, lp2, wa2)
    assert np.allclose(a2, a2_expected, atol=0)


# ── wheel-encoder unwrap ────────────────────────────────────────────────────
def test_wheel_unwrap_smooth_across_2pi():
    """A wheel spinning at constant speed across the 2*pi seam gives constant velocity."""
    dt = deploy.CTRL_DT
    step = 0.15  # rad/tick (~7.5 rad/s, under the 8 rad/s limit)
    angles = (np.arange(60) * step) % (2 * np.pi)         # wraps several times
    vels = []
    prev = None
    for a in angles:
        if prev is not None:
            vels.append(deploy._wrap_to_pi(np.array([a - prev])) / dt)
        prev = a
    vels = np.concatenate(vels)
    # every tick should report ~step/dt with no wrap spikes
    assert np.max(np.abs(vels - step / dt)) < 1e-6
