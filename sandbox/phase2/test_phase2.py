"""Standalone M8: load ppo_final + phi_final, run four-condition reward eval + figs."""
import os
os.environ["MUJOCO_GL"] = "egl"
import time, sys
from pathlib import Path
import jax, jax.numpy as jp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brax.io import model
from brax.training.acme import running_statistics
from mujoco_playground import registry, wrapper

import twmr
from twmr.networks import (
    STUDENT_OBS_SIZE, PRIV_OBS_SIZE, ENV_LATENT_SIZE,
    EnvFactorEncoder, TeacherPolicyNetwork,
)
from twmr.adaptation import HISTORY_LEN, ACTION_SIZE, PER_STEP_FEAT, AdaptationModule

sys.path.insert(0, "/home/imeg2025/Transformable-Leg-Wheel-Robot/sandbox/phase2")
from rollout_utils import init_history, update_history, extract_frame

print("device:", jax.default_backend())

# ── Locate checkpoints (latest run) ─────────────────────────────────────────
ckpt_dir = sorted(
    Path("/home/imeg2025/Transformable-Leg-Wheel-Robot/sandbox/logs").glob("TWMRLegFlat-*/checkpoints"),
    key=lambda p: p.stat().st_mtime,
)[-1]
ppo_path = ckpt_dir / "ppo_final"
phi_path = ckpt_dir / "phi_final"
print("ppo:", ppo_path)
print("phi:", phi_path)

# ── Build env (sized for B=64 eval — M8 batch is small) ─────────────────────
EVAL_B = 64
EVAL_T = 250
NACONMAX = 25 * 512
NJMAX    = 1000
raw_env = registry.load("TWMRLegFlat", config_overrides={"naconmax": NACONMAX, "njmax": NJMAX})
env = wrapper.wrap_for_brax_training(raw_env, episode_length=EVAL_T, action_repeat=1)
print(f"env ready: action_size={raw_env.action_size}")

# ── Load checkpoints ────────────────────────────────────────────────────────
norm_params, policy_params, _ = model.load_params(str(ppo_path))
phi_params = model.load_params(str(phi_path))

POLICY_HIDDEN = (64, 64, 64)
policy_module = TeacherPolicyNetwork(action_size=2*raw_env.action_size, hidden_layer_sizes=POLICY_HIDDEN, latent_dim=ENV_LATENT_SIZE)
mu_module     = EnvFactorEncoder(latent_dim=ENV_LATENT_SIZE)
mu_params     = {"params": policy_params["params"]["EnvFactorEncoder_0"]}
phi_module    = AdaptationModule(latent_dim=ENV_LATENT_SIZE)
print("modules instantiated; loaded checkpoints")

def normalize_obs(o): return running_statistics.normalize(o, norm_params)
def compute_z(on):
    priv = on[..., STUDENT_OBS_SIZE:STUDENT_OBS_SIZE+PRIV_OBS_SIZE]
    return mu_module.apply(mu_params, priv)
def policy_action_with_z(on, z):
    s = on[..., :STUDENT_OBS_SIZE]
    logits = policy_module.apply(policy_params, s, z, method=TeacherPolicyNetwork.apply_with_z)
    return jp.tanh(logits[..., :ACTION_SIZE])

# ── Compute z_std from a quick rollout (needed for the random-baseline) ─────
def quick_roll_for_zstd(rng, B=64, T=75):
    rng, k = jax.random.split(rng)
    state = env.reset(jax.random.split(k, B))
    z_zero = jp.zeros((B, ENV_LATENT_SIZE))
    def step(s, _):
        on = normalize_obs(s.obs); z = compute_z(on); a = policy_action_with_z(on, z_zero)
        return env.step(s, a), z
    _, zs = jax.lax.scan(step, state, None, length=T)
    return jp.std(zs.reshape(-1, ENV_LATENT_SIZE), axis=0)
Z_STD = jax.jit(quick_roll_for_zstd)(jax.random.PRNGKey(42))
Z_STD.block_until_ready()
print(f"z_std: {np.array(Z_STD).round(3)}")

# ── Four-condition reward eval ──────────────────────────────────────────────
def episode_return_rollout(rng, latent_fn, B=EVAL_B, T=EVAL_T):
    rng, kr = jax.random.split(rng)
    state = env.reset(jax.random.split(kr, B))
    buf = init_history(B, HISTORY_LEN, PER_STEP_FEAT)
    def step(carry, key):
        state, buf, ret = carry
        on = normalize_obs(state.obs)
        z  = latent_fn(on, buf, key)
        a  = policy_action_with_z(on, z)
        nb = update_history(buf, extract_frame(on[..., :STUDENT_OBS_SIZE], a))
        ns = env.step(state, a)
        return (ns, nb, ret + ns.reward), None
    keys = jax.random.split(rng, T)
    (_, _, total), _ = jax.lax.scan(step, (state, buf, jp.zeros(B)), keys)
    return jp.mean(total)

def eval_all(rng, phi_params, z_std):
    keys = jax.random.split(rng, 4)
    z0 = jp.zeros((EVAL_B, ENV_LATENT_SIZE))
    return {
        "phi":    episode_return_rollout(keys[0], lambda on, b, k: phi_module.apply(phi_params, b)),
        "mu":     episode_return_rollout(keys[1], lambda on, b, k: compute_z(on)),
        "zero":   episode_return_rollout(keys[2], lambda on, b, k: z0),
        "random": episode_return_rollout(keys[3], lambda on, b, k: jax.random.normal(k, (EVAL_B, ENV_LATENT_SIZE)) * z_std),
    }

eval_jit = jax.jit(eval_all)
print("running 4-condition eval (B=64, T=250)...")
t0 = time.monotonic()
rew = eval_jit(jax.random.PRNGKey(31337), phi_params, Z_STD)
{k: float(v) for k, v in rew.items()}  # block
print(f"eval done in {time.monotonic()-t0:.1f}s")
rew = {k: float(v) for k, v in rew.items()}

print("\nM8 final rewards:")
for k in ("phi", "mu", "zero", "random"):
    print(f"  {k:<7s}  {rew[k]:>8.2f}")

phi_mu      = rew["phi"] / max(rew["mu"], 1e-9)
phi_v_zero  = rew["phi"] / max(rew["zero"], 1e-9) - 1.0
phi_v_rand  = rew["phi"] - rew["random"]
print(f"\nphi/mu     : {phi_mu:.2%}   (strong ≥90%, soft ≥85%)")
print(f"phi vs 0   : {phi_v_zero:+.1%}   (target >+10%)")
print(f"phi vs rand: {phi_v_rand:+.2f}   (target >0)")
soft = phi_mu >= 0.85; strong = phi_mu >= 0.90
print(f"\nstrong pass: {strong}   soft pass: {soft}   beats 0: {phi_v_zero > 0.10}   beats rand: {phi_v_rand > 0}")

# ── Bar + calibration scatter ───────────────────────────────────────────────
fig, (axb, axc) = plt.subplots(1, 2, figsize=(12, 4))
labels = ["ẑ=φ", "z=μ\n(teacher)", "0", "rand"]
vals = [rew["phi"], rew["mu"], rew["zero"], rew["random"]]
axb.bar(labels, vals, color=["C0", "C2", "C3", "C7"])
axb.set_ylabel("mean episode return")
axb.set_title(f"M8 reward parity (B={EVAL_B}, T={EVAL_T})")
for x, v in zip(labels, vals):
    axb.text(x, v, f"{v:.1f}", ha="center", va="bottom")

# Calibration: fresh on-policy rollout then scatter ẑ vs z per dim.
def cal_roll(rng, B=64, T=75):
    rng, kr = jax.random.split(rng)
    state = env.reset(jax.random.split(kr, B))
    buf = init_history(B, HISTORY_LEN, PER_STEP_FEAT)
    def step(c, _):
        state, buf = c
        on = normalize_obs(state.obs); z = compute_z(on)
        zh = phi_module.apply(phi_params, buf)
        a = policy_action_with_z(on, zh)
        nb = update_history(buf, extract_frame(on[..., :STUDENT_OBS_SIZE], a))
        return (env.step(state, a), nb), (buf, z)
    (_, _), (h, z) = jax.lax.scan(step, (state, buf), None, length=T)
    h = h[HISTORY_LEN:]; z = z[HISTORY_LEN:]
    H = h.reshape(-1, HISTORY_LEN, PER_STEP_FEAT); Z = z.reshape(-1, ENV_LATENT_SIZE)
    return H, Z, phi_module.apply(phi_params, H)
H_cal, Z_cal, P_cal = jax.jit(cal_roll)(jax.random.PRNGKey(99))
P_cal.block_until_ready()
for d in range(ENV_LATENT_SIZE):
    axc.scatter(np.array(Z_cal[:, d]), np.array(P_cal[:, d]), s=3, alpha=0.3, label=f"d{d}")
mn = float(min(jp.min(Z_cal), jp.min(P_cal))); mx = float(max(jp.max(Z_cal), jp.max(P_cal)))
axc.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
axc.set_xlabel("true z"); axc.set_ylabel("ẑ = φ(buffer)"); axc.set_title("Per-dim calibration")
axc.legend(fontsize=7, markerscale=2, ncol=2)
fig.tight_layout()
out = ckpt_dir / "m8_eval.png"
fig.savefig(out, dpi=120)
print(f"\nsaved figure to {out}")

assert soft, "FAIL: phi/mu < 85% soft target"
assert phi_v_zero > 0.10, "FAIL: φ doesn't beat zero by 10%"
assert phi_v_rand > 0, "FAIL: φ doesn't beat random"
print("\n" + ("M8 STRONG PASS — φ matches teacher within 10%." if strong
              else "M8 SOFT PASS — φ recovers >85% of teacher reward."))
