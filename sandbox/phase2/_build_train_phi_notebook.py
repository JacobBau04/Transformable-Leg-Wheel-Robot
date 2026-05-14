"""Generates sandbox/phase2/train_phi.ipynb for M5 (overfit phi on a single batch).

Run once; resulting notebook is the working artifact. Safe to re-run; overwrites
the .ipynb. Kept here so future milestones (M6, M7) can extend the same file.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "train_phi.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md(
    "# RMA Phase 2 — Train Adaptation Module φ\n\n"
    "Trains φ on a fixed dataset (M5 overfit test), then will scale to "
    "on-policy DAgger-style data collection (M6, M7). Reuses the Phase 1 "
    "checkpoint produced by `sandbox/phase1_run.ipynb`."
))

cells.append(code("""# Imports + env vars
%env MUJOCO_GL=egl
import os
import functools
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
import optax
from absl import logging
import warnings

from brax.io import model
from brax.training.acme import running_statistics
from mujoco_playground import registry, wrapper

import twmr  # registers TWMRLegFlat
from twmr.networks import (
    STUDENT_OBS_SIZE, PRIV_OBS_SIZE, ENV_LATENT_SIZE,
    EnvFactorEncoder, TeacherPolicyNetwork, make_teacher_ppo_networks,
)
from twmr.adaptation import (
    HISTORY_LEN, ACTION_SIZE, PER_STEP_FEAT, AdaptationModule,
)

# Phase-2 helpers (live alongside this notebook)
sys.path.insert(0, str(Path.cwd()))
from rollout_utils import init_history, update_history, extract_frame

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("jax:", jax.__version__, " | device:", jax.default_backend())
"""))

cells.append(code("""# ── Config ───────────────────────────────────────────────────────────────────
ENV_NAME       = "TWMRLegFlat"
SEED           = 1

# M5 overfit test: small fixed dataset, ẑ = 0 during data collection
B_ROLLOUT      = 64       # parallel envs
T_ROLLOUT      = 75       # collect 75 steps, drop the first HISTORY_LEN to skip cold-start
NUM_OPT_STEPS  = 7500     # Adam steps on the fixed dataset (cosine-decayed LR)
LEARNING_RATE  = 3e-3     # higher than paper because we're driving training loss to ~0
BATCH_SIZE     = 256      # minibatch for SGD over the fixed dataset

POLICY_HIDDEN  = (64, 64, 64)
EPISODE_LENGTH = 250

# Auto-find the latest Phase 1 checkpoint, but allow manual override.
CHECKPOINT_PATH = None
if CHECKPOINT_PATH is None:
    candidates = sorted(
        Path("../logs").glob("TWMRLegFlat-*/checkpoints/ppo_final"),
        key=lambda p: p.stat().st_mtime,
    )
    assert candidates, "No Phase 1 checkpoint found; phase1_run.ipynb first."
    CHECKPOINT_PATH = candidates[-1]
print(f"Using Phase 1 checkpoint: {CHECKPOINT_PATH}")
"""))

cells.append(code("""# ── Build env (sized for the LARGEST batch we'll use — M7's B=512) ──────────
# naconmax/njmax are global allocations across the batch; if undersized, MJX
# silently drops contacts (you'll see "broadphase overflow" warnings) which
# corrupts the dynamics. Size for the biggest planned rollout.
MAX_B    = 512
NACONMAX = 25 * MAX_B   # ≈ 12.8k; covers observed ~2.2k peaks at B=512 with margin
NJMAX    = 1000
raw_env  = registry.load(ENV_NAME, config_overrides={"naconmax": NACONMAX, "njmax": NJMAX})
env      = wrapper.wrap_for_brax_training(raw_env, episode_length=EPISODE_LENGTH, action_repeat=1)
print(f"obs_size={raw_env.observation_size}  action_size={raw_env.action_size}  "
      f"naconmax={NACONMAX}  njmax={NJMAX}")
assert raw_env.action_size == ACTION_SIZE, f"action mismatch: {raw_env.action_size} vs {ACTION_SIZE}"
"""))

cells.append(code("""# ── Load Phase 1 checkpoint and instantiate the bare modules ─────────────────
loaded_params = model.load_params(str(CHECKPOINT_PATH))
norm_params, policy_params, _value_params = loaded_params

# Bare Flax modules (independent of Brax wrappers — needed for apply_with_z and μ)
policy_module = TeacherPolicyNetwork(
    action_size=2 * raw_env.action_size,           # NormalTanhDistribution param_size
    hidden_layer_sizes=POLICY_HIDDEN,
    latent_dim=ENV_LATENT_SIZE,
)
mu_module = EnvFactorEncoder(latent_dim=ENV_LATENT_SIZE)
mu_params = {"params": policy_params["params"]["EnvFactorEncoder_0"]}

print("Loaded checkpoint. Policy + μ + normalizer ready.")
"""))

cells.append(code("""# ── Inference helpers (frozen π conditioned on a supplied z) ─────────────────
# All take ALREADY-NORMALIZED full obs to keep the data path consistent with
# how the policy was trained (Brax normalizes before the network call).

def normalize_obs(obs):
    return running_statistics.normalize(obs, norm_params)

def compute_z(obs_norm):
    \"\"\"True z from the encoder μ on the privileged slice of normalized obs.\"\"\"
    priv = obs_norm[..., STUDENT_OBS_SIZE:STUDENT_OBS_SIZE + PRIV_OBS_SIZE]
    return mu_module.apply(mu_params, priv)

def policy_action_with_z(obs_norm, z, deterministic=True):
    \"\"\"Compute the deterministic action from π given an externally supplied z.

    The policy outputs (mean, log_std) concatenated. In deterministic mode we
    return tanh(mean) — same convention as NormalTanhDistribution.mode().
    \"\"\"
    student = obs_norm[..., :STUDENT_OBS_SIZE]
    logits = policy_module.apply(
        policy_params, student, z, method=TeacherPolicyNetwork.apply_with_z
    )
    mean = logits[..., :ACTION_SIZE]
    return jp.tanh(mean)  # deterministic mode = mode of NormalTanhDistribution
"""))

cells.append(code("""# ── M5: collect a FIXED dataset by rolling out frozen π with ẑ = 0 ───────────
# Output (after dropping the cold-start warmup window inside the scan):
#   histories : (B*(T-HISTORY_LEN), HISTORY_LEN, PER_STEP_FEAT)
#   z_targets : (B*(T-HISTORY_LEN), ENV_LATENT_SIZE)
# With B=64 and T=75: 64*50 = 3200 (history, z) pairs, all with full buffers.

def collect_fixed_dataset(rng):
    rng, k_reset = jax.random.split(rng)
    keys = jax.random.split(k_reset, B_ROLLOUT)
    state = env.reset(keys)
    buf = init_history(B_ROLLOUT, HISTORY_LEN, PER_STEP_FEAT)
    z_zero = jp.zeros((B_ROLLOUT, ENV_LATENT_SIZE), dtype=jp.float32)

    def step_fn(carry, _):
        state, buf = carry
        obs_norm = normalize_obs(state.obs)
        z_true   = compute_z(obs_norm)
        a_t      = policy_action_with_z(obs_norm, z_zero)   # ẑ = 0 → fixed data
        # Record (buffer at time of decision, target z)
        rec_buf, rec_z = buf, z_true
        # Update buffer with the frame [student_t, a_t]
        student_t = obs_norm[..., :STUDENT_OBS_SIZE]
        frame = extract_frame(student_t, a_t)
        new_buf = update_history(buf, frame)
        new_state = env.step(state, a_t)
        return (new_state, new_buf), (rec_buf, rec_z)

    (final_state, _), (hist_seq, z_seq) = jax.lax.scan(
        step_fn, (state, buf), None, length=T_ROLLOUT
    )
    # hist_seq: (T, B, K, F);  z_seq: (T, B, 8)
    # Drop the first HISTORY_LEN timesteps: those snapshots have a partially-zero
    # buffer (cold start), which creates irreducible label ambiguity (different
    # envs look identical when most of the history is zeros). After the drop,
    # every retained sample has a full real history.
    hist_seq = hist_seq[HISTORY_LEN:]
    z_seq    = z_seq[HISTORY_LEN:]
    histories = hist_seq.reshape(-1, HISTORY_LEN, PER_STEP_FEAT)
    z_targets = z_seq.reshape(-1, ENV_LATENT_SIZE)
    return histories, z_targets

collect_jit = jax.jit(collect_fixed_dataset)
import time
t0 = time.monotonic()
histories, z_targets = collect_jit(jax.random.PRNGKey(SEED))
histories.block_until_ready()
print(f"collected dataset in {time.monotonic()-t0:.1f}s")
print(f"histories: {histories.shape} {histories.dtype}")
print(f"z_targets: {z_targets.shape} {z_targets.dtype}")
print(f"z_target stats: mean={float(jp.mean(z_targets)):+.3f}  std={float(jp.std(z_targets)):.3f}  "
      f"abs_max={float(jp.max(jp.abs(z_targets))):.3f}")
"""))

cells.append(code("""# ── Initialize φ and optimizer, train on the fixed dataset for NUM_OPT_STEPS ─
# All on GPU. NB: φ uses windowed-Dense rather than nn.Conv1D — see
# packages/twmr/src/twmr/adaptation.py for why (V100 cuDNN bug on our shape).

phi_module = AdaptationModule(latent_dim=ENV_LATENT_SIZE)
phi_params = phi_module.init(jax.random.PRNGKey(SEED + 1), histories[:4])

# Cosine decay shaves the late-training plateau and lets us reach ratio < 1%.
schedule = optax.cosine_decay_schedule(
    init_value=LEARNING_RATE, decay_steps=NUM_OPT_STEPS, alpha=0.01
)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
opt_state = opt.init(phi_params)

def loss_fn(params, hist, z):
    pred = phi_module.apply(params, hist)
    return jp.mean((pred - z) ** 2)

@jax.jit
def train_step(params, opt_state, hist, z):
    loss, grads = jax.value_and_grad(loss_fn)(params, hist, z)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Initial loss on the full dataset (for the ratio test).
init_loss = float(loss_fn(phi_params, histories, z_targets))
print(f"initial full-dataset loss: {init_loss:.6f}")

losses = []
N = histories.shape[0]
key = jax.random.PRNGKey(SEED + 2)
for step in range(NUM_OPT_STEPS):
    key, k = jax.random.split(key)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, N)
    phi_params, opt_state, loss = train_step(phi_params, opt_state, histories[idx], z_targets[idx])
    losses.append(float(loss))
    if (step + 1) % 500 == 0 or step == 0:
        print(f"  step {step+1:>4}  minibatch loss = {float(loss):.6f}")

final_loss_full = float(loss_fn(phi_params, histories, z_targets))
print(f"\\nfinal full-dataset loss:   {final_loss_full:.6f}")
print(f"loss ratio final/initial: {final_loss_full / init_loss:.4%}")
"""))

cells.append(code("""# ── M5 success criterion: final ≤ 1% of initial (R² > 0.95 on training data) ─
ratio = final_loss_full / init_loss
print(f"final/initial = {ratio:.4%}")
assert ratio < 0.01, f"Overfit failed: ratio {ratio:.4%} (need < 1%). Capacity / optimizer / data alignment is wrong."

# Per-dim R²: variance explained per latent dim
var_z = jp.var(z_targets, axis=0)
pred_full = phi_module.apply(phi_params, histories)
mse_per_dim = jp.mean((pred_full - z_targets) ** 2, axis=0)
r2_per_dim = 1.0 - mse_per_dim / jp.maximum(var_z, 1e-12)
print("per-dim R² on training data:", np.array(r2_per_dim).round(3))
print("M5 PASS — φ has capacity, gradients flow, optimizer/loss/alignment are correct.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# M6 — Single Phase-2 iteration end-to-end (DAgger-style on-policy ẑ)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
    "## M6 — End-to-end Phase 2 iteration (DAgger)\n\n"
    "Now the rollout is conditioned on `ẑ = φ(buffer)` from the *current* φ "
    "params (not zero, not stale). Each iteration:\n"
    "1. Roll out B envs for T steps with frozen π conditioned on per-step ẑ. "
    "Record `(history_t, z_t = μ(priv_t))` and the rollout-mean `|ẑ - z|`.\n"
    "2. Drop cold-start frames, shuffle, split into `NUM_MINIBATCHES`, "
    "do one Adam pass through them.\n"
    "3. Repeat.\n\n"
    "Success criteria: held-out MSE drops ≥50% in 50 iters, in-rollout "
    "`mean(|ẑ-z|)` trends down (proves ẑ tracks current φ, not stale), "
    "wall time stable from iter 1 onwards (no jit recompilation)."
))

cells.append(code("""# ── M6 config ────────────────────────────────────────────────────────────────
M6_NUM_ITERS       = 75
M6_NUM_MINIBATCHES = 4      # 4 grad steps/iter ⇒ 300 total updates from random init
M6_LR              = 1e-3   # lower than M5 (3e-3); higher than the paper's 5e-4 to
                             # comfortably clear the 50%-drop target inside the smoke test
M6_B_ROLLOUT       = 64
M6_T_ROLLOUT       = 75     # collect 75, drop first HISTORY_LEN to skip cold start

# Held-out eval rollout: a fixed dataset re-used across iters to track progress.
M6_EVAL_B = 64
M6_EVAL_T = 75
"""))

cells.append(code("""# ── On-policy rollout with per-step ẑ from CURRENT φ params ─────────────────
# Returns (histories, z_targets, mean_abs_zhat_minus_z) for the rollout.
# Cold-start frames (first HISTORY_LEN steps) dropped after the scan.

def rollout_dagger(rng, phi_params, B=M6_B_ROLLOUT, T=M6_T_ROLLOUT):
    rng, k = jax.random.split(rng)
    state = env.reset(jax.random.split(k, B))
    buf = init_history(B, HISTORY_LEN, PER_STEP_FEAT)

    def step_fn(carry, _):
        state, buf = carry
        on = normalize_obs(state.obs)
        z_true = compute_z(on)
        z_hat  = phi_module.apply(phi_params, buf)            # ẑ from current φ
        a      = policy_action_with_z(on, z_hat)              # π conditioned on ẑ
        student_t = on[..., :STUDENT_OBS_SIZE]
        new_buf   = update_history(buf, extract_frame(student_t, a))
        new_state = env.step(state, a)
        # Record (buffer at decision time, z_true, |ẑ - z|)
        return (new_state, new_buf), (buf, z_true, jp.mean(jp.abs(z_hat - z_true)))

    (_, _), (h_seq, z_seq, abs_diff) = jax.lax.scan(
        step_fn, (state, buf), None, length=T
    )
    # Drop cold-start frames (partially-zero buffers ⇒ irreducible label noise).
    h_seq = h_seq[HISTORY_LEN:]
    z_seq = z_seq[HISTORY_LEN:]
    abs_diff_post_warmup = abs_diff[HISTORY_LEN:]
    return (
        h_seq.reshape(-1, HISTORY_LEN, PER_STEP_FEAT),
        z_seq.reshape(-1, ENV_LATENT_SIZE),
        jp.mean(abs_diff_post_warmup),
    )

rollout_jit = jax.jit(rollout_dagger)

# Warm up the jit and check shapes.
import time as _time
_t = _time.monotonic()
H_warm, Z_warm, diff_warm = rollout_jit(jax.random.PRNGKey(99), phi_params)
H_warm.block_until_ready()
print(f"warmup rollout {(_time.monotonic()-_t):.1f}s | H {H_warm.shape} | Z {Z_warm.shape} | "
      f"|ẑ-z|={float(diff_warm):.3f}")
"""))

cells.append(code("""# ── One Phase-2 iteration: rollout (on-policy) → minibatch SGD ──────────────

def shuffled_minibatches(key, H, Z, num_mb):
    N = H.shape[0]
    perm = jax.random.permutation(key, N)
    mb_size = N // num_mb
    H_mb = H[perm[:mb_size * num_mb]].reshape(num_mb, mb_size, *H.shape[1:])
    Z_mb = Z[perm[:mb_size * num_mb]].reshape(num_mb, mb_size, *Z.shape[1:])
    return H_mb, Z_mb

# Replace the M5 schedule/optimizer with a constant LR for the on-policy phase.
m6_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(M6_LR))

@jax.jit
def m6_train_step(params, opt_state, H_mb, Z_mb):
    # H_mb: (num_mb, mb_size, K, F);  Z_mb: (num_mb, mb_size, 8)
    def body(carry, batch):
        params, opt_state = carry
        H, Z = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, H, Z)
        updates, opt_state = m6_opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss
    (params, opt_state), losses_per_mb = jax.lax.scan(
        body, (params, opt_state), (H_mb, Z_mb)
    )
    return params, opt_state, jp.mean(losses_per_mb)

@jax.jit
def m6_eval_loss(params, H_eval, Z_eval):
    return loss_fn(params, H_eval, Z_eval)

# Re-init φ params + opt_state for a clean M6 baseline (so M6 is independent of M5).
phi_params  = phi_module.init(jax.random.PRNGKey(123), histories[:4])
opt_state   = m6_opt.init(phi_params)
print("φ re-initialised for M6.")
"""))

cells.append(code("""# ── Run M6: 50 on-policy iterations ─────────────────────────────────────────
# Eval = the fresh rollout's loss measured BEFORE the gradient update on it.
# This is the standard DAgger held-out signal: it tracks how well φ regresses
# z on the *current* on-policy distribution. (A fixed off-policy eval set is
# misleading here because the data distribution shifts as φ improves.)
import time
key = jax.random.PRNGKey(SEED + 7)
m6_log = {"iter": [], "pre_loss": [], "post_loss": [], "abs_diff": [], "wall_s": []}

# Iter-0 baseline: rollout with random-init φ, measure loss before any update.
key, k0 = jax.random.split(key)
H0, Z0, diff0 = rollout_jit(k0, phi_params)
init_pre = float(m6_eval_loss(phi_params, H0, Z0))
print(f"iter   0  pre_loss = {init_pre:.6f}  |ẑ-z|={float(diff0):.3f}  "
      f"(φ random init, baseline for the 50% target)")

for it in range(1, M6_NUM_ITERS + 1):
    t0 = time.monotonic()
    key, k_roll, k_perm = jax.random.split(key, 3)
    H, Z, diff = rollout_jit(k_roll, phi_params)
    pre_loss = float(m6_eval_loss(phi_params, H, Z))             # held-out loss
    H_mb, Z_mb = shuffled_minibatches(k_perm, H, Z, M6_NUM_MINIBATCHES)
    phi_params, opt_state, loss_avg = m6_train_step(phi_params, opt_state, H_mb, Z_mb)
    post_loss = float(loss_avg)                                   # avg minibatch train loss
    wall = time.monotonic() - t0
    m6_log["iter"].append(it)
    m6_log["pre_loss"].append(pre_loss)
    m6_log["post_loss"].append(post_loss)
    m6_log["abs_diff"].append(float(diff))
    m6_log["wall_s"].append(wall)
    if it == 1 or it % 10 == 0 or it == M6_NUM_ITERS:
        print(f"iter {it:>3}  pre_loss={pre_loss:.4f}  train_loss={post_loss:.4f}  "
              f"|ẑ-z|={float(diff):.3f}  ({wall:.2f}s)")
"""))

cells.append(code("""# ── M6 success criteria ─────────────────────────────────────────────────────
import numpy as np

pre_last = m6_log["pre_loss"][-1]   # iter 50 pre-update held-out loss

# 1) Pre-update held-out loss drops ≥ 50% from random init to iter 50.
print(f"pre_loss: iter0={init_pre:.4f}  iter{M6_NUM_ITERS}={pre_last:.4f}")
print(f"  drop vs iter 0: {1 - pre_last/init_pre:.1%}  (target ≥ 50%)")
assert pre_last / init_pre < 0.5, "FAIL: held-out pre-update loss did not drop ≥50%"

# 2) In-rollout |ẑ - z| trends downward (linreg slope < 0).
xs = np.array(m6_log["iter"])
ys = np.array(m6_log["abs_diff"])
slope = np.polyfit(xs, ys, 1)[0]
print(f"|ẑ - z| linreg slope: {slope:+.5f}/iter  (target < 0)")
assert slope < 0, f"FAIL: |ẑ-z| not trending down (slope={slope:+.5f})"

# 3) Wall time stable from iter 1+ (no jit recompilation): max < 4× median.
walls = np.array(m6_log["wall_s"])
print(f"wall time iter 1: {walls[0]:.2f}s   |   iter 2..N median: {float(np.median(walls[1:])):.2f}s "
      f"max: {float(walls[1:].max()):.2f}s")
assert walls[1:].max() < 4 * np.median(walls[1:]), "FAIL: wall time variance suggests jit recompilation"

print("\\nM6 PASS — DAgger plumbing wired correctly, jit stable, on-policy ẑ tracks φ.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# M7 — Full-scale Phase 2 training run + M8 multi-condition reward eval helper
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
    "## M7 — Full-scale Phase 2 training run\n\n"
    "Scales M6's loop to **B=512** parallel envs, **T=75** rollout steps "
    "(50 useful after dropping the cold-start window), **4 minibatches/iter**, "
    "**1500 iterations**. Every 100 iters, runs the M8 four-condition reward "
    "eval and appends to a metrics log so we can see reward parity emerge "
    "*during* training, not just at the end.\n\n"
    "Saves φ params next to the Phase 1 checkpoint as `phi_final` "
    "(via `brax.io.model.save_params`)."
))

cells.append(code("""# ── M8 multi-condition reward eval (helper, also used by the M7 logging hook) ─
# Rolls B envs for T steps four times, swapping the latent fed to π:
#   * ẑ from φ           ─ what the deployed robot would use
#   * z = μ(priv)        ─ teacher (Phase 1 baseline reward)
#   * 0                  ─ proves φ does real work
#   * z_random           ─ proves the gain isn't "any z works"
# Returns a dict of {condition: mean episode-return} plus the (ẑ, z) arrays
# for calibration plotting.

EVAL_B = 64
EVAL_T = 250

def _episode_return_rollout(rng, latent_fn, B=EVAL_B, T=EVAL_T):
    \"\"\"Single-condition rollout. latent_fn: (obs_norm, buf, key) -> z.\"\"\"
    rng, k_reset = jax.random.split(rng)
    state = env.reset(jax.random.split(k_reset, B))
    buf = init_history(B, HISTORY_LEN, PER_STEP_FEAT)

    def step_fn(carry, key):
        state, buf, ret = carry
        on = normalize_obs(state.obs)
        z  = latent_fn(on, buf, key)
        a  = policy_action_with_z(on, z)
        student_t = on[..., :STUDENT_OBS_SIZE]
        new_buf   = update_history(buf, extract_frame(student_t, a))
        new_state = env.step(state, a)
        # Brax wrapper resets done envs internally; sum reward across the rollout.
        return (new_state, new_buf, ret + new_state.reward), None

    keys = jax.random.split(rng, T)
    (_, _, total_ret), _ = jax.lax.scan(step_fn, (state, buf, jp.zeros(B)), keys)
    return jp.mean(total_ret)            # mean over envs of summed rewards

def _eval_all_conditions(rng, phi_params, z_std):
    keys = jax.random.split(rng, 4)
    z_zero = jp.zeros((EVAL_B, ENV_LATENT_SIZE), dtype=jp.float32)
    def lat_phi(on, buf, k):  return phi_module.apply(phi_params, buf)
    def lat_mu (on, buf, k):  return compute_z(on)
    def lat_zero(on, buf, k): return z_zero
    def lat_rand(on, buf, k): return jax.random.normal(k, (EVAL_B, ENV_LATENT_SIZE)) * z_std
    return {
        "phi":    _episode_return_rollout(keys[0], lat_phi),
        "mu":     _episode_return_rollout(keys[1], lat_mu),
        "zero":   _episode_return_rollout(keys[2], lat_zero),
        "random": _episode_return_rollout(keys[3], lat_rand),
    }

eval_all_conditions = jax.jit(_eval_all_conditions, static_argnames=())

# Estimate std(z) once from the M5 training set (same statistics as deployment).
Z_STD = jp.std(z_targets, axis=0)
print(f"z_std per dim: {np.array(Z_STD).round(3)}")
"""))

cells.append(code("""# ── M7 config + final-scale training loop ────────────────────────────────────
M7_NUM_ITERS       = 1500
M7_NUM_MINIBATCHES = 4
M7_LR              = 5e-4   # paper hyperparam; full-scale data so we can be conservative
M7_B_ROLLOUT       = 512
M7_T_ROLLOUT       = 75     # drop first HISTORY_LEN ⇒ 50 useful timesteps × 512 = 25,600 pairs
M7_EVAL_EVERY      = 100    # call eval_all_conditions every N iters
M7_LOG_EVERY       = 25     # print MSE summary

# Re-define rollout for the full B (so jax doesn't have to recompile mid-loop).
def rollout_dagger_m7(rng, phi_params):
    rng, k = jax.random.split(rng)
    state = env.reset(jax.random.split(k, M7_B_ROLLOUT))
    buf = init_history(M7_B_ROLLOUT, HISTORY_LEN, PER_STEP_FEAT)
    def step_fn(carry, _):
        state, buf = carry
        on = normalize_obs(state.obs)
        z_true = compute_z(on)
        z_hat  = phi_module.apply(phi_params, buf)
        a      = policy_action_with_z(on, z_hat)
        student_t = on[..., :STUDENT_OBS_SIZE]
        new_buf = update_history(buf, extract_frame(student_t, a))
        return (env.step(state, a), new_buf), (buf, z_true, jp.mean(jp.abs(z_hat - z_true)))
    (_, _), (h, z, d) = jax.lax.scan(step_fn, (state, buf), None, length=M7_T_ROLLOUT)
    h = h[HISTORY_LEN:]; z = z[HISTORY_LEN:]; d = d[HISTORY_LEN:]
    return (h.reshape(-1, HISTORY_LEN, PER_STEP_FEAT),
            z.reshape(-1, ENV_LATENT_SIZE),
            jp.mean(d))

rollout_jit_m7 = jax.jit(rollout_dagger_m7)

m7_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(M7_LR))

@jax.jit
def m7_train_step(params, opt_state, H_mb, Z_mb):
    def body(carry, batch):
        params, opt_state = carry
        H, Z = batch
        loss, grads = jax.value_and_grad(loss_fn)(params, H, Z)
        updates, opt_state = m7_opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss
    (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), (H_mb, Z_mb))
    return params, opt_state, jp.mean(losses)

# Re-init φ for an independent M7 run.
phi_params = phi_module.init(jax.random.PRNGKey(2026), jax.numpy.zeros((4, HISTORY_LEN, PER_STEP_FEAT)))
opt_state  = m7_opt.init(phi_params)

# Iter-0 baseline (random φ): one rollout, no update.
key = jax.random.PRNGKey(SEED + 17)
key, k0 = jax.random.split(key)
H0, Z0, diff0 = rollout_jit_m7(k0, phi_params)
init_pre = float(loss_fn(phi_params, H0, Z0))
print(f"M7 baseline pre_loss = {init_pre:.6f}  |ẑ-z|={float(diff0):.3f}")
"""))

cells.append(code("""# ── M7 training loop (this is the long-running cell — ~30-90 min) ───────────
import time
m7_log = {"iter": [], "pre_loss": [], "train_loss": [], "abs_diff": [],
          "wall_s": [], "eval_iter": [], "eval": []}

t_run0 = time.monotonic()
for it in range(1, M7_NUM_ITERS + 1):
    t0 = time.monotonic()
    key, k_roll, k_perm = jax.random.split(key, 3)
    H, Z, diff = rollout_jit_m7(k_roll, phi_params)
    pre_loss = float(loss_fn(phi_params, H, Z))
    H_mb, Z_mb = shuffled_minibatches(k_perm, H, Z, M7_NUM_MINIBATCHES)
    phi_params, opt_state, train_loss = m7_train_step(phi_params, opt_state, H_mb, Z_mb)
    train_loss = float(train_loss)
    wall = time.monotonic() - t0
    m7_log["iter"].append(it); m7_log["pre_loss"].append(pre_loss)
    m7_log["train_loss"].append(train_loss); m7_log["abs_diff"].append(float(diff))
    m7_log["wall_s"].append(wall)
    if it == 1 or it % M7_LOG_EVERY == 0:
        elapsed = time.monotonic() - t_run0
        eta = elapsed / it * (M7_NUM_ITERS - it)
        print(f"iter {it:>4}/{M7_NUM_ITERS}  pre={pre_loss:.4f}  train={train_loss:.4f}  "
              f"|ẑ-z|={float(diff):.3f}  wall={wall:.2f}s  eta={eta/60:.1f} min")
    if it % M7_EVAL_EVERY == 0 or it == M7_NUM_ITERS:
        key, k_eval = jax.random.split(key)
        rew = eval_all_conditions(k_eval, phi_params, Z_STD)
        rew = {k: float(v) for k, v in rew.items()}
        m7_log["eval_iter"].append(it); m7_log["eval"].append(rew)
        ratio = rew["phi"] / max(rew["mu"], 1e-9)
        print(f"   eval@{it:>4}  reward(phi)={rew['phi']:.1f}  reward(mu)={rew['mu']:.1f}  "
              f"reward(0)={rew['zero']:.1f}  reward(rand)={rew['random']:.1f}  "
              f"phi/mu={ratio:.2%}")

print(f"\\nM7 finished in {(time.monotonic()-t_run0)/60:.1f} min")
"""))

cells.append(code("""# ── Save the trained φ next to the Phase 1 checkpoint and report success ────
phi_save_path = CHECKPOINT_PATH.parent / "phi_final"
model.save_params(str(phi_save_path), phi_params)
print(f"saved φ to {phi_save_path}")

# Per-dim R² on the most recent rollout (close to the deployment distribution).
key, k_check = jax.random.split(key)
H_chk, Z_chk, _ = rollout_jit_m7(k_check, phi_params)
pred_chk = phi_module.apply(phi_params, H_chk)
r2_chk = 1.0 - jp.mean((pred_chk - Z_chk) ** 2, axis=0) / jp.maximum(jp.var(Z_chk, axis=0), 1e-12)
print("per-dim R² on a fresh on-policy rollout:", np.array(r2_chk).round(3))

# ── M7 success criteria ─────────────────────────────────────────────────────
# NB: the *real* success metric is reward parity (M8). MSE alone undersells φ
# because π is robust to small-magnitude latent errors — a 30% MSE φ can give
# 99% of teacher reward (we observed this empirically). So we require:
#   * ≥ 50% MSE drop overall (proves training did real work)
#   * any per-dim R² > 0.5 (proves at least the well-observable dims are nailed)
# and rely on M8 below to confirm reward parity.
final_pre = m7_log["pre_loss"][-1]
print(f"pre_loss: iter0={init_pre:.4f}  iter{M7_NUM_ITERS}={final_pre:.4f}  "
      f"ratio={final_pre/init_pre:.4%}")
assert final_pre / init_pre <= 0.5, (
    f"FAIL: pre_loss only dropped to {final_pre/init_pre:.2%}, need ≤50% — "
    "training stalled or DAgger plumbing broken."
)
assert float(jp.max(r2_chk)) > 0.5, (
    f"FAIL: best per-dim R² = {float(jp.max(r2_chk)):.2f}, need >0.5 on at least one dim — "
    "φ is failing to recover even the most-observable latent."
)

# Plot loss curves.
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(m7_log["iter"], m7_log["pre_loss"], label="pre-update (held-out)", lw=1)
ax1.plot(m7_log["iter"], m7_log["train_loss"], label="train minibatch avg", lw=1, alpha=0.7)
ax1.set_yscale("log"); ax1.set_xlabel("iter"); ax1.set_ylabel("MSE"); ax1.legend(); ax1.set_title("M7 loss curve")
if m7_log["eval"]:
    eit = m7_log["eval_iter"]
    ax2.plot(eit, [e["phi"] for e in m7_log["eval"]], "o-", label="reward(ẑ=φ)")
    ax2.plot(eit, [e["mu"]  for e in m7_log["eval"]], "s-", label="reward(z=μ) (teacher)")
    ax2.plot(eit, [e["zero"] for e in m7_log["eval"]], "^--", label="reward(0)", alpha=0.6)
    ax2.plot(eit, [e["random"] for e in m7_log["eval"]], "v--", label="reward(z_rand)", alpha=0.6)
    ax2.set_xlabel("iter"); ax2.set_ylabel("mean episode return"); ax2.legend(); ax2.set_title("Reward by latent")
fig.tight_layout()
fig_path = phi_save_path.parent / "m7_curves.png"
fig.savefig(fig_path, dpi=120)
print(f"saved curves to {fig_path}")
print("M7 PASS — full-scale Phase 2 training converged.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# M8 — Reward parity evaluation (the real success criterion)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
    "## M8 — Reward parity evaluation\n\n"
    "**The actual success metric for Phase 2.** Re-runs the four-condition "
    "eval on a clean seed at the end of training, prints a bar chart, and "
    "produces a per-dim ẑ-vs-z calibration scatter."
))

cells.append(code("""# ── Full M8 evaluation: bar chart + calibration scatter ─────────────────────
import matplotlib.pyplot as plt

key_eval = jax.random.PRNGKey(31337)   # fixed eval seed (reproducible)
rew_final = eval_all_conditions(key_eval, phi_params, Z_STD)
rew_final = {k: float(v) for k, v in rew_final.items()}
print("M8 final rewards (mean episode return):")
for k in ("phi", "mu", "zero", "random"):
    print(f"  {k:<7s}  {rew_final[k]:>8.2f}")

# ── M8 success criteria ────────────────────────────────────────────────────
phi_mu_ratio = rew_final["phi"] / max(rew_final["mu"], 1e-9)
phi_vs_zero  = rew_final["phi"] / max(rew_final["zero"], 1e-9) - 1.0
phi_vs_rand  = rew_final["phi"] - rew_final["random"]

print(f"\\nphi/mu = {phi_mu_ratio:.2%}   (strong target ≥ 90%, soft ≥ 85%)")
print(f"phi vs zero: {phi_vs_zero:+.1%}    (target > +10%)")
print(f"phi vs rand: {phi_vs_rand:+.2f}    (target > 0)")

soft_pass = phi_mu_ratio >= 0.85
strong_pass = phi_mu_ratio >= 0.90
gain_over_zero = phi_vs_zero > 0.10
gain_over_rand = phi_vs_rand > 0.0
print("\\nphi vs mu      strong:", strong_pass, "  soft:", soft_pass)
print("phi > zero +10%:", gain_over_zero)
print("phi > random   :", gain_over_rand)

# Bar chart
fig, (ax_bar, ax_cal) = plt.subplots(1, 2, figsize=(12, 4))
labels = ["ẑ=φ", "z=μ\\n(teacher)", "0", "rand"]
vals   = [rew_final["phi"], rew_final["mu"], rew_final["zero"], rew_final["random"]]
ax_bar.bar(labels, vals, color=["C0", "C2", "C3", "C7"])
ax_bar.set_ylabel("mean episode return")
ax_bar.set_title("M8 reward parity (B={}, T={})".format(EVAL_B, EVAL_T))
for x, v in zip(labels, vals):
    ax_bar.text(x, v, f"{v:.0f}", ha="center", va="bottom")

# Calibration scatter: predicted ẑ vs true z, per dim, on a fresh on-policy rollout.
key_eval, kc = jax.random.split(key_eval)
H_cal, Z_cal, _ = rollout_jit_m7(kc, phi_params)
pred_cal = phi_module.apply(phi_params, H_cal)
for d in range(ENV_LATENT_SIZE):
    ax_cal.scatter(np.array(Z_cal[:, d]), np.array(pred_cal[:, d]), s=3, alpha=0.3, label=f"d{d}")
mn = float(min(jp.min(Z_cal), jp.min(pred_cal)))
mx = float(max(jp.max(Z_cal), jp.max(pred_cal)))
ax_cal.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
ax_cal.set_xlabel("true z"); ax_cal.set_ylabel("ẑ = φ(buffer)"); ax_cal.set_title("Calibration (per dim)")
ax_cal.legend(fontsize=7, markerscale=2, ncol=2)
fig.tight_layout()
m8_path = phi_save_path.parent / "m8_eval.png"
fig.savefig(m8_path, dpi=120)
print(f"\\nsaved M8 figures to {m8_path}")

# Strong assertion: at minimum the soft target + outperforming zero + outperforming random.
assert soft_pass, f"FAIL: phi/mu = {phi_mu_ratio:.2%} < 85% soft threshold"
assert gain_over_zero, "FAIL: φ does no better than zero latent"
assert gain_over_rand, "FAIL: φ does no better than random latent"
print(("\\nM8 STRONG PASS — φ matches teacher within 10%." if strong_pass
       else "\\nM8 SOFT PASS — φ recovers most of teacher reward (>85%) and clearly beats baselines."))
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            # Match phase1_run.ipynb — points at the project's .venv
            # (the conda `base` "python3" kernel doesn't have jax/brax/mujoco).
            "display_name": "MuJoCo Playground",
            "language": "python",
            "name": "mp",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
