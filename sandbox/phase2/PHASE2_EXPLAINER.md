# RMA in this repo, from scratch

This doc walks through how Rapid Motor Adaptation is implemented in this repo, 
so a future reader can understand the system end-to-end without reading every file. It 
assumes you know what a neural network is and can read Python; it doesn't assume 
you know PPO, DAgger, or RMA specifically.

If you want the deployment design (Phase 3), see `PHASE3_NOTES.md` instead.

---


## TODO

1. Check that domain randomization parameters are actually being implemented and how

## 0. Test phase 2

Run these two commands in terminal:

```bash
cd /home/imeg2025/Transformable-Leg-Wheel-Robot/sandbox/phase2
/home/imeg2025/Transformable-Leg-Wheel-Robot/.venv/bin/python test_phase2.py
```

(The full path is required — `python test_phase2.py` from the conda `base`
env will fail with `ModuleNotFoundError: No module named 'jax'` because
JAX/Brax/MuJoCo only live in the project's `.venv`, not in conda.)

**What the test does** (~30 s on GPU): rolls out the saved policy in 64
parallel envs for 250 steps, **four times**, swapping the latent fed to π:

1. `ẑ` from φ — what the deployed robot would use
2. `z = μ(priv_obs)` — the teacher (Phase 1 baseline)
3. `0` — proves φ is doing real work
4. `z_random ~ N(0, std(z))` — proves the gain isn't "any z works"

Prints the four mean episode-returns, asserts `reward(φ) ≥ 0.85·reward(μ)`,
`reward(φ) > reward(0) by 10%+`, and `reward(φ) > reward(random)`. Also writes
a bar chart + per-dim ẑ-vs-z calibration scatter to `m8_eval.png` next to
the loaded checkpoints.

**What it loads.** Auto-finds the most recent run dir matching
`sandbox/logs/TWMRLegFlat-*/checkpoints/` and loads two checkpoint files
from it:

| File | Size | Loaded with | Contents the test uses |
|---|---|---|---|
| `ppo_final` | ~136 KB | `model.load_params()` returns 3-tuple `(norm_params, policy_params, value_params)` | `norm_params` = obs normalisation stats; `policy_params["params"]["EnvFactorEncoder_0"]` = **μ** weights; the rest of `policy_params` = **π** MLP weights. `value_params` is loaded but discarded. |
| `phi_final` | ~67 KB | `model.load_params()` returns the φ params pytree | **φ** weights — the adaptation module trained in Phase 2. |

So *all three trained networks* (π, μ, φ) live in those two files; μ rides
inside the policy checkpoint because Flax saves submodule params nested.
Nothing else needs to be loaded for the test — the env is built fresh from
the XML on each run.

---

## 1. The problem RMA solves

You want a controller for a robot that works in the **real world**, where you
don't know friction or motor strengths. But you can train in
**simulation**, where the simulator knows everything.

The naive option — train one policy in sim with domain randomization and hope
for the best — produces conservative, sluggish policies. The policy can't behave
optimally for any specific friction value because it has no idea which friction
it's in.

RMA's idea: split the problem in two.

- **Phase 1 — Teacher.** Train a policy that *does* know the privileged stuff.
  This gives a strong, non-conservative policy.
- **Phase 2 — Student.** Train a small network to *estimate* the privileged
  information from observable history (recent state + action sequence) alone.
  At deployment, run estimator + Phase 1 policy with no privileged info.

The bet: the *consequences* of the privileged stuff (how the robot's state
actually evolves under your commands) leak enough info to reconstruct what the
privileged values must be. A robot on slippery ground responds to wheel
commands differently than one on grippy ground, and the recent history reveals
which world you're in.

---

## 2. The robot, env, observations, action

All in `packages/twmr/src/twmr/twmr.py`.

**The robot.** Transformable leg-wheel robot — 4 wheels, 4 extending legs.
8 actuators total (`_NUM_ACTUATORS = 8`).

**Action space — `action_size = 8`, range `[-1, 1]`.**

| slice | meaning | mapping |
|---|---|---|
| `action[0:4]` | desired wheel angular velocity | scaled by `_WHEEL_MAX_SPEED = 8.0` rad/s (twmr.py:233) |
| `action[4:8]` | desired leg position | mapped to `[_LEG_MIN, _LEG_MAX]` = `[-1.047, 3.427]` rad (twmr.py:244) |

**How the policy action becomes torque** (`step()`, twmr.py:228-260):

- *Wheels*: a single **P controller** on velocity error. `wheel_torque = Kp * (desired_vel − actual_vel)`. (`Kd = 0.0` currently — so it's literally a P controller, not PD.)
- *Legs*: a **cascaded** controller — outer loop is P on position error → "desired velocity"; inner loop is P on velocity error → torque. Both Kd terms are 0. So it's two nested P controllers
- After the controller computes torque: multiply by the per-episode `motor_strength` scalar (a privileged DR parameter), then clip to the actuator limit. **This is where motor-strength domain randomization actually enters the dynamics.**

**Observation space.**

- `student_obs` (29 dims, `_get_obs()`, twmr.py:329-351):
  `[accel(3), gyro(3), projected_gravity(3), leg_pos(4), wheel_vel(4), leg_vel(4), prev_action(8)]`.
  **Note**: `prev_action` is *part of* `student_obs`. It's not a separate input.
- `privileged_obs` (9 dims, twmr.py:127-132):
  `[friction(1), motor_strengths(8)]`. Only the simulator knows these.
- `teacher_obs = concat(student_obs, privileged_obs) = 38 dims`. This is what
  `state.obs` actually is during training.

**Domain randomization** (`_sample_domain_params()`, twmr.py:99-125, ranges twmr.py:31-34):

| Param | Range | Sampled per |
|---|---|---|
| `friction` | U[0.7, 1.3] | episode reset |
| `wheel_motor_strength` | U[0.85, 1.15]^4 | episode reset |
| `leg_motor_strength` | U[0.85, 1.15]^4 | episode reset |

Constant within an episode, varying between episodes. **The DR ranges define
what the final policy will be robust to.** If you later care about rougher
terrain or weaker motors, widen these *before* re-training Phase 1.

**Reward** (twmr.py:266-272):

```
reward = vx − 0.0005·||torque||² − 0.01·sum(leg_extension_penalty)
```

"Go forward, don't burn torque, don't keep legs extended". The privileged
parameters don't appear in the reward — they're parameters of the **dynamics**,
not of the goal.

**Done conditions**: base height < 0, severe tilt, base ang-vel > 20 rad/s,
NaN. Basically "fell over or blew up".

---

## 3. Phase 1 — Teacher policy, trained with PPO

Code: `sandbox/phase1_run.ipynb` + `packages/twmr/src/twmr/networks.py`.

**Three networks instantiated:**

1. **`EnvFactorEncoder μ`** (networks.py:19) — 3-layer MLP `9 → 64 → 64 → 8` (tanh, tanh, linear). Compresses the 9-dim `privileged_obs` into an 8-dim latent `z`. (`ENV_LATENT_SIZE = 8`.)
2. **`TeacherPolicyNetwork π`** (networks.py:32). Takes the full 38-dim `teacher_obs`. Internally: extracts the privileged slice, runs μ on it to produce `z`, concatenates `[student_obs, z]` (37 dims), runs through MLP `(64, 64, 64)` → outputs `2 * action_size = 16` numbers (8 means + 8 log-stds for the action distribution). Brax wraps this with `NormalTanhDistribution`.
3. **`TeacherValueNetwork V`** (networks.py:60) — same input pipeline as π (and instantiates its own copy of μ internally), outputs a scalar value estimate.

**Two μ instances during Phase 1.** One inside π, one inside V. They learn
*independent* params. We only ship the one inside π (saved as
`policy_params["params"]["EnvFactorEncoder_0"]`); the V-side encoder is
discarded with the value network.

**Why the policy gets z, not raw priv_obs.** Compressing through μ is a
regularization. It forces a low-dimensional bottleneck the policy has to use,
which is exactly the bottleneck φ later has to reproduce in Phase 2. The 8 dims
aren't pre-defined — μ learns to project priv_obs into whatever 8-dim space
best helps π maximize reward.

**What PPO actually does**:

PPO is **not** "regression on collected data". It's an actor-critic
policy-gradient algorithm:

1. *Roll out* the current π in vectorized parallel envs to collect trajectories `(obs, action, reward, next_obs)`. Brax does this on the GPU; default ~hundreds of parallel envs.
2. *Compute returns* (discounted future rewards) and *advantages* (how much better was each action than the value baseline predicted? — uses GAE).
3. *Update the actor π*: SGD on the **PPO clipped surrogate objective** — informally, "increase log-probability of actions that had positive advantage, but don't move too far from the old policy in one step (the clip)". This is policy gradient with a trust-region-ish constraint.
4. *Update the critic V*: SGD on **MSE regression to empirical returns**. **This is the regression part — only the value network does regression. The policy doesn't.**
5. μ gets gradients through both π and V. Its weights flow through every actor and critic update. So μ learns whatever projection of priv_obs is most useful to both maximizing reward (via π) and predicting returns (via V).

After ~tens of millions of env steps, the checkpoint contains:

- `norm_params` — running statistics for input normalization
- `policy_params` — π's MLP **and** μ's params (under `EnvFactorEncoder_0`)
- `value_params` — V's MLP and a separate μ (we don't ship this)

Saved via `model.save_params()` after every progress callback and at the end
of training. This is the M1 checkpoint round-trip.

---

## 4. Phase 2 — Student φ

Code: `packages/twmr/src/twmr/adaptation.py`, training in
`sandbox/phase2/train_phi.ipynb`.

Now π and μ are **frozen forever**. φ's job is to look at *observable* history
and reproduce what μ would have produced if it had access to the privileged obs.

**φ's architecture:**

- Input: a `(B, 25, 37)` window — 25 timesteps of `[student_obs(29), action(8)]` frames. Index 0 is oldest, index −1 is newest.
- Per-step MLP `37 → 64 → 32` (tanh) — applied independently to each timestep.
- Three "Conv1D" layers over the time axis (`[k=4,s=2][k=3,s=1][k=3,s=1]`), reducing temporal dim 25 → 11 → 9 → 7.
- Flatten 7×32=224 → Dense(8) → ẑ.
- 16,648 params total.

**Implementation note**: the "Conv1D" layers are actually implemented as
**windowed-Dense** (`_windows()` in `adaptation.py`) because cuDNN on V100
fails (status 5003) for these specific Conv1D shapes. Math is identical, params
are identical (16,648 either way), runs on GPU. Switch back to `nn.Conv` if you
move to A100/H100 hardware where cuDNN works for this shape.

**The training objective.** Element-wise MSE:

```python
loss = jp.mean((phi(history) - mu(priv_obs)) ** 2)
```

Adam optimizer. π and μ are frozen — no gradients flow into them.

**Important nuance**: the regression target is `μ(priv_obs)`, **not raw priv_obs**.
φ doesn't try to predict friction/motor-strengths directly. It predicts
whatever 8-dim representation μ found useful to π. Those dims are not
interpretable as "this is friction" — they're a learned latent space.

This matters for diagnosis: if the policy doesn't actually use, say, the
friction dim much, μ may collapse it to nearly-constant and φ trivially predicts
it. Conversely, if motor-strength variation matters a lot, μ devotes capacity
to it and φ has to work harder. The pattern of per-dim R² we see (some dims
R²~0.7, some R²~0.17) is exactly this — and the dims π cares less about don't
hurt downstream reward when φ predicts them poorly. **This is why MSE
undersells φ's quality, and reward parity (M8) is the bar that matters.**

**The non-trivial part: how the data is collected.**

This is the **DAgger** trick:

- **Wrong way (off-policy)**: roll out π conditioned on the *true* `z = μ(priv_obs)`. Save (history, true z). Train φ on it. Problem: at deployment you'd use π conditioned on ẑ, not z. The histories that distribution generates are different from what you trained φ on. φ overfits to a distribution it never sees in production.
- **Right way (on-policy DAgger)**: roll out π conditioned on the **current φ's estimate ẑ**. Save (history, true z). The history distribution now matches deployment because the policy is making decisions exactly as it will at deployment. After the rollout, update φ. Next iteration's rollout uses the just-updated φ.

This is M6's main success criterion: verify ẑ is being recomputed each step
with the *current* φ params. If you accidentally cache ẑ or use ẑ=0 during
data collection, the loop is wrong and you train against the wrong distribution.

**Per iteration** of the M7 full-scale loop:

- Collect 75 timesteps × 512 envs.
- Drop the first 25 timesteps from each (the buffer hadn't filled yet — it's
  still mostly zeros, which makes φ's input ambiguous between envs).
- → 25,600 (history, z) pairs.
- Shuffle, split into 4 minibatches, do 4 Adam steps.
- Repeat 1500 times.
- Save `phi_final` next to `ppo_final`.

---

## 5. Phase 3 — deployment

Designed but not yet implemented. See `PHASE3_NOTES.md` for the full contract.
One-sentence summary:

> π runs synchronously at 50 Hz calling **`apply_with_z(student_obs, latched_ẑ)`**;
> φ runs asynchronously on a worker thread at ~10 Hz updating the latched ẑ from
> a persistent ring buffer; the privileged encoder μ is **never** loaded on the
> robot.

The 4-condition reward eval (M8) is the sim-side proxy for "did we get the
contract right". On our final run: φ matched the teacher within seed noise
(100.9% of teacher reward), zero-latent baseline ran at ~23%, random latent at
~6%. So φ is doing real, structure-recovering work.

---

## 6. Common misunderstandings

These are the things people (including past-me) routinely got wrong about this
system:

- **PPO is not regression.** PPO is a policy-gradient method with a clipped
  surrogate objective. Only the *value network* (critic) does regression
  (regress to bootstrapped returns). The actor learns by policy gradient. The
  *training data* isn't `(state → optimal_action)` pairs — it's trajectories.

- **`prev_action` is part of `student_obs`.** It's not a separate input to π.
  See twmr.py:347-350.

- **During Phase 1, π is fed the *true* z = μ(priv_obs), not ẑ.** The estimated
  ẑ from φ only enters at Phase 2 data-collection rollouts (DAgger) and at
  deployment. Don't conflate the two latents — they have different sources
  even though they live in the same 8-dim space and the policy treats them
  identically.

- **φ regresses to `μ(priv_obs)`, not to `priv_obs`.** Don't expect φ's outputs
  to be human-interpretable as "this is friction" — they're a learned
  representation, not the raw physical parameters.

- **Wheel controller is P, not PD; leg controller is two nested P loops, not
  PD.** All Kd values are currently 0 in twmr.py.

- **Both π and V have their own internal μ during Phase 1.** Two encoders
  learn in parallel. We ship only π's.

- **`TeacherPolicyNetwork.__call__` requires the privileged slice and must
  not be used at deployment.** Use `apply_with_z(student_obs, z)` instead — M4
  verified the two paths are bit-exact when given the true z.

- **MSE undersells φ's quality.** Use M8 reward parity as the actual success
  bar; M5/M6/M7 MSE numbers are diagnostics.

- **Domain randomization is the lifeblood of the whole approach.** Without DR,
  μ would learn a constant (every episode looks the same to it), the policy
  would be brittle, and φ wouldn't have varied histories to learn from. The
  DR ranges in twmr.py:31-34 *define* what the final controller is robust to.
