# Phase 3 — Async deployment scaffolding (design notes)

This file documents the contract that Phase 3 needs to
satisfy so that the robot can run φ + π on its own compute without privileged
information. The corresponding pieces in Phase 2 were built to match this
contract; this file traces those pieces so Phase 3 can pick them up directly.

---

## Deployment topology

Two processes, two clocks:

```
                 ┌──────────────── robot CPU ────────────────┐
sensors ──50 Hz──►  obs preprocessing  ──►  ring buffer       │
                 │                                            │
                 │                  φ (async, ~10 Hz)         │
                 │                       │  ẑ (latched)       │
                 │  ▼                                          │
                 │  π (synchronous, 50 Hz) ──► motor commands ─┼──► motors
                 └──────────────────────────────────────────────┘
```

* **π is synchronous** with the control loop (50 Hz). One forward pass per
  sensor frame. Latency budget ~5 ms.
* **φ is asynchronous and slower**. Reads the latest history from the ring
  buffer (no copy), produces ẑ, latches it. Target rate ~10 Hz; π keeps using
  the previously latched ẑ between updates. This is the same pattern Kumar et
  al. use on the A1.

The async split means φ can be heavier than the control budget allows
synchronously, and we can drop a φ tick under load without losing the control
loop. φ doesn't have to keep up with π — it just has to keep ẑ "fresh
enough" relative to environment dynamics, which change on the seconds
timescale.

---

## What Phase 2 already nailed down

| Concern | Where it lives in Phase 2 | What carries over |
|---|---|---|
| Per-step frame definition `[student_obs, action]`, dtype, shape | `sandbox/phase2/rollout_utils.py::extract_frame` | Identical concat order. Robot must emit frames the same way. |
| Ring buffer semantics — newest at index −1, oldest at 0 | `rollout_utils.py::update_history` (`jp.roll(-1) + .at[-1].set(frame)`) | Phase 3 buffer is the same logical layout, just implemented as a contiguous numpy/torch tensor with a write-index pointer instead of `jp.roll` (faster on CPU). Conv sees the same temporal ordering. |
| HISTORY_LEN = 25 (= 0.5 s of 50 Hz history) | `packages/twmr/src/twmr/adaptation.py` | Hard constraint; baked into φ's first windowed-Dense layer. |
| φ's input contract: `(B, 25, 37) float32` | `twmr.adaptation.AdaptationModule` | At deployment B = 1; same shape otherwise. |
| Inference path that skips μ entirely | `twmr.networks.TeacherPolicyNetwork.apply_with_z` (validated bit-exact in M4) | This is the **only** policy method Phase 3 should call. The full `TeacherPolicyNetwork.__call__` takes the full obs and runs μ — that path **must not be exposed to the robot** (it would need privileged obs the robot doesn't have). |
| Observation normalization | `running_statistics.normalize(obs, norm_params)` from `brax.training.acme` | The norm_params live in the saved Phase 1 checkpoint (`ppo_final`); they're an array, not a learned module. Phase 3 needs them frozen and applied before the obs hits π or the buffer. |

---

## Robot-compute concerns

### Latency budget

Numbers from the V100 dev rig, B=1:

* μ + π forward (apply_with_z): tens of microseconds.
* φ forward at B=1: ~16 K params, all matmul. <1 ms on a Jetson-class CPU
  with no SIMD; faster with NEON. Even the ring-buffer copy dominates.

So the real concern isn't "can φ run on the robot" — it's "do we trigger φ
on a worker thread so its variable runtime doesn't stall π's deterministic
50 Hz loop". Latch the most-recent ẑ; π reads the latched value every tick.

### Export path

φ and π are both pure Flax/JAX modules with **no Brax wrappers**, **no
running_statistics state inside the graph**, and **no `EnvFactorEncoder` on
the deployment path**. That makes export straightforward:

* Convert φ params (a pytree of arrays) to `jax2tf` then to TFLite, or to
  ONNX via `jax.export` + `onnxruntime`. The graph is trivial: 4 dense
  layers + a few reshapes + ReLU/tanh.
* Convert π via the same path, but use **`apply_with_z`** as the entry
  point — *not* `__call__`. That ensures the exported graph does not depend
  on privileged obs. (Bit-exact equivalence to the standard path with true z
  was verified in M4.)
* Normalization is a fixed `(mean, var)` array — load alongside the model
  and inline before the network call. Don't try to embed
  `running_statistics.normalize` into the exported graph (it carries
  bookkeeping state we don't need at inference).

### What does NOT carry over

* The MJX env. Obviously.
* `EnvFactorEncoder` (μ). Used only at training time to produce supervision
  z. Don't ship it.
* `TeacherValueNetwork`. Critic; training-only.
* The Brax PPO `network_factory` plumbing. The robot only needs raw Flax
  apply functions, plus norm_params, plus φ params, plus π params.

---

## Validation checklist before flashing

1. Load `ppo_final` and `phi_final`.
2. Render a deterministic sim rollout using **only** `apply_with_z(...)`
   (i.e., never call `TeacherPolicyNetwork.__call__`). Reward should match
   the M8 phi-condition number to within seed noise.
3. Mock the ring buffer as a pure numpy `(25, 37)` tensor with a manual
   write index. Feed φ the buffer one step at a time, compare ẑ to the
   jit-scanned version in this notebook for the same seed. Should match
   within float32 noise.
4. Profile φ latency on the actual robot CPU. Target: φ tick rate at least
   2 Hz (one buffer turnover every 12.5 control steps); ideally 10 Hz.
