# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavior & Communication

Before taking any action (editing a file, running a command, refactoring code, etc.):
1. **Explain what you're about to do and why** — what problem it solves, what tradeoff it makes, and how it fits into the broader codebase.
2. **Call out assumptions** — if you're inferring intent or filling in a gap, say so explicitly.
3. **Reference relevant context** — e.g., "because `ctrl_dt=0.02s` and `sim_dt=0.01s`, this means..." rather than just making the change silently.

When making non-trivial changes, structure your response as:
- **Goal**: what we're trying to achieve
- **Approach**: the specific strategy and why it was chosen over alternatives
- **Changes**: what files/lines are being touched and why each one matters

## Environment Setup

This project uses **Pixi** (for tooling) and **uv** (for Python packages). Always prefix Python/pip commands with `pixi run`.

```bash
# Sync/install all dependencies
pixi run uv sync

# Activate the virtual environment (for interactive use)
source .venv/bin/activate
```

The virtual environment is at `.venv/`. Use `.venv/bin/python` to run scripts directly without activating.

## Running Code

```bash
# Run the demo training
./.venv/bin/python train_jax_ppo.py --env_name="TransformableWheelMobileRobot"

# Verify JAX sees the GPU (should print "gpu")
./.venv/bin/python -c "import jax; print(jax.default_backend())"

# Verify mujoco_playground is installed correctly (no warnings)
./.venv/bin/python -c "import mujoco_playground"
```

## Dependency Management

```bash
# Upgrade a specific package (e.g., when a pinned version fetch fails)
pixi run uv sync --upgrade-package mujoco

# Recreate dependency error (ignoring cache, for debugging)
pixi run uv --no-cache sync --upgrade-package mujoco
```

Custom PyPI indexes are configured in `pyproject.toml`:
- MuJoCo: `https://py.mujoco.org`
- NVIDIA (warp-lang): `https://pypi.nvidia.com`

## Code Style

Ruff is configured with a **96-character line length** (`pyproject.toml`). Python >=3.11 is required.

## Architecture

### Package Structure

```
packages/twmr/              # Main installable package (uv workspace member)
  src/twmr/
    __init__.py             # Exports TransformableWheelMobileRobot
    twmr.py                 # Core environment class
  assets/
    wmr-spheres.xml         # Primary robot model (spherical wheels, simpler)
    wmr-cylinders.xml       # Cylindrical wheel variant
    trans_wheel_robo2_2*.xml # Full transformable leg-wheel models (FLAT/BOX/GEN_TERR terrain)

sandbox/                    # Experimental scripts (not part of the package)
  Demo.ipynb                # Environment registration demo and simulation rollout
  isabella/                 # Alternative env + full PPO training script (train_twmr.py, play_twmr.py)
  jacob/                    # PPO training variants with terrain support (train_transform_PPO*.py)
```

### Core Environment (`packages/twmr/src/twmr/twmr.py`)

`TransformableWheelMobileRobot` extends `MjxEnv` from `mujoco_playground` and is registered in the `dm_control_suite` registry under the name `"TransformableWheelMobileRobot"`.

Key design points:
- **Model**: loads `wmr-spheres.xml` (4-wheeled robot with wheel joints + motor actuators)
- **Simulator backend**: defaults to `impl="warp"` (NVIDIA Warp); `ctrl_dt=0.02s`, `sim_dt=0.01s`
- **Observation**: concatenation of `qpos` and `qvel` (joint positions and velocities)
- **Reward**: currently returns `0.0` — the reward function is a TODO
- **Done condition**: NaN in `qpos` or `qvel`

### RL Training

Sandbox scripts use **Brax PPO** for training. Key differences between sandbox implementations:
- `sandbox/isabella/train_twmr.py`: flag-based config, W&B/TensorBoard logging, domain randomization, vision support, checkpoint save/load
- `sandbox/jacob/train_transform_PPO3_*.py`: multi-terrain support (FLAT, BOX, GEN_TERR), complex reward with control cost and leg extension penalty, 8 actuators (4 wheels + 4 leg motors)

### Key TODOs in the Codebase

- Implement `_compute_reward_and_metrics()` in `twmr.py`
- Randomize initial `qpos`/`qvel` in `reset()`
- Get the initial position from the XML for environment resets
- Decide on `action_repeat` relationship to `ctrl_dt`/`sim_dt`
