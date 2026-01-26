# Transformable Wheel Robot

## Notes

* Using cylinders for wheels causes the warp backend to crash without an error message.

## DEMO

<img src="assets/robot_step.gif" alt="Robot climbing over a step" style="width: 400px; max-width: 100%;">

Run the demo training script with the following (assuming you have the environment setup done (see below) and you have the training script.):

```bash
./.venv/bin/python train_jax_ppo.py --env_name="TransformableWheelMobileRobot"
```


## Git Workflow

### Prerequisites

- Your system must have git installed.
- You should install [Pixi](https://pixi.sh/latest/).

Although a bit confusing, we'll be using both Pixi and uv. You should think of Pixi as a package manager for any tools we'll need, and uv as the Python-only package/workspace manager. We'll mostly use Pixi to install uv, and then use uv to manage our Python packages.

### Intial Setup

```bash
# This only needs to be done once and you've likely done it already
git config --global user.name "YOUR NAME"
git config --global user.email "YOU@EXAMPLE.COM"

# Clone the repository
git clone https://github.com/JacobBau04/Transformable-Leg-Wheel-Robot
```

### Work cycle

#### For Smaller Changes

For smaller changes you can commit directly to main.

```bash
# Make changes to the code and then (repeat as is useful)
git add FILES_YOU_CHANGED/ADDED
git commit -m "SHORT DESCRIPTIVE MESSAGE"

# Pull latest changes from main (this will fetch and merge)
git pull origin main

# Cleanup any merge conflicts and then push changes to main
git push origin main
```

#### For Larger Changes

I recommend creating a branch when you are working on a large change (a new feature, a bug fix, etc).
This keeps your changes isolated from the main branch that we all use until they are well tested and ready to be merged in.

```bash
# Create and switch to a new branch named
git checkout -b feat/SHORT-DESCRIPTIVE-NAME

# Make changes to the code and then (repeat as is useful)
git add FILES_YOU_CHANGED/ADDED
git commit -m "SHORT DESCRIPTIVE MESSAGE"

# Push branch to remote repository
git push -u origin feat/SHORT-DESCRIPTIVE-NAME

# Continue working on the feature branch until ready to merge
# Open a Pull Request (PR) on GitHub targeting main and then send a message to have it reviewed
```

## Project Setup

### Initial Setup

You shouldn't have to do any of this if you clone the repository. These are notes for the from-scratch setup.

```bash
# These commands only need to be run once to setup the project

# Assuming the directory was already created or cloned
cd Transformable-Leg-Wheel-Robot
pixi init .
pixi add uv

# Create a virtual environment with Python 3.11 and create the twmr package
# This will create a uv workspace and virtual environment
pixi run uv init --python 3.11 --bare
pixi run uv init --package packages/twmr

# Install playground dependencies and then playground
# First grab tool.uv.indexs and tool.uv.sources from mujoco_playground
pixi run uv add "jax[cuda12]"
# test: .venv/bin/python -c "import jax; print(jax.default_backend())" --> gpu
pixi run uv add warp-lang
pixi run uv pip install "git+https://github.com/google-deepmind/mujoco_playground.git"
# test: .venv/bin/python -c "import mujoco_playground" --> no warnings
# Now manually add to the pyproject.toml file
```

### Setup From Clone

```bash
# Initial setup process on the OnDemand sever

# 1. Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Install and authenticate gh (you might need to create an authentication token)
pixi global install gh
gh auth login

# 3. Clone the repository
gh repo clone JacobBau04/Transformable-Leg-Wheel-Robot

# 4. Setup the project
cd Transformable-Leg-Wheel-Robot
pixi run uv sync
source .venv/bin/activate
python -m ipykernel install --name mp --display-name "MuJoCo Playground" --user

# Now you're ready to go!
```
