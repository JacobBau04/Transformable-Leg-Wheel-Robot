"""Play with a trained PPO agent."""

import functools
import json
import os
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

from brax.io import model

from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

import twmr_env

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "TransformableWheelMobileRobot",
    "Name of the environment.",
)
#replaces all the training flags (lr, batch size)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from (required)", required=True
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_VIDEOS = flags.DEFINE_integer("num_videos", 1, "Number of videos to record.")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")

#network flags
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes", [256, 256, 256], "Policy hidden layer sizes"
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes", [256, 256, 256], "Value hidden layer sizes"
)

#suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    return dm_control_suite_params.brax_ppo_config(env_name) # removed IMPL.value bc only 1 positional arg taken
  raise ValueError(f"Env {env_name} not found.")

def main(argv):
  del argv

  #load env
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value
  env_cfg_overrides = {}
  
  ppo_params = get_rl_config(_ENV_NAME.value)

  env = registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides)

  #reconstructing neural network
  network_fn = ppo_networks.make_ppo_networks
  network_factory = functools.partial(
      network_fn, 
      policy_hidden_layer_sizes=list(map(int, _POLICY_HIDDEN_LAYER_SIZES.value)),
      value_hidden_layer_sizes=list(map(int, _VALUE_HIDDEN_LAYER_SIZES.value))
  )

  make_inference_fn = ppo.make_inference_fn(
      network_factory(
          observation_size=env.observation_size,
          action_size=env.action_size,
          preprocess_observations_fn=wrapper.preprocess_observations_fn
      )
  )

  #load params
  ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
  print(f"Restoring from: {ckpt_path}")
  
  params = model.load_params(ckpt_path)

  print("Starting inference...")

  #run the simulation
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  def do_rollout(rng, state):
    empty_data = state.data.__class__(**{k: None for k in state.data.__annotations__})
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = env.step(state, act)

      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.ctrl": state.data.ctrl,
      })
      return (state, rng), traj_data

    _, traj = jax.lax.scan(step, (state, rng), None, length=_EPISODE_LENGTH.value)
    return traj

  #render video
  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(env.reset))(rng)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [jax.tree.map(lambda x, j=j: x[j], t) for j in range(_EPISODE_LENGTH.value)]

  render_every = 2
  fps = 1.0 / env.dt / render_every
  print(f"FPS for rendering: {fps}")
  
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = env.render(traj, height=480, width=640, scene_option=scene_option)
    filename = f"rollout{i}.mp4"
    media.write_video(filename, frames, fps=fps)
    print(f"Rollout video saved as '{filename}'.")

if __name__ == "__main__":
  app.run(main)