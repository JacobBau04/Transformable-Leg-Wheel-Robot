from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = Path("trans_wheel_robo2_2FLAT.xml")

# ignore image implementation for now


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,  # 50 hz control
        sim_dt=0.002,  # 500 hz physics
        episode_length=1000,  # 20 seconds
        action_repeat=10,  # ratio of 0.02 / 0.002
        impl="jax",  # use warp? yes
        nconmax=100,  # allow collisions
        njmax=500,  # allow complex joints
    )


class TransformableWheelMobileRobot(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:  # find chassis to track speed and orientation
        self._root_body_id = self._mj_model.body("root").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_init = jax.random.split(rng)

        # randomize initial position
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(0.2)
        qpos = qpos + 0.01 * jax.random.normal(rng_init, qpos.shape)

        qvel = jp.zeros(self.mjx_model.nv)

        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data, {})

        metrics = {
            "reward/forward_vel": jp.array(0.0),
            "reward/survival": jp.array(0.0),
            "reward/energy": jp.array(0.0),
            "reward": jp.array(0.0),
        }

        info = {"rng": rng}

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, rng_next = jax.random.split(state.info["rng"])

        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        forward_vel = data.qvel[0]
        reward_vel = forward_vel
        reward_survival = jp.array(0.1)
        reward_energy = -0.001 * jp.sum(jp.square(action))

        reward = reward_vel + reward_survival + reward_energy

        chassis_height = data.qpos[2]
        done = (chassis_height < 0.1) | jp.isnan(data.qpos).any()
        done = done.astype(jp.float32)

        obs = self._get_obs(data, state.info)

        metrics = state.metrics.copy()
        metrics["reward/forward_vel"] = reward_vel
        metrics["reward/survival"] = reward_survival
        metrics["reward/energy"] = reward_energy
        metrics["reward"] = reward

        info = state.info.copy()
        info["rng"] = rng_next

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data: mjx.Data, info: dict = {}) -> jax.Array:
        # joint positions and velocities
        qpos = data.qpos[7:]
        qvel = data.qvel[6:]
        orientation = data.qpos[3:7]
        return jp.concatenate([qpos, qvel, orientation])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


dm_control_suite.register_environment(
    env_name="TransformableWheelMobileRobot",
    env_class=TransformableWheelMobileRobot,
    cfg_class=default_config,
)
