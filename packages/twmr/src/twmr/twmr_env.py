from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
from pathlib import Path

_XML_PATH = Path("trans_wheel_robo3_3FLAT.xml") 

#ignore image implementation for now

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,        #50 hz control
        sim_dt=0.002,        #500 hz physics
        episode_length=1000, #20 seconds
        action_repeat=10,    #ratio of 0.02 / 0.002
        impl="jax",          #use warp? yes
        nconmax=100,         #allow collisions
        njmax=500,           #allow complex joints
        #domain randomization
        # friction_range=(0.5, 1.2),
        # mass_range=(0.8, 1.2),
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

    def _post_init(self) -> None: #find chassis to track speed and orientation
        self._root_body_id = self._mj_model.body("root").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1 = jax.random.split(rng, 2)

        #randomize initial position
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(0.2) #lifted so it doesn't spawn in floor
        
        #noise
        qpos = qpos + 0.01 * jax.random.normal(rng1, (self.mjx_model.nq,))
        #might not be worth randomizing initial velocity
        qvel = jp.zeros(self.mjx_model.nv)

        #CHANGED: create standard MJX data structures bc mjx_env didn't work
        data = mjx.make_data(self.mjx_model)
        
        #set the initial position and velocity
        data = data.replace(qpos=qpos, qvel=qvel)
        
        data = mjx.forward(self.mjx_model, data)

        #initialize metrics to zero
        metrics = {
            "reward/forward_vel": jp.zeros(()),
            "reward/survival": jp.zeros(()),
            "reward/energy": jp.zeros(()),
        }
        
        info = {"rng": rng}
        reward, done = jp.zeros(2)
        
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        
        reward = self._get_reward(data, action, state.info, state.metrics)
        
        obs = self._get_obs(data, state.info)
        
        #check z-height of chassis to see if fallen
        chassis_height = data.qpos[2] 
        done = (chassis_height < 0.1) | jp.isnan(data.qpos).any()
        done = done.astype(float)

        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        
        #joint positions and velocities
        qpos = data.qpos[7:]
        
        qvel = data.qvel[6:]

        orientation = data.qpos[3:7] 

        return jp.concatenate([qpos, qvel, orientation])

    def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict, metrics: dict) -> jax.Array:
        
        #forward vel reward
        forward_vel = data.qvel[0]
        reward_vel = forward_vel * 1.0
        metrics["reward/forward_vel"] = reward_vel
        
        #survival reward
        reward_survival = 0.1
        metrics["reward/survival"] = reward_survival

        #energy penalty
        reward_energy = -0.001 * jp.sum(jp.square(action))
        metrics["reward/energy"] = reward_energy

        return reward_vel + reward_survival + reward_energy

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