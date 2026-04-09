from pathlib import Path
from typing import Any

import jax.numpy as jp
from jax import Array as JaxArray
from ml_collections import config_dict
from mujoco import MjModel, mjx  # type: ignore
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_utils
from mujoco_playground._src.dm_control_suite import common

ConfigOverridesDict = dict[str, str | int | list]
_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "trans_wheel_robo2_2FLAT.xml"

# Reward constants (matched to DemoModes.ipynb)
_LEG_MIN = -1.047                            # rad: contracted position
_LEG_MAX =  3.427                            # rad: fully extended position
_LEG_CENTER = (_LEG_MAX + _LEG_MIN) / 2.0   # 1.19 rad
_LEG_HALF_RANGE = (_LEG_MAX - _LEG_MIN) / 2.0  # 2.237 rad
_LEG_IDX = (8, 12, 16, 20)                  # qpos indices for the 4 leg joints
_CTRL_COST_W = 0.0005
_LEG_EXT_COST_W = 0.01


# def default_vision_config() -> config_dict.ConfigDict:
#     return config_dict.create(
#         gpu_id=0,
#         render_batch_size=512,
#         render_width=64,
#         render_height=64,
#         enable_geom_groups=[0, 1, 2],
#         use_rasterizer=False,
#         history=3,
#     )


# TODO: check all of these default values
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,  # 50 hz control
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,  # TODO: should this be a ratio of ctrl_dt / sim_dt?
        vision=False,
        # vision_config=default_vision_config(),
        impl="warp",  # TODO: cartpole uses jax
        nconmax=100,  # allow collisions
        njmax=500,  # allow complex joints
    )


class TWMRLegFlatRawTorques(MjxEnv):
    def __init__(
        self,
        # Task specific config
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        model_xml = _XML_PATH.read_text()
        self._model_assets = common.get_assets()
        self._mj_model: MjModel = MjModel.from_xml_string(model_xml, self._model_assets)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore
        self._mj_model.opt.timestep = self.sim_dt

        # TODO: figure out vision with the madrona batch renderer

        # TODO: what does this do for us exactly?
        # self._root_body_id = self._mj_model.body("root").id

    def reset(self, rng: JaxArray) -> State:
        # TODO: randomize initial state (qpos, qvel)
        # qpos = qpos.at[2].set(0.2)
        # qpos = qpos + 0.01 * jax.random.normal(rng_init, qpos.shape)

        # Initially reset to the original position
        # qpos = jp.zeros(self.mjx_model.nq)
        # qvel = jp.zeros(self.mjx_model.nv)

        data = mjx_env.make_data(
            self.mj_model,
            # qpos=qpos,
            # qvel=qvel,
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,  # type: ignore
            njmax=self._config.njmax,  # type: ignore
        )

        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward":                    jp.array(0.0),
            "reward/forward_vel":        jp.array(0.0),
            "reward/ctrl_cost":          jp.array(0.0),
            "reward/leg_extension_cost": jp.array(0.0),
            "max_x_dist":                jp.array(0.0),
        }

        info = {"rng": rng, "max_x": data.qpos[0]}

        obs = self._get_obs(data, info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: JaxArray) -> State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        vx          = data.qvel[0]
        frwd_reward = vx
        ctrl_cost   = _CTRL_COST_W * jp.sum(jp.square(action))
        leg_angles  = data.qpos[jp.array(list(_LEG_IDX))]
        leg_ext_norm = 1.0 - jp.abs(leg_angles - _LEG_CENTER) / _LEG_HALF_RANGE
        leg_ext_cost = _LEG_EXT_COST_W * jp.sum(leg_ext_norm)
        reward       = frwd_reward - ctrl_cost - leg_ext_cost

        # ── Track max forward distance (delta trick: sum of deltas = final max) ─
        new_max_x = jp.maximum(state.info["max_x"], data.qpos[0])
        delta_max_x = new_max_x - state.info["max_x"]
        info = {**state.info, "max_x": new_max_x}

        obs  = self._get_obs(data, info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        metrics = {
            "reward":                    reward,
            "reward/forward_vel":        frwd_reward,
            "reward/ctrl_cost":          ctrl_cost,
            "reward/leg_extension_cost": leg_ext_cost,
            "max_x_dist":                delta_max_x,
        }

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        # TODO: center of mass dynamics
        qpos = data.qpos
        # print(f"==>> qpos: {qpos}")
        qvel = data.qvel
        # print(f"==>> qvel: {qvel}")
        return jp.concatenate([qpos, qvel])

    def _compute_reward_and_metrics(self) -> JaxArray:
        # TODO: this function will compute the reward and set both metrics and info
        return jp.array(0.0)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> MjxModel:
        return self._mjx_model


dm_control_suite.register_environment(
    env_name="TWMRLegFlatRawTorques",
    env_class=TWMRLegFlatRawTorques,
    cfg_class=default_config,
)