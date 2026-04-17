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
_CTRL_COST_W = 0.0005
_LEG_EXT_COST_W = 0.01

# ── PD controller constants ──────────────────────────────────────────────
# Wheel velocity P(D) controller
_WHEEL_MAX_SPEED = 8.0           # rad/s  (action ±1 maps to ±8, physical limit)
_WHEEL_KP = 0.2                 # P gain → max torque 0.08*8 = 0.64, within 0.8 clip
_WHEEL_KD = 0.0                  # D gain (jerk damping; 0 unless oscillation observed)
_WHEEL_TORQUE_LIMIT = 0.8        # N·m  (matches XML ctrlrange)
_WHEEL_QVEL_IDX = jp.array([6, 10, 14, 18])

# Leg cascaded position→velocity controller
_LEG_POS_KP = 5.0                # outer loop: pos error → desired velocity
_LEG_POS_KD = 0.0                # outer loop D term: -Kd * actual_vel (0.1?)
_LEG_VEL_KP = 0.3                # inner loop: vel error → torque
_LEG_TORQUE_LIMIT = 0.6          # N·m
_LEG_MAX_VEL_CMD = 10.0          # rad/s clamp on outer loop output
_LEG_QPOS_IDX = jp.array([8, 12, 16, 20])
_LEG_QVEL_IDX = jp.array([7, 11, 15, 19])

# Wheel encoder qpos indices (hinge joints after 7-dim freejoint)
_WHEEL_QPOS_IDX = jp.array([7, 11, 15, 19])


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
        sim_dt=0.002,  # MuJoCo default physics timestep
        episode_length=1000,
        action_repeat=1,  # TODO: should this be a ratio of ctrl_dt / sim_dt?
        vision=False,
        # vision_config=default_vision_config(),
        impl="warp",  # TODO: cartpole uses jax
        nconmax=100,  # allow collisions
        njmax=500,  # allow complex joints
    )


class TWMRLegFlat(MjxEnv):
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
        self._mj_model.opt.timestep = self.sim_dt  # set BEFORE put_model
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore

        # Cache IMU site ID for projected gravity computation
        self._imu_site_id = self._mj_model.site("root_site").id

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

        info = {
            "rng": rng,
            "max_x": data.qpos[0],
            "prev_wheel_pos": data.qpos[_WHEEL_QPOS_IDX],
            "prev_leg_pos": data.qpos[_LEG_QPOS_IDX],
            "prev_action": jp.zeros(self.action_size),
        }

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
        # ── Decode policy action [8] into ctrl torques [8] via PD ────────
        # action[0:4] → desired wheel speeds,  action[4:8] → desired leg positions

        # --- Wheel velocity P(D) controller ---
        desired_wheel_vel = action[:4] * _WHEEL_MAX_SPEED          # [-20, 20] rad/s
        actual_wheel_vel = state.data.qvel[_WHEEL_QVEL_IDX]
        wheel_vel_err = desired_wheel_vel - actual_wheel_vel
        wheel_torque = _WHEEL_KP * wheel_vel_err - _WHEEL_KD * actual_wheel_vel
        # wheel_torque = jp.full(4, 5.0)
        wheel_torque = jp.clip(wheel_torque, -_WHEEL_TORQUE_LIMIT, _WHEEL_TORQUE_LIMIT)

        # --- Leg cascaded position → velocity → torque controller ---
        desired_leg_pos = _LEG_CENTER + action[4:] * _LEG_HALF_RANGE  # map [-1,1] to [min,max]
        actual_leg_pos = state.data.qpos[_LEG_QPOS_IDX]
        actual_leg_vel = state.data.qvel[_LEG_QVEL_IDX]
        # Outer loop: position error → desired velocity
        desired_leg_vel = _LEG_POS_KP * (desired_leg_pos - actual_leg_pos) #- _LEG_POS_KD * actual_leg_vel
        desired_leg_vel = jp.clip(desired_leg_vel, -_LEG_MAX_VEL_CMD, _LEG_MAX_VEL_CMD)
        # Inner loop: velocity error → torque
        leg_vel_err = desired_leg_vel - actual_leg_vel
        leg_torque = _LEG_VEL_KP * leg_vel_err
        # leg_torque = jp.full(4, -5.0)
        leg_torque = jp.clip(leg_torque, -_LEG_TORQUE_LIMIT, _LEG_TORQUE_LIMIT)

        # Assemble ctrl: actuator order is [4 wheels, 4 legs]
        ctrl = jp.concatenate([wheel_torque, leg_torque])

        # Step physics with computed torques
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        # ── Reward (penalize actual torques, not raw action) ─────────────
        vx          = data.qvel[0]
        frwd_reward = vx
        ctrl_cost   = _CTRL_COST_W * jp.sum(jp.square(ctrl))
        leg_angles  = data.qpos[_LEG_QPOS_IDX]
        leg_ext_norm = 1.0 - jp.abs(leg_angles - _LEG_CENTER) / _LEG_HALF_RANGE
        leg_ext_cost = _LEG_EXT_COST_W * jp.sum(leg_ext_norm)
        reward       = frwd_reward - ctrl_cost - leg_ext_cost

        # ── Track max forward distance (delta trick: sum of deltas = final max) ─
        new_max_x = jp.maximum(state.info["max_x"], data.qpos[0])
        delta_max_x = new_max_x - state.info["max_x"]
        info = {
            **state.info,
            "max_x": new_max_x,
            # Store pre-step encoder positions for discrete velocity computation
            "prev_wheel_pos": state.data.qpos[_WHEEL_QPOS_IDX],
            "prev_leg_pos": state.data.qpos[_LEG_QPOS_IDX],
            "prev_action": action,
        }

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
        # IMU sensors: proper acceleration (includes gravity) and angular velocity
        accel = mjx_env.get_sensor_data(self.mj_model, data, "root_acc")   # 3
        gyro = mjx_env.get_sensor_data(self.mj_model, data, "root_gyro")   # 3

        # Projected gravity: world [0,0,-1] rotated into body frame
        # site_xmat is stored as a flattened 9-element row-major 3x3 matrix
        imu_rot = data.site_xmat[self._imu_site_id].reshape(3, 3) # Projected gravity. IMPORTANT: THIS IS THE TRUE ROBOT ORIENTATION (TDLR: ADD NOISE OR DRIFT BEFORE TRYING SIM TO REAL TRANSFER)
        gravity = imu_rot.T @ jp.array([0.0, 0.0, -1.0])                   # 3

        # Leg encoder positions
        leg_pos = data.qpos[_LEG_QPOS_IDX]                                 # 4

        # Encoder velocities via discrete differentiation at ctrl_dt rate
        wheel_vel = (data.qpos[_WHEEL_QPOS_IDX] - info["prev_wheel_pos"]) / self.dt  # 4
        leg_vel = (data.qpos[_LEG_QPOS_IDX] - info["prev_leg_pos"]) / self.dt        # 4

        # Previous action
        prev_action = info["prev_action"]                                   # 8

        return jp.concatenate([
            accel, gyro, gravity, leg_pos, wheel_vel, leg_vel, prev_action,
        ])

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
    env_name="TWMRLegFlat",
    env_class=TWMRLegFlat,
    cfg_class=default_config,
)
