from pathlib import Path
from typing import Any
import jax

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

# RMA / privileged-observation sizes
_NUM_WHEELS = 4
_NUM_LEGS = 4
_NUM_ACTUATORS = 8   # 4 wheel + 4 leg motors

# simplified privileged obs layout:
# friction (1)
# motor strengths (8)
# slope angle (1)
_PRIV_GLOBAL_SIZE = 1 + 8 + 1
_PRIV_OBS_SIZE = _PRIV_GLOBAL_SIZE # = 10; no privileged contacts for now, but could add in the future if needed

# domain-randomization ranges
_FRICTION_RANGE = (0.7, 1.3)
_WHEEL_MOTOR_RANGE = (0.85, 1.15)
_LEG_MOTOR_RANGE = (0.85, 1.15)
_SLOPE_RANGE = (-0.12, 0.12)  # slope angle (radians) *prob change later for rough terrain

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
        naconmax=100,  # GLOBAL contact budget across all Warp worlds (scale with NUM_ENVS at training time)
        njmax=500,     # per-env constraint budget (stacked by vmap; ~20-30 constraints/env in practice)
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

    def _sample_domain_params(self, rng: JaxArray) -> tuple[JaxArray, dict[str, JaxArray]]:
        rng, r_mu, r_wheel, r_leg, r_slope = jax.random.split(rng, 5)

        friction = jax.random.uniform(
            r_mu, (), minval=_FRICTION_RANGE[0], maxval=_FRICTION_RANGE[1]
        )

        wheel_motor_strength = jax.random.uniform(
            r_wheel, (_NUM_WHEELS,), minval=_WHEEL_MOTOR_RANGE[0], maxval=_WHEEL_MOTOR_RANGE[1]
        )

        leg_motor_strength = jax.random.uniform(
            r_leg, (_NUM_LEGS,), minval=_LEG_MOTOR_RANGE[0], maxval=_LEG_MOTOR_RANGE[1]
        )

        slope_angle = jax.random.uniform(
            r_slope, (), minval=_SLOPE_RANGE[0], maxval=_SLOPE_RANGE[1]
        )

        params = {
            "friction": friction,
            "wheel_motor_strength": wheel_motor_strength,
            "leg_motor_strength": leg_motor_strength,
            "motor_strengths": jp.concatenate([wheel_motor_strength, leg_motor_strength]),
            "slope_angle": slope_angle,
        }
        return rng, params
    
    def _get_privileged_globals(self, info: dict[str, Any]) -> JaxArray:
        return jp.concatenate([
            jp.array([info["friction"]]),           # (1)
            info["motor_strengths"],                # (8)
            jp.array([info["slope_angle"]]),        # (1)
        ])
    
    def _get_privileged_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        return self._get_privileged_globals(info)
    
    def _get_teacher_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        student_obs = self._get_obs(data, info)
        privileged_obs = self._get_privileged_obs(data, info)
        return jp.concatenate([student_obs, privileged_obs])

    def reset(self, rng: JaxArray) -> State:
        rng, r_x, r_y, r_z, r_yaw, r_leg, r_vel = jax.random.split(rng, 7)

        qpos = self.mjx_model.qpos0
        qvel = jp.zeros(self.mjx_model.nv)

        # root position
        qpos = qpos.at[0].add(jax.random.uniform(r_x, (), minval=-0.03, maxval=0.03))
        qpos = qpos.at[1].add(jax.random.uniform(r_y, (), minval=-0.03, maxval=0.03))
        qpos = qpos.at[2].add(jax.random.uniform(r_z, (), minval=-0.005, maxval=0.005))

        # root yaw quaternion, since XML default is identity
        yaw = jax.random.uniform(r_yaw, (), minval=-0.15, maxval=0.15)
        cy = jp.cos(yaw / 2.0)
        sy = jp.sin(yaw / 2.0)
        qpos = qpos.at[3:7].set(jp.array([cy, 0.0, 0.0, sy]))

        # sample one leg angle per wheel around a nominal stance
        nominal_leg = jp.array([1.0, 1.0, 1.0, 1.0])  # example
        leg_noise = jax.random.uniform(r_leg, (4,), minval=-0.1, maxval=0.1)
        leg_vals = jp.clip(nominal_leg + leg_noise, _LEG_MIN, _LEG_MAX)

        # set actuated extension joints
        qpos = qpos.at[_LEG_QPOS_IDX].set(leg_vals)

        qvel = qvel + 0.05 * jax.random.normal(r_vel, qvel.shape)

        rng, params = self._sample_domain_params(rng)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "reward":                    jp.array(0.0),
            "reward/forward_vel":        jp.array(0.0),
            "reward/ctrl_cost":          jp.array(0.0),
            "reward/leg_extension_cost": jp.array(0.0),
            "max_x_dist":                jp.array(0.0),
            "done/height":               jp.array(0.0),
            "done/tilt":                 jp.array(0.0),
            "done/angvel":               jp.array(0.0),
            "done/nan":                  jp.array(0.0),
        }
        

        info = {
            "rng": rng,
            "max_x": data.qpos[0],
            "prev_wheel_pos": data.qpos[_WHEEL_QPOS_IDX],
            "prev_leg_pos": data.qpos[_LEG_QPOS_IDX],
            "prev_action": jp.zeros(self.action_size),

            # privileged episode-global params
            "friction": params["friction"],
            "wheel_motor_strength": params["wheel_motor_strength"],
            "leg_motor_strength": params["leg_motor_strength"],
            "motor_strengths": params["motor_strengths"],
            "slope_angle": params["slope_angle"],
        }

        student_obs = self._get_obs(data, info)
        priv_obs = self._get_privileged_obs(data, info)
        teacher_obs = jp.concatenate([student_obs, priv_obs])

        info["student_obs"] = student_obs
        info["privileged_obs"] = priv_obs

        obs = teacher_obs

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
        
        # Apply sampled motor-strength randomization
        wheel_torque = state.info["wheel_motor_strength"] * wheel_torque
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

        leg_torque = state.info["leg_motor_strength"] * leg_torque # apply domain-randomized motor strength

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

        student_obs = self._get_obs(data, info)
        priv_obs = self._get_privileged_obs(data, info)
        teacher_obs = jp.concatenate([student_obs, priv_obs])

        info["student_obs"] = student_obs
        info["privileged_obs"] = priv_obs

        obs = teacher_obs
        base_height = data.qpos[2]

        imu_rot = data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity = imu_rot.T @ jp.array([0.0, 0.0, -1.0])

        gyro = mjx_env.get_sensor_data(self.mj_model, data, "root_gyro")

        height_fail = base_height < 0 #0.045
        tilt_fail = gravity[2] > 10 #-0.6
        angvel_fail = jp.linalg.norm(gyro) > 20.0
        nan_fail = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        done = (height_fail | tilt_fail | angvel_fail | nan_fail).astype(float)

        metrics = {
            "reward": reward,
            "reward/forward_vel": frwd_reward,
            "reward/ctrl_cost": ctrl_cost,
            "reward/leg_extension_cost": leg_ext_cost,
            "max_x_dist": delta_max_x,
            "done/height": height_fail.astype(float),
            "done/tilt": tilt_fail.astype(float),
            "done/angvel": angvel_fail.astype(float),
            "done/nan": nan_fail.astype(float),
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
    
    @property
    def privileged_obs_size(self) -> int:
        return _PRIV_OBS_SIZE


dm_control_suite.register_environment(
    env_name="TWMRLegFlat",
    env_class=TWMRLegFlat,
    cfg_class=default_config,
)

