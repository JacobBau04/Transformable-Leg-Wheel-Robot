"""Pure-numpy on-robot inference + controller for the RMA policy.

NO jax / flax / torch / mujoco. Copy THIS FILE plus the exported
`deploy_params.npz` (see sandbox/export_params.py) to the robot and run the
50 Hz loop. Depends only on numpy, so it imports cleanly on a Jetson/Pi without
the rest of the `twmr` package.

The control loop, per tick (mirrors phase2_run's rollout EXACTLY — ordering is
load-bearing because phi must see strictly-past frames):

    rma = RMADeploy("deploy_params.npz", torque_to_voltage_gain=R_over_Kt)
    rma.reset()
    while running:
        action, torque = rma.step(accel, gyro, quat_wxyz, leg_pos, wheel_angle)
        send_voltages(rma.torque_to_voltage(torque))

Inside step():
    gravity = R(quat).T @ [0,0,-1]                 # match twmr.py:_get_obs
    wheel_vel/leg_vel = finite-diff of encoders / dt   (wheel angle unwrapped)
    student = [accel, gyro, gravity, leg_pos, wheel_vel, leg_vel, prev_action]  # 29
    sn      = (student - norm_mean) / norm_std
    z       = phi(history)        # history holds the previous HISTORY_LEN frames
    a       = pi(sn, z)           # tanh(mlp([sn,z])[:action_size])
    history.push([sn, a])         # push AFTER computing z
    prev_action = a

Frame conventions the CALLER must satisfy (all already in the robot body frame —
apply any fixed IMU->body mounting rotation upstream):
    accel : m/s^2, proper acceleration (includes gravity; reads +g "up" at rest)
    gyro  : rad/s
    quat  : body->world unit quaternion, scalar-first [w, x, y, z] (MuJoCo order)
    leg_pos / wheel_angle : rad
"""
from __future__ import annotations

import numpy as np

# ── low-level controller constants — MUST mirror twmr.py ────────────────────
# (test_deploy_parity.py asserts these equal the twmr.py values so they can't drift.)
WHEEL_MAX_SPEED = 8.0
WHEEL_KP = 0.2
WHEEL_KD = 0.0
WHEEL_TORQUE_LIMIT = 0.8
LEG_CENTER = 1.19
LEG_HALF_RANGE = 2.237
LEG_MIN = -1.047
LEG_MAX = 3.427
LEG_POS_KP = 5.0
LEG_VEL_KP = 0.3
LEG_TORQUE_LIMIT = 0.6
LEG_MAX_VEL_CMD = 10.0
CTRL_DT = 0.02   # 50 Hz


def quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body->world rotation matrix from a scalar-first unit quaternion [w,x,y,z].

    Matches MuJoCo's site_xmat convention so `quat_to_rot(q).T @ [0,0,-1]`
    equals the projected-gravity used in twmr.py:_get_obs.
    """
    w, x, y, z = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def gravity_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame: world [0,0,-1] rotated into the body."""
    R = quat_to_rot(quat_wxyz)
    return R.T @ np.array([0.0, 0.0, -1.0])


def _wrap_to_pi(d: np.ndarray) -> np.ndarray:
    """Wrap angle deltas to (-pi, pi]. Handles wheel-encoder 2*pi wraparound;
    a no-op for the small per-tick deltas seen in sim (so parity is preserved)."""
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def _dense(x, W, b):
    return x @ W + b


def _windows(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    """im2col over the time axis: (T, C) -> (T_out, kernel*C). VALID padding.

    Numerically identical to adaptation.py's `_windows` for a single sequence.
    """
    T, C = x.shape
    t_out = (T - kernel) // stride + 1
    return np.stack(
        [x[i * stride:i * stride + kernel].reshape(-1) for i in range(t_out)],
        axis=0,
    )


class RMADeploy:
    def __init__(self, npz_path: str, ctrl_dt: float = CTRL_DT,
                 torque_to_voltage_gain: float | None = None):
        z = np.load(npz_path)
        self.norm_mean = z["norm_mean"].astype(np.float64)
        self.norm_std = z["norm_std"].astype(np.float64)

        self.student_obs_size = int(z["meta_student_obs_size"])
        self.latent_dim = int(z["meta_latent_dim"])
        self.action_size = int(z["meta_action_size"])
        self.history_len = int(z["meta_history_len"])
        self.per_step_feat = int(z["meta_per_step_feat"])
        self._phi_n_per_step = int(z["meta_phi_n_per_step"])
        self._phi_conv_specs = z["meta_phi_conv_specs"].astype(int)   # (n_conv, 2)
        n_pi = int(z["meta_n_pi_layers"])

        self._pi = [(z[f"pi_W{i}"].astype(np.float64), z[f"pi_b{i}"].astype(np.float64))
                    for i in range(n_pi)]
        # phi layer count = per-step (n) + conv (len specs) + 1 final
        n_phi = self._phi_n_per_step + len(self._phi_conv_specs) + 1
        self._phi = [(z[f"phi_W{i}"].astype(np.float64), z[f"phi_b{i}"].astype(np.float64))
                     for i in range(n_phi)]

        self.ctrl_dt = ctrl_dt
        self.torque_to_voltage_gain = torque_to_voltage_gain
        self.reset()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Zero the history (matches init_history) and clear finite-diff state.

        For the first ~HISTORY_LEN ticks phi sees mostly zeros, so hold the robot
        stationary (action ~ 0) for ~0.5 s before engaging.
        """
        self.history = np.zeros((self.history_len, self.per_step_feat))
        self.prev_action = np.zeros(self.action_size)
        self._prev_wheel_angle = None
        self._prev_leg_pos = None

    # ── networks ──────────────────────────────────────────────────────────────
    def phi_forward(self, history: np.ndarray) -> np.ndarray:
        x = history                                   # (T, F)
        for i in range(self._phi_n_per_step):         # per-step MLP, tanh
            W, b = self._phi[i]
            x = np.tanh(_dense(x, W, b))
        for j, (k, s) in enumerate(self._phi_conv_specs):   # windowed-Dense "conv", relu
            W, b = self._phi[self._phi_n_per_step + j]
            x = np.maximum(_dense(_windows(x, int(k), int(s)), W, b), 0.0)
        W, b = self._phi[-1]                          # final Dense -> z
        return _dense(x.reshape(-1), W, b)

    def pi_forward(self, student_norm: np.ndarray, z: np.ndarray) -> np.ndarray:
        h = np.concatenate([student_norm, z])         # 37 = student(29) + z(8)
        for W, b in self._pi[:-1]:                    # hidden, tanh
            h = np.tanh(_dense(h, W, b))
        W, b = self._pi[-1]
        logits = _dense(h, W, b)                      # 2*action_size (mean | log_std)
        return np.tanh(logits[:self.action_size])     # deterministic action in [-1,1]

    # ── controller (mirrors twmr.py:step) ──────────────────────────────────────
    def controller(self, action, wheel_vel, leg_pos, leg_vel) -> np.ndarray:
        """Map a policy action to per-actuator torque [4 wheel, 4 leg] (N*m).

        motor_strength is 1.0 on the real robot (it was a training-time DR knob).
        """
        # wheels: velocity P-control
        desired_wheel_vel = action[:4] * WHEEL_MAX_SPEED
        wheel_torque = WHEEL_KP * (desired_wheel_vel - wheel_vel) - WHEEL_KD * wheel_vel
        wheel_torque = np.clip(wheel_torque, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

        # legs: cascaded position -> velocity -> torque
        desired_leg_pos = LEG_CENTER + action[4:] * LEG_HALF_RANGE
        desired_leg_vel = np.clip(LEG_POS_KP * (desired_leg_pos - leg_pos),
                                  -LEG_MAX_VEL_CMD, LEG_MAX_VEL_CMD)
        leg_torque = LEG_VEL_KP * (desired_leg_vel - leg_vel)
        leg_torque = np.clip(leg_torque, -LEG_TORQUE_LIMIT, LEG_TORQUE_LIMIT)

        return np.concatenate([wheel_torque, leg_torque])

    def torque_to_voltage(self, torque: np.ndarray) -> np.ndarray:
        """V = (R/Kt) * torque. Requires torque_to_voltage_gain to be calibrated."""
        if self.torque_to_voltage_gain is None:
            raise ValueError("torque_to_voltage_gain (R/Kt) not set; calibrate it first.")
        return self.torque_to_voltage_gain * torque

    # ── observation assembly ────────────────────────────────────────────────
    def assemble_student(self, accel, gyro, gravity, leg_pos, wheel_vel, leg_vel) -> np.ndarray:
        """29-dim student obs in the EXACT order of twmr.py:_get_obs."""
        return np.concatenate([accel, gyro, gravity, leg_pos, wheel_vel, leg_vel,
                               self.prev_action])

    # ── one control tick ──────────────────────────────────────────────────────
    def step(self, accel, gyro, quat_wxyz, leg_pos, wheel_angle, dt: float | None = None):
        """Run one 50 Hz tick. Returns (action[-1,1]^8, torque[N*m]^8).

        See module docstring for input frame/unit conventions.
        """
        dt = self.ctrl_dt if dt is None else dt
        accel = np.asarray(accel, float)
        gyro = np.asarray(gyro, float)
        leg_pos = np.asarray(leg_pos, float)
        wheel_angle = np.asarray(wheel_angle, float)

        gravity = gravity_from_quat(np.asarray(quat_wxyz, float))

        # finite-difference velocities (match twmr.py: (qpos - prev)/dt). Unwrap
        # the wheel angle for 2*pi encoder wraparound. First tick -> zero velocity.
        if self._prev_wheel_angle is None:
            wheel_vel = np.zeros(4)
            leg_vel = np.zeros(4)
        else:
            wheel_vel = _wrap_to_pi(wheel_angle - self._prev_wheel_angle) / dt
            leg_vel = (leg_pos - self._prev_leg_pos) / dt

        student = self.assemble_student(accel, gyro, gravity, leg_pos, wheel_vel, leg_vel)
        sn = (student - self.norm_mean) / self.norm_std

        z = self.phi_forward(self.history)          # strictly-past frames
        action = self.pi_forward(sn, z)

        # push current frame AFTER computing z (mirrors update_history)
        frame = np.concatenate([sn, action])
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = frame

        torque = self.controller(action, wheel_vel, leg_pos, leg_vel)

        self.prev_action = action
        self._prev_wheel_angle = wheel_angle
        self._prev_leg_pos = leg_pos
        return action, torque
