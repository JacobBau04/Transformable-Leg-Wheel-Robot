
"""

TO RUN (from root): python3 packages/twmr/src/twmr/train_transform_flat_PPO2_0.py

PPO training for transformable leg-wheel robot on a flat plane,
with trans_wheel_robo2_0.xml:
  - 8 torque actuators total:
      *4 wheel torque motors
      *4 leg0 extension torque motors
  - coupling leg1 and leg2 to leg0


Reward:
  forward_reward = vx
  ctrl_cost = 0.0005 * sum(action^2)
  leg_extension_cost = more complex, see code
  finishes using a fixed horizon


23-31 seconds per minibatch / PPO update

"""

import os
import subprocess
import time
from typing import Dict, Any, Optional, List

import numpy as np

# --- Configure JAX & MuJoCo before importing them ---

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["MUJOCO_GL"] = "egl"

NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write(
            """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""
        )

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import mediapy as media
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco


print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

print("Checking that MuJoCo is installed and working...")
try:
    mujoco.MjModel.from_xml_string("<mujoco/>")
except Exception as e:
    raise RuntimeError("MuJoCo test model failed to load. Check installation.") from e
print("MuJoCo installation check passed.")

print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

if subprocess.run("nvidia-smi", shell=True).returncode != 0:
    print("Warning: nvidia-smi failed; GPU may not be visible on this node.")


# ---------------------------------------------------------------------------
# 1) Environment around trans_wheel_robo2_0.xml
# ---------------------------------------------------------------------------

DESIRED_ACTUATORS: List[str] = [
    # Wheel torques
    "front_left_wheel_joint_ctrl",
    "front_right_wheel_joint_ctrl",
    "rear_left_wheel_joint_ctrl",
    "rear_right_wheel_joint_ctrl",
    # Leg0 extension torques (leg1 & leg2 are coupled in the XML)
    "front_left_wheel_0_extension_joint_ctrl",
    "front_right_wheel_0_extension_joint_ctrl",
    "rear_left_wheel_0_extension_joint_ctrl",
    "rear_right_wheel_0_extension_joint_ctrl",
]

LEG0_JOINTS: List[str] = [
    "front_left_wheel_0_extension_joint",
    "front_right_wheel_0_extension_joint",
    "rear_left_wheel_0_extension_joint",
    "rear_right_wheel_0_extension_joint",
]

from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    """Walk upwards until we find a README.md (repo root marker)."""
    for p in [start, *start.parents]:
        if (p / "README.md").is_file():
            return p
    # Fallback: two levels up from this file (keeps old behavior if README not found)
    return start.parents[2] if len(start.parents) >= 2 else start


# Results directory at repo root (same level as README.md)
_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
RESULTS_DIR = _REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _results_path(path_like) -> str:
    """Return an absolute path inside RESULTS_DIR unless an absolute path is given."""
    p = Path(path_like)
    return str(p if p.is_absolute() else (RESULTS_DIR / p))


def _resolve_xml_path(xml_path: str) -> str:
    p = Path(xml_path)

    # If the user provided an absolute path or a valid relative path from cwd, accept it.
    if p.is_file():
        return str(p.resolve())

    # Otherwise, look relative to this python file's directory
    here = Path(__file__).resolve().parent
    candidates = [
        here / p.name,                  # same folder as this script
        here / p,                       # relative to this script folder
        here.parent / p.name,           # one level up
        here.parent / p,                # one level up + relative
    ]

    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    raise FileNotFoundError(
        f"Could not find XML '{xml_path}'. Tried:\n" + "\n".join(str(c) for c in candidates)
    )

class TransformableWheelFlatEnv:
    """
    - Obs: [qpos, qvel]
    - Action: 8D torque vector ordered as DESIRED_ACTUATORS
    - Reward: forward velocity-like term - torque penalty - leg-extension penalty
    - Done: fixed horizon only
    """

    def __init__(
        self,
        xml_path: str = "trans_wheel_robo2_0.xml",
        desired_ctrl_dt: float = 0.02,
        frame_skip: Optional[int] = None,
        max_steps: int = 1000,
        
    ):
        xml_path = _resolve_xml_path(xml_path)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.sim_dt = float(self.model.opt.timestep)

        # Choose frame_skip so action interval ~ desired_ctrl_dt
        if frame_skip is None:
            frame_skip = max(1, int(round(desired_ctrl_dt / self.sim_dt)))
        self.frame_skip = int(frame_skip)

        # Actual control dt used by the environment (used in reward scaling + video fps)
        self.ctrl_dt = self.frame_skip * self.sim_dt
        self.max_steps = int(max_steps)

        # Find freejoint qpos address (so x,z indices are correct)
        self.free_qpos_adr = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_adr = int(self.model.jnt_qposadr[j])
                break
        if self.free_qpos_adr is None:
            raise RuntimeError("No free joint found. This env assumes the root has a freejoint.")

        self.x_index = self.free_qpos_adr + 0
        self.y_index = self.free_qpos_adr + 1



        # Map actions to actuator ids
        self.nu = int(self.model.nu)
        self.actuator_ids = []



        for name in DESIRED_ACTUATORS:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{name}' not found in model. Check XML.")
            self.actuator_ids.append(int(aid))

        print("\n Actuator -> transmission target")
        for i, name in enumerate(DESIRED_ACTUATORS):
            aid = self.actuator_ids[i]
            trnid0 = int(self.model.actuator_trnid[aid, 0])
            trnid1 = int(self.model.actuator_trnid[aid, 1])
            trntype = int(self.model.actuator_trntype[aid])

            # Try to interpret target as a joint (common for motor/position actuators)
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, trnid0)
            print(f"  a[{i}] '{name}' -> aid={aid}, trntype={trntype}, trnid=({trnid0},{trnid1}), joint0={jname}")

        self.action_dim = len(self.actuator_ids)
        if self.action_dim != 8:
            raise RuntimeError(f"Expected 8 desired actuators, got {self.action_dim}.")

        # Per-actuator ctrlrange scale (used by the policy squashing)
        ctrlranges = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        scales = []
        for aid in self.actuator_ids:
            lo, hi = float(ctrlranges[aid, 0]), float(ctrlranges[aid, 1])
            s = max(abs(lo), abs(hi))
            scales.append(s if s > 0.0 else 1.0)
        self.action_scale = np.asarray(scales, dtype=np.float32)

        self.full_ext_angle = 2.237 # Angle in radians where the leg is fully extended

        # Leg qpos indices + max angle for normalization
        self.leg_qpos_idx = []
        max_abs = []
        for jname in LEG0_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Leg joint '{jname}' not found in model. Check XML.")
            adr = int(self.model.jnt_qposadr[jid])
            self.leg_qpos_idx.append(adr)

            # Use joint range if available; fallback to 1.0
            if int(self.model.jnt_limited[jid]) == 1:
                lo, hi = float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])
                max_abs.append(max(abs(lo), abs(hi)))
        self.leg_max_angle = float(max(max_abs) if len(max_abs) else 1.0)

        print(f"[Env] Loaded XML: {xml_path}")
        print(f"[Env] sim_dt={self.sim_dt:.6f}, frame_skip={self.frame_skip}, ctrl_dt={self.ctrl_dt:.6f}")
        print("[Env] Action order (policy outputs) -> actuator name:")
        for i, name in enumerate(DESIRED_ACTUATORS):
            print(f"   a[{i}] -> {name}  (scale≈{self.action_scale[i]:.3f} N·m)")
        print(f"[Env] leg_qpos_idx={self.leg_qpos_idx}, leg_max_angle≈{self.leg_max_angle:.3f} rad")

        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        qpos = np.array(self.data.qpos, dtype=np.float32)
        qvel = np.array(self.data.qvel, dtype=np.float32)
        return np.concatenate([qpos, qvel], axis=0)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        for _ in range(50):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")

        # Clip to per-actuator ctrlrange
        action = np.clip(action, -self.action_scale, self.action_scale)

        x_before = float(self.data.qpos[self.x_index])
        y_before = float(self.data.qpos[self.y_index])

        # Fill full ctrl vector in model actuator order
        ctrl_full = np.zeros((self.nu,), dtype=np.float32)
        for i, aid in enumerate(self.actuator_ids):
            ctrl_full[aid] = action[i]

        self.data.ctrl[:] = ctrl_full
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = float(self.data.qpos[self.x_index])
        y_after = float(self.data.qpos[self.y_index])


        # vx = (x_after - x_before) / self.ctrl_dt
        # vy = (y_after - y_before) / self.ctrl_dt
        vx = float(self.data.qvel[self.x_index]) 
        vy = float(self.data.qvel[self.y_index])

        forward_reward = vx
        sideways_cost = 0.03 * (vy ** 2) / (vx ** 2 + vy ** 2 + 1e-8)

        # Penalize large torques
        ctrl_cost = 0.0005 * float(np.sum(np.square(action)))

        # -1.047 3.427
        # a = angles + 1.047
        # norm = a / 4.474 # from -1 to 1
        
        
        # leg_angles = np.array([self.data.qpos[i] for i in self.leg_qpos_idx], dtype=np.float32)
        # leg_angles = np.array([self.data.qpos[i] for i in [8, 12, 16, 20]], dtype=np.float32)


        def calc_ext_cost(data,full_ext_angle):
            leg_angles = np.array([data.qpos[i] for i in [8, 12, 16, 20]], dtype=np.float32)
            leg_omegas = np.array([data.qvel[i] for i in [8, 12, 16, 20]], dtype=np.float32)
            leg_angles_normalized = np.clip(((leg_angles - 1.19) / full_ext_angle), -1, 1)  # [-1, 1]
            dist_from_closed = 1 - np.abs(leg_angles_normalized)
            angular_vel_dir_cost = (-np.sign(leg_angles_normalized) * leg_omegas * dist_from_closed)
            leg_extension_cost = 1e-3 * np.sqrt(np.sum(np.square(angular_vel_dir_cost + 2 * dist_from_closed)))
            return leg_extension_cost, leg_angles
        
        leg_extension_cost, leg_angles = calc_ext_cost(self.data,self.full_ext_angle)
        retract_err = []


        # leg_angles_normalized = (leg_angles - 1.19) / self.full_ext_angle # from -1 to 1
        # leg_extension_cost = -0.01 * float(np.sum(leg_angles_normalized ** 2))
        # leg_extension_cost = 0.5 * float(np.sum(retract_err))   # try 0.1 to 2.0

        # Encourage all legs to behave similarly
        # leg_sym_cost = 0.2 * float(np.sum((leg_angles - leg_angles.mean())**2))  # try 0.05 to 0.5

        reward = forward_reward - ctrl_cost - leg_extension_cost - sideways_cost
        # print("forward_reward: ",forward_reward)
        # print("leg_extension_cost: ",leg_extension_cost)

        # reward =  - leg_extension_cost #- leg_sym_cost


        self.step_count += 1
        done = self.step_count >= self.max_steps
        # print("forward_reward: ",forward_reward)
        # print("ctrl_cost: ",ctrl_cost)
        # print("leg_extension_cost: ",leg_extension_cost)

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "x_before": x_before,
            "x_after": x_after,
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "leg_extension_cost": leg_extension_cost,
            "leg_angles": leg_angles,
            "leg_extension_frac": retract_err,
        }
        return obs, reward, done, info

    def render_episode(
        self,
        policy_fn,
        params,
        video_path: str,
        deterministic: bool = True,
        fps: Optional[float] = None,
    ):
        obs = self.reset()
        frames = []

        # track x displacement for this trial
        x_start = float(self.data.qpos[self.x_index])

        renderer = mujoco.Renderer(self.model, height=480, width=640)

        rng = jax.random.PRNGKey(0)
        if fps is None:
            fps = 1.0 / self.ctrl_dt

        while True:
            renderer.update_scene(self.data)
            frames.append(renderer.render())

            obs_jnp = jnp.asarray(obs[None, :], dtype=jnp.float32)
            rng, subkey = jax.random.split(rng)
            action, _, _ = policy_fn(params, obs_jnp, subkey, deterministic=deterministic)
            action_np = np.asarray(action[0], dtype=np.float32)

            obs, _, done, _ = self.step(action_np)
            if done:
                break

        renderer.close()
        video_path = _results_path(video_path)
        media.write_video(video_path, frames, fps=fps)

        # compute and return this trial's +x distance
        x_end = float(self.data.qpos[self.x_index])
        x_delta = x_end - x_start

        print(f"Saved video to {video_path} | trial +x distance = {x_delta:.3f} m")
        return x_delta



# ---------------------------------------------------------------------------
# 2) PPO Actor-Critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        for h in self.hidden_sizes:
            x = nn.tanh(nn.Dense(h)(x))
        mean = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        return mean, jnp.squeeze(value, axis=-1)


def gaussian_log_prob(actions, mean, log_std):
    std = jnp.exp(log_std)
    var = std ** 2
    return jnp.sum(
        -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi)),
        axis=-1,
    )


def atanh(x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-6
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))


def make_policy_value_fn(model, action_scale: jnp.ndarray):
    action_scale = action_scale.astype(jnp.float32)
    log_action_scale_sum = jnp.sum(jnp.log(action_scale))

    def apply_fn(params, obs, rng, deterministic: bool):
        mean, value = model.apply(params["model"], obs)
        log_std = params["log_std"]
        std = jnp.exp(log_std)

        if deterministic:
            pre = mean
            action = jnp.tanh(pre) * action_scale
            return action, None, value

        noise = jax.random.normal(rng, shape=mean.shape)
        pre = mean + noise * std
        tanh_pre = jnp.tanh(pre)
        action = tanh_pre * action_scale

        eps = 1e-6
        log_prob_pre = gaussian_log_prob(pre, mean, log_std)
        log_det = log_action_scale_sum + jnp.sum(jnp.log(1.0 - tanh_pre ** 2 + eps), axis=-1)
        log_prob = log_prob_pre - log_det
        return action, log_prob, value

    return apply_fn


def save_trial_x_plot(trial_labels: List[str], trial_x_deltas: List[float], out_path: str = "transform_ppo_trial_x_distance.png"):
    out_path = _results_path(out_path)
    plt.figure(figsize=(7, 4))
    xs = np.arange(len(trial_x_deltas))
    plt.plot(xs, trial_x_deltas, marker="o")
    plt.xticks(xs, trial_labels, rotation=30, ha="right")
    plt.xlabel("Saved video trial")
    plt.ylabel("Episode +x displacement (m)")
    plt.title("+x Distance Traveled per Saved Video Trial")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Updated trial x-distance plot -> {out_path}")


# ---------------------------------------------------------------------------
# 3) PPO training loop
# ---------------------------------------------------------------------------

def train_ppo(
    env: TransformableWheelFlatEnv,
    total_timesteps: int = 200_000,
    rollout_steps: int = 1024,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    minibatch_size: int = 256,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 0,
):
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_dim

    key = jax.random.PRNGKey(seed)
    model = ActorCritic(action_dim=act_dim, hidden_sizes=(64, 64))
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    key, subkey = jax.random.split(key)
    model_params = model.init(subkey, dummy_obs)

    log_std = jnp.full((act_dim,), -0.5, dtype=jnp.float32)
    params = {"model": model_params, "log_std": log_std}

    action_scale_jnp = jnp.asarray(env.action_scale, dtype=jnp.float32)
    policy_value_fn = make_policy_value_fn(model, action_scale_jnp)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )
    opt_state = optimizer.init(params)

    num_updates = total_timesteps // rollout_steps
    assert rollout_steps % minibatch_size == 0

    best_return = -np.inf
    returns_history = []
    x_delta_history = []
    trial_labels = []
    trial_x_deltas = []


    def ppo_loss(params, obs, actions, old_log_probs, returns, advantages):
        mean, value_pred = model.apply(params["model"], obs)
        log_std = params["log_std"]

        eps = 1e-6
        a_scaled = actions / action_scale_jnp
        pre = atanh(a_scaled)

        log_prob_pre = gaussian_log_prob(pre, mean, log_std)
        log_det = jnp.sum(jnp.log(action_scale_jnp)) + jnp.sum(jnp.log(1.0 - a_scaled ** 2 + eps), axis=-1)
        log_prob = log_prob_pre - log_det

        ratio = jnp.exp(log_prob - old_log_probs)
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -adv_norm * ratio
        pg_loss2 = -adv_norm * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        v_loss = jnp.mean((returns - value_pred) ** 2)

        entropy = jnp.mean(0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + log_std)
        loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
        return loss, (pg_loss, v_loss, entropy)

    @jax.jit
    def update_step(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params,
            batch["obs"],
            batch["actions"],
            batch["log_probs"],
            batch["returns"],
            batch["advantages"],
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    video_updates = {
        "10pct": max(0, int(num_updates * 0.1) - 1),
        "20pct": max(0, int(num_updates * 0.2) - 1),
        "30pct": max(0, int(num_updates * 0.3) - 1),
        "40pct": max(0, int(num_updates * 0.4) - 1),
        "50pct": max(0, int(num_updates * 0.5) - 1),
        "75pct": max(0, int(num_updates * 0.75) - 1),
        "100pct": num_updates - 1,
    }

    ep_return = 0.0
    ep_x_start = float(env.data.qpos[env.x_index])

    for update in range(num_updates):
        obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((rollout_steps, act_dim), dtype=np.float32)
        logprob_buf = np.zeros((rollout_steps,), dtype=np.float32)
        rewards_buf = np.zeros((rollout_steps,), dtype=np.float32)
        dones_buf = np.zeros((rollout_steps,), dtype=np.float32)
        values_buf = np.zeros((rollout_steps,), dtype=np.float32)


        ep_returns = []
        ep_x_deltas = []

        for t in range(rollout_steps):
            obs_buf[t] = obs

            obs_jnp = jnp.asarray(obs[None, :], dtype=jnp.float32)
            key, subkey = jax.random.split(key)
            action_jnp, log_prob_jnp, value_jnp = policy_value_fn(params, obs_jnp, subkey, deterministic=False)

            action = np.asarray(action_jnp[0], dtype=np.float32)
            log_prob = float(np.asarray(log_prob_jnp[0]))
            value = float(np.asarray(value_jnp[0]))

            actions_buf[t] = action
            logprob_buf[t] = log_prob
            values_buf[t] = value

            next_obs, reward, done, _ = env.step(action)
            rewards_buf[t] = reward
            dones_buf[t] = float(done)

            ep_return += reward
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                # Episode finished: log how far it moved in +x this episode
                ep_x_end = float(env.data.qpos[env.x_index])
                ep_x_deltas.append(ep_x_end - ep_x_start)
                next_obs = env.reset()
                # New episode start x after reset
                ep_x_start = float(env.data.qpos[env.x_index])
            obs = next_obs

        # Bootstrap last value
        obs_jnp = jnp.asarray(obs[None, :], dtype=jnp.float32)
        _, _, last_value_jnp = policy_value_fn(params, obs_jnp, rng=None, deterministic=True)
        last_value = float(np.asarray(last_value_jnp[0]))

        advantages = np.zeros_like(rewards_buf, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_nonterminal = 1.0 - dones_buf[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones_buf[t + 1]
                next_value = values_buf[t + 1]
            delta = rewards_buf[t] + gamma * next_nonterminal * next_value - values_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_buf

        if not ep_returns:
            ep_returns.append(float(np.sum(rewards_buf)))
        mean_return = float(np.mean(ep_returns))

        # Mean +x displacement (m) for episodes completed during this PPO update
        if len(ep_x_deltas) == 0:
            mean_x_delta = 0.0
        else:
            mean_x_delta = float(np.mean(ep_x_deltas))
        x_delta_history.append(mean_x_delta)

        returns_history.append(mean_return)
        best_return = max(best_return, mean_return)

        print(f"[PPO] Update {update+1}/{num_updates} mean_return={mean_return:.3f}, best_return={best_return:.3f}")

        obs_jax = jnp.asarray(obs_buf, dtype=jnp.float32)
        actions_jax = jnp.asarray(actions_buf, dtype=jnp.float32)
        logprob_jax = jnp.asarray(logprob_buf, dtype=jnp.float32)
        returns_jax = jnp.asarray(returns, dtype=jnp.float32)
        advantages_jax = jnp.asarray(advantages, dtype=jnp.float32)

        idxs = np.arange(rollout_steps)
        for _epoch in range(num_epochs):
            np.random.shuffle(idxs)
            for start in range(0, rollout_steps, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]
                batch = {
                    "obs": obs_jax[mb_idx],
                    "actions": actions_jax[mb_idx],
                    "log_probs": logprob_jax[mb_idx],
                    "returns": returns_jax[mb_idx],
                    "advantages": advantages_jax[mb_idx],
                }
                params, opt_state, _, _ = update_step(params, opt_state, batch)

        for tag, upd in video_updates.items():
            if update == upd:
                video_name = f"transform_ppo_{tag}.mp4"
                print(f"Recording {tag} training video at update {update+1}...")

                x_delta_trial = env.render_episode(
                    lambda p, o, r, deterministic: policy_value_fn(p, o, r, deterministic),
                    params,
                    video_path=_results_path(video_name),
                    deterministic=True,
                    fps=1.0 / env.ctrl_dt,
                )

                trial_labels.append(f"{tag}\n(upd {update+1})")
                trial_x_deltas.append(float(x_delta_trial))

                # update plot distance in +x traveled
                save_trial_x_plot(trial_labels, trial_x_deltas)


    print("Recording final policy video (transform_ppo_final_best.mp4)...")
    x_delta_trial = env.render_episode(
        lambda p, o, r, deterministic: policy_value_fn(p, o, r, deterministic),
        params,
        video_path=_results_path("transform_ppo_final_best.mp4"),
        deterministic=True,
        fps=1.0 / env.ctrl_dt,
    )
    trial_labels.append("final_best")
    trial_x_deltas.append(float(x_delta_trial))
    save_trial_x_plot(trial_labels, trial_x_deltas)


    plt.figure(figsize=(6, 4))
    plt.plot(returns_history, marker="o")
    plt.xlabel("PPO update")
    plt.ylabel("Mean episode return (sum of rewards)")
    plt.title("PPO Training Progress (Transformable Wheel Robot on Flat Ground)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_results_path("transform_ppo_rewards.png"), dpi=150)
    plt.close()
    print("Saved training curve to transform_ppo_rewards.png")

    plt.figure(figsize=(6, 4))
    plt.plot(x_delta_history, marker="o")
    plt.xlabel("PPO update")
    plt.ylabel("Mean episode +x displacement (m)")
    plt.title("Distance Traveled in +x per PPO Update")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_results_path("transform_ppo_x_distance.png"), dpi=150)
    plt.close()
    print("Saved +x distance curve to transform_ppo_x_distance.png")

    return params, returns_history, x_delta_history


def main():
    desired_ctrl_dt = 0.02
    max_steps = 1024

    print("Creating TransformableWheelFlatEnv...")
    env = TransformableWheelFlatEnv(
        xml_path="trans_wheel_robo2_0.xml",
        desired_ctrl_dt=desired_ctrl_dt,
        frame_skip=None,
        max_steps=max_steps,
    )

    print("Starting PPO training...")
    start_time = time.time()
    _params, returns_history, x_delta_history = train_ppo(
        env,
        total_timesteps=4_000,
        rollout_steps=1024,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=2e-4, # 3e-4
        num_epochs=10, # 10
        minibatch_size=256,
        clip_coef=0.2, # 0.2
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=0,
    )
    elapsed = time.time() - start_time
    print(f"PPO training finished in {elapsed:.1f} seconds.")
    print(f"Best return achieved: {max(returns_history):.3f}")


if __name__ == "__main__":
    main()
