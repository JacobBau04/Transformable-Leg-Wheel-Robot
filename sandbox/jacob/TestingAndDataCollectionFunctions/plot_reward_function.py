'''''

RUN WITH: python plot_reward_function.py --plot

High level goal:
- Understand the magnitudes of reward function inputs before using them

Lower level goals:
- Keyboard torque controller for trans_wheel_robo2_2FLAT.xml
- Live plotting (same axes): leg_extension_cost (blue), forward_reward (orange), sideways_cost (green)

'''''

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from collections import deque
import multiprocessing as mp

import numpy as np

try:
    import mujoco
except Exception as e:
    raise SystemExit(
        "Could not import 'mujoco'. Install it with:\n"
        "  pip install mujoco glfw\n\n"
        f"Original import error:\n{e}"
    )

try:
    import glfw
except Exception as e:
    raise SystemExit(
        "Could not import 'glfw'. Install it with:\n"
        "  pip install glfw\n\n"
        f"Original import error:\n{e}"
    )


@dataclass
class ActKey:
    name: str
    inc_key: int
    dec_key: int


def _keycode(key_name: str) -> int:
    """Map a human-friendly key label to a GLFW key code."""
    key_name = key_name.strip().upper()
    special = {
        ";": glfw.KEY_SEMICOLON,
        "[": glfw.KEY_LEFT_BRACKET,
        "]": glfw.KEY_RIGHT_BRACKET,
        "SPACE": glfw.KEY_SPACE,
        "ESC": glfw.KEY_ESCAPE,
        "BACKSPACE": glfw.KEY_BACKSPACE,
    }
    if key_name in special:
        return special[key_name]
    if len(key_name) == 1 and "A" <= key_name <= "Z":
        return getattr(glfw, f"KEY_{key_name}")
    if len(key_name) == 1 and "0" <= key_name <= "9":
        return getattr(glfw, f"KEY_{key_name}")
    raise ValueError(f"Unknown key label: {key_name}")


def build_default_bindings() -> List[ActKey]:
    return [
        ActKey("front_left_wheel_joint_ctrl",  _keycode("Q"), _keycode("A")),
        ActKey("front_right_wheel_joint_ctrl", _keycode("W"), _keycode("S")),
        ActKey("rear_left_wheel_joint_ctrl",   _keycode("E"), _keycode("D")),
        ActKey("rear_right_wheel_joint_ctrl",  _keycode("R"), _keycode("F")),
        ActKey("front_left_wheel_0_extension_joint_ctrl",  _keycode("U"), _keycode("J")),
        ActKey("front_right_wheel_0_extension_joint_ctrl", _keycode("I"), _keycode("K")),
        ActKey("rear_left_wheel_0_extension_joint_ctrl",   _keycode("O"), _keycode("L")),
        ActKey("rear_right_wheel_0_extension_joint_ctrl",  _keycode("P"), _keycode(";")),
    ]


def plotter_process(q: "mp.Queue", window_s: float, plot_hz: float):
    """
    Runs in a separate process. Receives tuples (t, leg_extension_cost, forward_reward, sideways_cost)
    and live-plots all signals on the SAME axes.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            "Plotter: could not import matplotlib. Install with:\n"
            "  pip install matplotlib\n"
            f"Original error:\n{e}"
        )
        return

    plt.ion()
    fig, ax = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Live Metrics")

    ts = deque()
    leg_costs = deque()
    fwd_rewards = deque()
    side_costs = deque()

    # Requested colors: blue + orange. Added: green for sideways_cost.
    (line_leg,) = ax.plot([], [], color="tab:blue", linewidth=1.5, label="leg_extension_cost")
    (line_fwd,) = ax.plot([], [], color="tab:orange", linewidth=1.5, label="forward_reward")
    (line_side,) = ax.plot([], [], color="tab:green", linewidth=1.5, label="sideways_cost")

    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend(loc="upper right")

    txt = fig.text(0.01, 0.01, "", fontsize=10)

    last_draw = time.perf_counter()
    draw_period = 1.0 / max(1.0, float(plot_hz))

    while True:
        got_any = False
        while True:
            try:
                item = q.get_nowait()
            except Exception:
                break

            if item is None:
                plt.ioff()
                plt.close(fig)
                return

            # Expect: (t, leg_c, fwd_r, side_c)
            t, leg_c, fwd_r, side_c = item
            ts.append(float(t))
            leg_costs.append(float(leg_c))
            fwd_rewards.append(float(fwd_r))
            side_costs.append(float(side_c))
            got_any = True

        # Keep only last window_s seconds
        if ts:
            tmax = ts[-1]
            tmin = tmax - float(window_s)
            while ts and ts[0] < tmin:
                ts.popleft()
                leg_costs.popleft()
                fwd_rewards.popleft()
                side_costs.popleft()

        now = time.perf_counter()
        if got_any and (now - last_draw) >= draw_period and ts:
            line_leg.set_data(ts, leg_costs)
            line_fwd.set_data(ts, fwd_rewards)
            line_side.set_data(ts, side_costs)

            ax.relim()
            ax.autoscale_view()

            txt.set_text(
                f"t={ts[-1]:.2f}s | leg_cost={leg_costs[-1]:.4g} | "
                f"fwd_reward={fwd_rewards[-1]:.4g} | sideways_cost={side_costs[-1]:.4g}"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_draw = now

        plt.pause(0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="trans_wheel_robo2_2FLAT.xml")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--print_hz", type=float, default=1.0)

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_hz", type=float, default=20.0)
    ap.add_argument("--plot_window_s", type=float, default=15.0)

    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    bindings = build_default_bindings()
    act_name_to_id: Dict[str, int] = {}
    for i in range(model.nu):
        act_name_to_id[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] = i

    keymap: Dict[int, Tuple[int, float]] = {}
    missing = []
    for b in bindings:
        if b.name not in act_name_to_id:
            missing.append(b.name)
            continue
        aid = act_name_to_id[b.name]
        keymap[b.inc_key] = (aid, +1.0)
        keymap[b.dec_key] = (aid, -1.0)

    if missing:
        print("WARNING: These actuators were not found in the model (bindings will be skipped):")
        for m in missing:
            print("  -", m)
        print()

    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    target = np.zeros(model.nu, dtype=np.float64)
    step = float(args.step)

    plot_q = None
    plot_proc = None
    if args.plot:
        plot_q = mp.Queue(maxsize=2000)
        plot_proc = mp.Process(
            target=plotter_process,
            args=(plot_q, float(args.plot_window_s), float(args.plot_hz)),
            daemon=True,
        )
        plot_proc.start()
        print("Live plot enabled (separate process). If you don't see it:\n  pip install matplotlib\n")

    if not glfw.init():
        raise SystemExit("Failed to initialize GLFW.")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, 1)
    window = glfw.create_window(1200, 800, "MuJoCo torque keyboard control", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Could not create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=20000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam.azimuth = 90
    cam.elevation = -20
    cam.distance = 3.0
    cam.lookat[:] = np.array([0.0, 0.0, 0.2])

    def scroll_callback(_window, xoffset, yoffset):
        cam.distance *= (0.9 ** yoffset)
        cam.distance = float(np.clip(cam.distance, 0.2, 50.0))

    glfw.set_scroll_callback(window, scroll_callback)

    def clamp_and_apply():
        nonlocal target
        target = np.clip(target, ctrl_low, ctrl_high)
        data.ctrl[:] = target

    def zero_all():
        nonlocal target
        target[:] = 0.0
        clamp_and_apply()

    def print_help():
        print("\n--- Key Bindings ---")
        print("Wheels:  Q/A (FL), W/S (FR), E/D (RL), R/F (RR)")
        print("Leg0:    U/J (FL), I/K (FR), O/L (RL), P/; (RR)")
        print("Global:  [ / ] step-,step+ | SPACE zero | BACKSPACE reset | ESC quit")
        print("--------------------\n")

    print_help()

    last_print = 0.0
    print_period = 1.0 / max(0.1, float(args.print_hz))

    def key_callback(_window, key, scancode, action, mods):
        nonlocal step
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        if key == glfw.KEY_SPACE:
            zero_all()
            return

        if key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetData(model, data)
            zero_all()
            return

        if key == glfw.KEY_LEFT_BRACKET:
            step = max(1e-4, step * 0.8)
            print(f"step = {step:.5f}")
            return

        if key == glfw.KEY_RIGHT_BRACKET:
            step = min(10.0, step * 1.25)
            print(f"step = {step:.5f}")
            return

        if key in keymap:
            aid, sgn = keymap[key]
            target[aid] += sgn * step
            clamp_and_apply()

    glfw.set_key_callback(window, key_callback)

    sim_dt = float(model.opt.timestep)
    t_prev = time.perf_counter()
    acc = 0.0

    clamp_and_apply()

    def calc_ext_cost(data, full_ext_angle: float) -> float:
        leg_angles = np.array([data.qpos[i] for i in [8, 12, 16, 20]], dtype=np.float32)
        leg_omegas = np.array([data.qvel[i] for i in [8, 12, 16, 20]], dtype=np.float32)
        leg_angles_normalized = np.clip(((leg_angles - 1.19) / full_ext_angle), -1, 1)
        dist_from_closed = 1 - np.abs(leg_angles_normalized)
        angular_vel_dir_cost = (-np.sign(leg_angles_normalized) * leg_omegas * dist_from_closed)
        return 1e-3 * np.sqrt(np.sum(np.square(angular_vel_dir_cost + 2 * dist_from_closed)))



    while not glfw.window_should_close(window):
        t_now = time.perf_counter()
        acc += (t_now - t_prev)
        t_prev = t_now

        while acc >= sim_dt:
            mujoco.mj_step(model, data)
            acc -= sim_dt

        

        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        if (t_now - last_print) >= print_period:
            vx = float(data.qvel[0])
            vy = float(data.qvel[1])

            forward_reward = vx
            sideways_cost = 0.03 * (vy ** 2) / (vx ** 2 + vy ** 2 + 1e-8)

            full_ext_angle = 2.237
            leg_extension_cost = calc_ext_cost(data, full_ext_angle)



            print("leg_extension_cost: ", leg_extension_cost)
            print("forward_reward: ", forward_reward)
            print("sideways_cost: ", sideways_cost)

            if plot_q is not None:
                try:
                    plot_q.put_nowait(
                        (float(data.time), float(leg_extension_cost), float(forward_reward), float(sideways_cost))
                    )
                except Exception:
                    pass

            last_print = t_now

    print("\nExiting.")
    if plot_q is not None:
        try:
            plot_q.put_nowait(None)
        except Exception:
            pass
        if plot_proc is not None and plot_proc.is_alive():
            plot_proc.join(timeout=1.0)

    glfw.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
