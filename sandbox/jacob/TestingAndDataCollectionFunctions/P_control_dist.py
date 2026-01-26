#!/usr/bin/env python3
'''''

RUN WITH: python P_control_dist.py --plot

Proportional wheel controller for trans_wheel_robo2_2FLAT.xml

Goals:
- Face the +X world direction (yaw -> 0 rad)
- Drive to x = +0.50 m (50 cm)

Control (P-only):
- forward_cmd = Kp_x * (x_goal - x)
- turn_cmd    = Kp_yaw * (yaw_goal - yaw)
- left_wheels  = forward_cmd - turn_cmd
- right_wheels = forward_cmd + turn_cmd



'''''

import argparse
import time
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


# ---------------- Utilities ----------------

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Quaternion q = [w, x, y, z] -> yaw about world Z (assuming Z-up)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # yaw (Z) from quaternion (w,x,y,z)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def find_free_joint_root(model: "mujoco.MjModel"):
    """Return (qposadr, dofadr) for the first FREE joint, else raise."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    raise RuntimeError("No FREE joint found. This controller expects a floating-base model.")


# ---------------- Plotter (separate process) ----------------

def plotter_process(q: "mp.Queue", window_s: float, plot_hz: float):
    """
    Receives tuples (t, x, yaw_err) and plots:
      x (blue) and yaw_err (orange) on the same axes.
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
    fig.canvas.manager.set_window_title("x + angular_err (live)")

    ts = deque()
    xs = deque()
    yawerrs = deque()

    (line_x,) = ax.plot([], [], color="tab:blue", linewidth=1.6, label="x (m)")
    (line_yaw,) = ax.plot([], [], color="tab:orange", linewidth=1.6, label="angular_err (rad)")

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

            t, x, yaw_err = item
            ts.append(float(t))
            xs.append(float(x))
            yawerrs.append(float(yaw_err))
            got_any = True

        if ts:
            tmax = ts[-1]
            tmin = tmax - float(window_s)
            while ts and ts[0] < tmin:
                ts.popleft()
                xs.popleft()
                yawerrs.popleft()

        now = time.perf_counter()
        if got_any and (now - last_draw) >= draw_period and ts:
            line_x.set_data(ts, xs)
            line_yaw.set_data(ts, yawerrs)

            ax.relim()
            ax.autoscale_view()

            txt.set_text(f"t={ts[-1]:.2f}s | x={xs[-1]:.3f} m | yaw_err={yawerrs[-1]:+.3f} rad")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_draw = now

        plt.pause(0.001)


# ---------------- Main ----------------

def main(dist):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="trans_wheel_robo2_2FLAT.xml")
    ap.add_argument("--x_goal", type=float, default=dist, help="Target x position (m) (default: 0.50)")
    ap.add_argument("--yaw_goal", type=float, default=0.0, help="Target yaw (rad) for +X (default: 0.0)")

    ap.add_argument("--kp_x", type=float, default=6.0, help="P gain on x position error")
    ap.add_argument("--kp_yaw", type=float, default=3.0, help="P gain on yaw error")

    ap.add_argument("--tol_x", type=float, default=0.01, help="Stop band for x (m)")
    ap.add_argument("--tol_yaw", type=float, default=0.05, help="Stop band for yaw (rad)")

    ap.add_argument("--print_hz", type=float, default=10.0)

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_hz", type=float, default=30.0)
    ap.add_argument("--plot_window_s", type=float, default=15.0)

    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    qposadr, dofadr = find_free_joint_root(model)

    # Actuator IDs (wheels only)
    def actuator_id(name: str) -> int:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator not found: {name}")
        return int(aid)

    A_FL = actuator_id("front_left_wheel_joint_ctrl")
    A_FR = actuator_id("front_right_wheel_joint_ctrl")
    A_RL = actuator_id("rear_left_wheel_joint_ctrl")
    A_RR = actuator_id("rear_right_wheel_joint_ctrl")

    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    ctrl = np.zeros(model.nu, dtype=np.float64)
    paused = False

    # ---- optional plot process ----
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

    # ---- GLFW + MuJoCo render setup ----
    if not glfw.init():
        raise SystemExit("Failed to initialize GLFW.")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, 1)
    window = glfw.create_window(1200, 800, "MuJoCo P controller: x + yaw", None, None)
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
        cam.distance = float(np.clip(cam.distance, 0.2, 100*dist))

    glfw.set_scroll_callback(window, scroll_callback)

    def apply_ctrl():
        nonlocal ctrl
        ctrl = np.clip(ctrl, ctrl_low, ctrl_high)
        data.ctrl[:] = ctrl

    def zero_wheels():
        ctrl[A_FL] = 0.0
        ctrl[A_FR] = 0.0
        ctrl[A_RL] = 0.0
        ctrl[A_RR] = 0.0
        apply_ctrl()

    def key_callback(_window, key, scancode, action, mods):
        nonlocal paused
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        if key == glfw.KEY_SPACE:
            paused = not paused
            if paused:
                zero_wheels()
                print("[PAUSED] wheel torques zeroed (SPACE to resume)")
            else:
                print("[RUNNING] controller resumed")
            return

        if key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetData(model, data)
            zero_wheels()
            paused = False
            print("[RESET] simulation reset")

    glfw.set_key_callback(window, key_callback)

    # ---- Simulation loop (real-time) ----
    sim_dt = float(model.opt.timestep)
    t_prev = time.perf_counter()
    acc = 0.0

    last_print = 0.0
    print_period = 1.0 / max(0.2, float(args.print_hz))

    zero_wheels()

    while not glfw.window_should_close(window):
        # timing
        t_now = time.perf_counter()
        acc += (t_now - t_prev)
        t_prev = t_now

        # Controller update at sim rate (before stepping)
        if not paused:
            x = float(data.qpos[qposadr + 0])
            quat = np.array(data.qpos[qposadr + 3: qposadr + 7], dtype=np.float64)  # wxyz
            yaw = yaw_from_quat_wxyz(quat)

            x_err = float(args.x_goal - x)
            yaw_err = wrap_to_pi(float(args.yaw_goal - yaw))

            # P-only commands
            forward_cmd = float(args.kp_x * x_err)
            turn_cmd = float(args.kp_yaw * yaw_err)

            # Optional: reduce forward push if you’re pointed away from +X (helps stability)
            forward_cmd *= float(max(0.0, np.cos(yaw_err)))

            # Stop band: if you're basically there, stop driving
            if abs(x_err) < float(args.tol_x) and abs(yaw_err) < float(args.tol_yaw):
                forward_cmd = 0.0
                turn_cmd = 0.0

            left = forward_cmd - turn_cmd
            right = forward_cmd + turn_cmd

            ctrl[A_FL] = left
            ctrl[A_RL] = left
            ctrl[A_FR] = right
            ctrl[A_RR] = right
            apply_ctrl()

            if plot_q is not None:
                try:
                    plot_q.put_nowait((float(data.time), x, yaw_err))
                except Exception:
                    pass

        # step simulation
        while acc >= sim_dt:
            mujoco.mj_step(model, data)
            acc -= sim_dt

        # render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # printing
        if (t_now - last_print) >= print_period:
            x = float(data.qpos[qposadr + 0])
            quat = np.array(data.qpos[qposadr + 3: qposadr + 7], dtype=np.float64)
            yaw = yaw_from_quat_wxyz(quat)
            x_err = float(args.x_goal - x)
            yaw_err = wrap_to_pi(float(args.yaw_goal - yaw))
            print(f"t={data.time:6.2f} | x={x:+.3f} m (err {x_err:+.3f}) | yaw={yaw:+.3f} rad (err {yaw_err:+.3f})")
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
    dist = 0.50
    main(dist)
