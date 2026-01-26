'''''

RUN WITH: python collect_IMU_data.py --plot

High level goal:
- understand acceleration profile + bump sensor data of robot bumping a step to maybe run that through a CNN to detect box hits without the bump sensor

Lower level goals:
- Live plot IMU data while running
- Log 120 seconds of data
- Keyboard torque controller for trans_wheel_robo2_2BOX.xml
- Save data to ONE file (.npz, in logs folder)
- After saving: load that same file and plot the saved data
    Figure 1: accelerometer (ax, ay, az)
    Figure 2: gyroscope (wx, wy, wz)

ONLY WORKS WITH trans_wheel_robo2_2BOX.xml (needs bump sensor)
Requires XML sensors:
  <accelerometer name="root_acc" site="root_site"/>
  <gyro          name="root_gyro" site="root_site"/>
Optional:
  <touch         name="box_touch" site="traverse_box_site"/>

Also expects a geom named:
  <geom name="traverse_box" ... />

'''''

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import multiprocessing as mp
import os
from datetime import datetime

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
    special = {
        "SPACE": glfw.KEY_SPACE,
        "ESC": glfw.KEY_ESCAPE,
        "BACKSPACE": glfw.KEY_BACKSPACE,
        "SEMICOLON": glfw.KEY_SEMICOLON,
        ";": glfw.KEY_SEMICOLON,
        "[": glfw.KEY_LEFT_BRACKET,
        "]": glfw.KEY_RIGHT_BRACKET,
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
        ActKey("rear_right_wheel_0_extension_joint_ctrl",  _keycode("P"), _keycode("SEMICOLON")),
    ]


def build_name_to_act_id(model: "mujoco.MjModel") -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            mapping[name] = i
    return mapping


def _sensor_slice(model: "mujoco.MjModel", sensor_name: str, expect_dim: int = None) -> slice:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise SystemExit(
            f"Sensor '{sensor_name}' not found in model.\n"
            "Make sure your XML includes the named sensor."
        )
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    if expect_dim is not None and dim != expect_dim:
        raise SystemExit(f"Sensor '{sensor_name}' expected dim={expect_dim} but got dim={dim}.")
    return slice(adr, adr + dim)


def plotter_process(q: "mp.Queue", window_s: float, plot_hz: float):
    """
    Receives tuples:
      (t, ax, ay, az, wx, wy, wz, hit_box, box_touch)
    and live-plots:
      - primary axis: ax,ay,az, wx,wy,wz
      - secondary axis: hit_box, box_touch
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
    fig, ax1 = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Live IMU + Box Contact")
    ax2 = ax1.twinx()

    ts = deque()

    ax_d = deque(); ay_d = deque(); az_d = deque()
    wx_d = deque(); wy_d = deque(); wz_d = deque()
    hit_d = deque()
    touch_d = deque()

    (l_ax,) = ax1.plot([], [], linewidth=1.2, label="ax [m/s^2]")
    (l_ay,) = ax1.plot([], [], linewidth=1.2, label="ay [m/s^2]")
    (l_az,) = ax1.plot([], [], linewidth=1.2, label="az [m/s^2]")

    (l_wx,) = ax1.plot([], [], linewidth=1.2, label="wx [rad/s]")
    (l_wy,) = ax1.plot([], [], linewidth=1.2, label="wy [rad/s]")
    (l_wz,) = ax1.plot([], [], linewidth=1.2, label="wz [rad/s]")

    (l_hit,) = ax2.plot([], [], linewidth=2.0, label="hit_box (0/1)")
    (l_touch,) = ax2.plot([], [], linewidth=1.5, label="box_touch")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("IMU (acc + gyro)")
    ax2.set_ylabel("Contact signals")
    ax1.set_title("IMU + Box Contact")
    ax1.grid(True, alpha=0.3)

    ax1.legend(loc="upper left", ncols=2)
    ax2.legend(loc="upper right")

    txt = ax1.text(
        0.02, 0.98, "",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    last_draw = 0.0
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

            t, axv, ayv, azv, wxv, wyv, wzv, hitv, touchv = item

            ts.append(float(t))
            ax_d.append(float(axv)); ay_d.append(float(ayv)); az_d.append(float(azv))
            wx_d.append(float(wxv)); wy_d.append(float(wyv)); wz_d.append(float(wzv))
            hit_d.append(float(hitv))
            touch_d.append(float(touchv))
            got_any = True

        if ts:
            tmax = ts[-1]
            tmin = tmax - float(window_s)
            while ts and ts[0] < tmin:
                ts.popleft()
                ax_d.popleft(); ay_d.popleft(); az_d.popleft()
                wx_d.popleft(); wy_d.popleft(); wz_d.popleft()
                hit_d.popleft()
                touch_d.popleft()

        now = time.perf_counter()
        if got_any and (now - last_draw) >= draw_period and ts:
            l_ax.set_data(ts, ax_d); l_ay.set_data(ts, ay_d); l_az.set_data(ts, az_d)
            l_wx.set_data(ts, wx_d); l_wy.set_data(ts, wy_d); l_wz.set_data(ts, wz_d)
            l_hit.set_data(ts, hit_d)
            l_touch.set_data(ts, touch_d)

            ax1.relim()
            ax1.autoscale_view()

            ax2.relim()
            ax2.autoscale_view()
            y0, y1 = ax2.get_ylim()
            ax2.set_ylim(min(y0, -0.05), max(y1, 1.2))

            txt.set_text(
                "t={:.2f}s | hit_box={} | box_touch={:.3g}\n"
                "a=[{:+.2f},{:+.2f},{:+.2f}] m/s^2\n"
                "w=[{:+.2f},{:+.2f},{:+.2f}] rad/s".format(
                    ts[-1],
                    int(round(hit_d[-1])),
                    touch_d[-1],
                    ax_d[-1], ay_d[-1], az_d[-1],
                    wx_d[-1], wy_d[-1], wz_d[-1],
                )
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_draw = now

        plt.pause(0.001)


def plot_saved_npz(npz_path: str):
    """
    Load the saved .npz log and produce:
      Figure 1: accelerometer
      Figure 2: gyroscope
    Both figures overlay hit_box + box_touch on a secondary axis.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            "Could not import matplotlib for plotting. Install with:\n"
            "  pip install matplotlib\n"
            f"Original error:\n{e}"
        )
        return

    d = np.load(npz_path, allow_pickle=True)

    t = d["t"]          # shape (N,)
    acc = d["acc"]      # shape (N,3)
    gyro = d["gyro"]    # shape (N,3)
    hit = d["hit_box"]  # shape (N,)
    touch = d["box_touch"]  # shape (N,)

    # Figure 1: accelerometer
    fig1, ax1 = plt.subplots(1, 1)
    ax1.plot(t, acc[:, 0], label="ax [m/s^2]")
    ax1.plot(t, acc[:, 1], label="ay [m/s^2]")
    ax1.plot(t, acc[:, 2], label="az [m/s^2]")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Acceleration [m/s^2]")
    ax1.set_title("Saved accelerometer (root_acc)")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(t, hit, linewidth=2.0, label="hit_box (0/1)")
    ax1b.plot(t, touch, linewidth=1.2, label="box_touch")
    ax1b.set_ylabel("Contact signals")
    y0, y1 = ax1b.get_ylim()
    ax1b.set_ylim(min(y0, -0.05), max(y1, 1.2))

    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")

    # Figure 2: gyroscope
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(t, gyro[:, 0], label="wx [rad/s]")
    ax2.plot(t, gyro[:, 1], label="wy [rad/s]")
    ax2.plot(t, gyro[:, 2], label="wz [rad/s]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angular velocity [rad/s]")
    ax2.set_title("Saved gyroscope (root_gyro)")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(t, hit, linewidth=2.0, label="hit_box (0/1)")
    ax2b.plot(t, touch, linewidth=1.2, label="box_touch")
    ax2b.set_ylabel("Contact signals")
    y0, y1 = ax2b.get_ylim()
    ax2b.set_ylim(min(y0, -0.05), max(y1, 1.2))

    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="trans_wheel_robo2_2BOX.xml")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--print_hz", type=float, default=10.0)

    ap.add_argument("--plot", action="store_true", help="Enable live plotting in separate process")
    ap.add_argument("--plot_window_s", type=float, default=10.0)
    ap.add_argument("--plot_hz", type=float, default=20.0)

    ap.add_argument("--log_s", type=float, default=120.0, help="How many seconds of sim-time to log before auto-saving")
    ap.add_argument("--outdir", type=str, default="logs", help="Directory to save .npz log file")
    ap.add_argument("--tag", type=str, default="", help="Optional tag appended to filename")

    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # Sensors
    acc_sl = _sensor_slice(model, "root_acc", expect_dim=3)
    gyro_sl = _sensor_slice(model, "root_gyro", expect_dim=3)

    # Optional: box_touch
    try:
        box_touch_sl = _sensor_slice(model, "box_touch", expect_dim=None)
        box_touch_ok = True
    except SystemExit:
        box_touch_ok = False
        box_touch_sl = slice(0, 0)

    # Geom IDs for robust contact detection
    box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "traverse_box")
    if box_gid < 0:
        raise SystemExit("Geom named 'traverse_box' not found. Check your XML geom name.")

    ground_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    # ground may not exist by that exact name; that's okay, we just won't exclude it then.

    act_map = build_name_to_act_id(model)
    bindings = build_default_bindings()

    missing = [b.name for b in bindings if b.name not in act_map]
    if missing:
        print("WARNING: Some actuator names in bindings not found in model:")
        for n in missing:
            print("  -", n)
        print("You can still run, but those keys won't do anything.\n")

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

    def clamp_and_apply():
        np.clip(target, ctrl_low, ctrl_high, out=target)
        data.ctrl[:] = target

    def zero_all():
        target[:] = 0.0
        clamp_and_apply()

    def print_help():
        print("\n--- Key Bindings ---")
        print("Wheels:  Q/A (FL), W/S (FR), E/D (RL), R/F (RR)")
        print("Leg0:    U/J (FL), I/K (FR), O/L (RL), P/; (RR)")
        print("Global:  [ / ] step-,step+ | SPACE zero | BACKSPACE reset | ESC quit")
        print("--------------------\n")
        print(f"Logging will auto-stop after {args.log_s:.2f}s sim-time and save a .npz log.\n")

    print_help()

    # GLFW setup
    if not glfw.init():
        raise SystemExit("Could not init GLFW.")

    glfw.window_hint(glfw.VISIBLE, 1)
    window = glfw.create_window(1200, 800, "MuJoCo torque keyboard control (contact + logging)", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Could not create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=20000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    def on_key(window, key, scancode, action, mods):
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
            step = max(1e-4, step * 2)
            print(f"step = {step:.5f}")
            return

        if key == glfw.KEY_RIGHT_BRACKET:
            step = min(10.0, step * 1.25)
            print(f"step = {step:.5f}")
            return

        for b in bindings:
            act_id = act_map.get(b.name, None)
            if act_id is None:
                continue
            if key == b.inc_key:
                target[act_id] += step
            elif key == b.dec_key:
                target[act_id] -= step

        clamp_and_apply()

    glfw.set_key_callback(window, on_key)

    # -------- Logging buffers (store EVERY sim step) --------
    t_log = []
    acc_log = []
    gyro_log = []
    hit_log = []
    touch_log = []

    # Timing control
    sim_dt = float(model.opt.timestep)
    wall_prev = time.perf_counter()
    acc_wall = 0.0

    last_print_wall = time.perf_counter()
    print_period = 1.0 / max(0.1, float(args.print_hz))

    next_plot_t = 0.0

    # Stop after log_s sim seconds
    t0 = float(data.time)
    t_end = t0 + float(args.log_s)

    print(f"Logging started at sim t={t0:.3f}s, will stop at t={t_end:.3f}s\n")

    while (not glfw.window_should_close(window)) and (float(data.time) < t_end):
        wall_now = time.perf_counter()
        acc_wall += (wall_now - wall_prev)
        wall_prev = wall_now

        # Step sim in real-time-ish and log every sim step
        while acc_wall >= sim_dt and float(data.time) < t_end:
            mujoco.mj_step(model, data)
            acc_wall -= sim_dt

            # Read sensors
            acc_vec = np.array(data.sensordata[acc_sl], dtype=np.float64)
            gyro_vec = np.array(data.sensordata[gyro_sl], dtype=np.float64)

            if box_touch_ok:
                box_touch_val = float(np.sum(np.array(data.sensordata[box_touch_sl], dtype=np.float64)))
            else:
                box_touch_val = 0.0

            # Robust hit_box
            hit_box = 0
            for i in range(int(data.ncon)):
                c = data.contact[i]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                if (g1 == box_gid) or (g2 == box_gid):
                    other = g2 if (g1 == box_gid) else g1
                    if ground_gid >= 0 and other == ground_gid:
                        continue
                    hit_box = 1
                    break

            # Log
            t_log.append(float(data.time))
            acc_log.append(acc_vec)
            gyro_log.append(gyro_vec)
            hit_log.append(float(hit_box))
            touch_log.append(float(box_touch_val))

            # Throttle live plot messages (plot_hz)
            if plot_q is not None and float(data.time) >= next_plot_t:
                try:
                    plot_q.put_nowait(
                        (
                            float(data.time),
                            float(acc_vec[0]), float(acc_vec[1]), float(acc_vec[2]),
                            float(gyro_vec[0]), float(gyro_vec[1]), float(gyro_vec[2]),
                            float(hit_box),
                            float(box_touch_val),
                        )
                    )
                except Exception:
                    pass
                next_plot_t = float(data.time) + (1.0 / max(1.0, float(args.plot_hz)))

        # Render
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, con)
        glfw.swap_buffers(window)
        glfw.poll_events()

        # Status print
        if (wall_now - last_print_wall) >= print_period and len(t_log) > 0:
            remaining = max(0.0, t_end - float(data.time))
            print(
                f"t={data.time:8.3f} | remaining={remaining:6.2f}s | "
                f"hit_box={int(round(hit_log[-1]))} | box_touch={touch_log[-1]:8.3g}"
            )
            last_print_wall = wall_now

    # Stop viewer
    glfw.terminate()

    if len(t_log) == 0:
        print("No data logged (exited early).")
        if plot_q is not None:
            try:
                plot_q.put_nowait(None)
            except Exception:
                pass
        return

    # Convert logs to arrays and normalize time to start at 0
    t_log = np.asarray(t_log, dtype=np.float64)
    t_rel = t_log - t_log[0]
    acc_arr = np.asarray(acc_log, dtype=np.float64)
    gyro_arr = np.asarray(gyro_log, dtype=np.float64)
    hit_arr = np.asarray(hit_log, dtype=np.float64)
    touch_arr = np.asarray(touch_log, dtype=np.float64)

    # Save ONE file (.npz)
    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    npz_path = os.path.join(args.outdir, f"imu_contact_{stamp}{tag}.npz")

    np.savez(
        npz_path,
        t=t_rel,
        acc=acc_arr,
        gyro=gyro_arr,
        hit_box=hit_arr,
        box_touch=touch_arr,
        model_path=str(args.model),
        dt=float(model.opt.timestep),
        note="t is relative to start of logging; gyro is angular velocity rad/s; acc is m/s^2",
    )

    print("\n--- Saved log ---")
    print(f"{npz_path}")
    print(f"Samples: {t_rel.shape[0]} | Duration (s): {t_rel[-1]:.6f} | dt (nominal): {model.opt.timestep}\n")

    # Close live plotter (if any)
    if plot_q is not None:
        try:
            plot_q.put_nowait(None)
        except Exception:
            pass
        if plot_proc is not None and plot_proc.is_alive():
            plot_proc.join(timeout=1.0)

    # Now load the same file and plot what was saved
    plot_saved_npz(npz_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
