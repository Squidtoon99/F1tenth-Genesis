#!/usr/bin/env python3
"""Measure the real car's full-lock turning radius from rosbag2 bags and
compare it to the simulator's kinematic model.

Truth source: /pf/pose/odom (LiDAR particle-filter pose) -> circle fit + yaw-rate
fit give the *measured* steady-state turning radius, independent of the VESC
steering model. /odom provides speed, /drive the commanded steering angle, and
/sensors/servo_position_command the post-clamp servo value.

Run:  ./venv/bin/python analysis/analyze_full_lock.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# --- Sim reference (config.py DEFAULT_CONFIG) ---
SIM_DELTA_MAX = 0.44          # rad
WHEELBASE = 0.33              # m (vesc.yaml; config.py uses 0.325)
SIM_RADIUS = WHEELBASE / math.tan(SIM_DELTA_MAX)

# --- Real-car servo map (f1tenth_stack vesc.yaml) ---
SERVO_GAIN = -1.2135
SERVO_OFFSET = 0.4495
SERVO_MIN = 0.05
SERVO_MAX = 0.85

BAGS = {
    "RIGHT": Path(__file__).parent / "bags" / "full_lock_right",
    "LEFT": Path(__file__).parent / "bags" / "full_lock_left",
}

# ackermann_msgs is not in the default typestore; register it.
ACKERMANN_DRIVE = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""
ACKERMANN_DRIVE_STAMPED = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""


def build_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg(ACKERMANN_DRIVE, "ackermann_msgs/msg/AckermannDrive"))
    types.update(
        get_types_from_msg(ACKERMANN_DRIVE_STAMPED, "ackermann_msgs/msg/AckermannDriveStamped")
    )
    ts.register(types)
    return ts


def yaw_from_quat(z: float, w: float) -> float:
    # planar yaw from (x=0,y=0) quaternion components
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def servo_to_steer(servo: float) -> float:
    return (servo - SERVO_OFFSET) / SERVO_GAIN


def fit_circle(x: np.ndarray, y: np.ndarray):
    """Algebraic (Kasa) circle fit. Returns (cx, cy, R, rms_residual)."""
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    R = math.sqrt(max(c + cx**2 + cy**2, 0.0))
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - R
    return cx, cy, R, float(np.sqrt(np.mean(resid**2)))


def read_bag(path: Path, ts):
    out = {
        "pf": [],     # (t, x, y, yaw)
        "odom": [],   # (t, v, yawrate)
        "drive": [],  # (t, steer_cmd, speed_cmd)
        "servo": [],  # (t, value)
    }
    with AnyReader([path], default_typestore=ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        for conn, t, raw in reader.messages():
            tsec = t * 1e-9
            topic = conn.topic
            msg = reader.deserialize(raw, conn.msgtype)
            if topic == "/pf/pose/odom":
                p = msg.pose.pose
                out["pf"].append((tsec, p.position.x, p.position.y,
                                  yaw_from_quat(p.orientation.z, p.orientation.w)))
            elif topic == "/odom":
                out["odom"].append((tsec, msg.twist.twist.linear.x, msg.twist.twist.angular.z))
            elif topic == "/drive":
                out["drive"].append((tsec, msg.drive.steering_angle, msg.drive.speed))
            elif topic == "/sensors/servo_position_command":
                out["servo"].append((tsec, msg.data))
    for k in out:
        out[k] = np.array(out[k], dtype=float) if out[k] else np.empty((0, 4))
    return out


def steady_mask(t: np.ndarray, frac_lo=0.2, frac_hi=0.95):
    """Trim startup/stop transients: keep the middle of the time span."""
    t0, t1 = t.min(), t.max()
    lo = t0 + frac_lo * (t1 - t0)
    hi = t0 + frac_hi * (t1 - t0)
    return (t >= lo) & (t <= hi)


def analyze(name: str, d: dict):
    print(f"\n{'='*60}\n {name} FULL LOCK\n{'='*60}")

    pf = d["pf"]
    odom = d["odom"]
    drive = d["drive"]
    servo = d["servo"]

    # --- commanded steering & servo (confirm the clamp) ---
    if len(drive):
        m = steady_mask(drive[:, 0])
        steer_cmd = float(np.median(drive[m, 1]))
        speed_cmd = float(np.median(drive[m, 2]))
        ideal_servo = SERVO_GAIN * steer_cmd + SERVO_OFFSET
        print(f"/drive commanded steering : {steer_cmd:+.4f} rad   (sim full lock = "
              f"{math.copysign(SIM_DELTA_MAX, steer_cmd):+.3f})")
        print(f"/drive commanded speed    : {speed_cmd:.3f} m/s")
        print(f"ideal servo for that cmd  : {ideal_servo:+.4f}  "
              f"(limits [{SERVO_MIN}, {SERVO_MAX}])  -> "
              f"{'CLAMPED' if ideal_servo < SERVO_MIN or ideal_servo > SERVO_MAX else 'in range'}")
    if len(servo):
        m = steady_mask(servo[:, 0])
        servo_val = float(np.median(servo[m, 1]))
        steer_from_servo = servo_to_steer(servo_val)
        print(f"servo actually sent       : {servo_val:.4f}  -> "
              f"effective wheel angle = {steer_from_servo:+.4f} rad")

    # --- measured speed (longitudinal, reasonably trustworthy) ---
    v_odom = None
    if len(odom):
        m = steady_mask(odom[:, 0])
        v_odom = float(np.median(np.abs(odom[m, 1])))
        yawrate_model = float(np.median(odom[m, 2]))
        print(f"/odom speed (median)      : {v_odom:.3f} m/s")
        print(f"/odom yaw rate (MODEL)    : {yawrate_model:+.4f} rad/s  "
              f"(derived from steering model -- not ground truth)")

    # --- TRUTH: particle-filter trajectory ---
    if len(pf) < 8:
        print("!! too few /pf/pose/odom samples for a reliable fit")
        return
    m = steady_mask(pf[:, 0])
    t = pf[m, 0]
    x = pf[m, 1]
    y = pf[m, 2]
    yaw = np.unwrap(pf[m, 3])

    cx, cy, R_fit, rms = fit_circle(x, y)

    # yaw-rate from PF heading (true), speed from PF path length (true)
    dt = t[-1] - t[0]
    yawrate_pf = (yaw[-1] - yaw[0]) / dt
    path_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    v_pf = path_len / dt
    R_kin = abs(v_pf / yawrate_pf) if yawrate_pf != 0 else float("nan")

    # effective max steering angle implied by the measured radius
    delta_fit = math.atan(WHEELBASE / R_fit)
    delta_kin = math.atan(WHEELBASE / R_kin)

    print(f"\n--- measured turning (PF, ground truth) ---")
    print(f"circle-fit radius         : {R_fit:.3f} m   (fit RMS {rms*1000:.0f} mm, "
          f"{len(x)} pts)")
    print(f"v/omega radius            : {R_kin:.3f} m   "
          f"(v_pf={v_pf:.3f} m/s, omega={yawrate_pf:+.4f} rad/s)")
    print(f"effective max steer       : {math.degrees(delta_fit):.1f} deg "
          f"({delta_fit:.4f} rad)  [from circle fit]")
    print(f"                            {math.degrees(delta_kin):.1f} deg "
          f"({delta_kin:.4f} rad)  [from v/omega]")

    print(f"\n--- gap vs sim ---")
    print(f"sim min radius @0.44 rad  : {SIM_RADIUS:.3f} m")
    print(f"real min radius (fit)     : {R_fit:.3f} m   "
          f"-> {(R_fit/SIM_RADIUS - 1)*100:+.0f}% vs sim")
    print(f"sim max steer             : {math.degrees(SIM_DELTA_MAX):.1f} deg "
          f"({SIM_DELTA_MAX:.3f} rad)")
    print(f"real max steer (fit)      : {math.degrees(delta_fit):.1f} deg "
          f"({delta_fit:.3f} rad)")
    return {"R_fit": R_fit, "R_kin": R_kin, "delta_fit": delta_fit, "v": v_pf}


def plot(bags_data: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"RIGHT": "tab:red", "LEFT": "tab:blue"}
    for name, d in bags_data.items():
        pf = d["pf"]
        if len(pf) < 8:
            continue
        m = steady_mask(pf[:, 0])
        x, y = pf[m, 1], pf[m, 2]
        cx, cy, R, _ = fit_circle(x, y)
        x0, y0 = x - cx, y - cy
        ax.plot(x0, y0, ".", color=colors[name], ms=4,
                label=f"{name} path (R_fit={R:.2f} m)")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R * np.cos(th), R * np.sin(th), "-", color=colors[name], alpha=0.4)
    # sim reference circle
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(SIM_RADIUS * np.cos(th), SIM_RADIUS * np.sin(th), "k--",
            label=f"sim min radius @0.44 rad ({SIM_RADIUS:.2f} m)")
    ax.set_aspect("equal")
    ax.set_xlabel("x - center (m)")
    ax.set_ylabel("y - center (m)")
    ax.set_title("Full-lock turning circles: real car (PF) vs sim")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nsaved plot -> {out_png}")


def main():
    ts = build_typestore()
    print(f"Sim reference: delta_max={SIM_DELTA_MAX} rad, L={WHEELBASE} m, "
          f"min radius={SIM_RADIUS:.3f} m")
    results = {}
    bags_data = {}
    for name, path in BAGS.items():
        if not path.exists():
            print(f"!! missing bag {path}")
            continue
        bags_data[name] = read_bag(path, ts)
        results[name] = analyze(name, bags_data[name])

    try:
        plot(bags_data, Path(__file__).parent / "full_lock_circles.png")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")

    if "LEFT" in results and "RIGHT" in results and results["LEFT"] and results["RIGHT"]:
        dl = results["LEFT"]["delta_fit"]
        dr = results["RIGHT"]["delta_fit"]
        print(f"\n{'='*60}\n SUMMARY\n{'='*60}")
        print(f"effective max steer  L={math.degrees(dl):.1f} deg  "
              f"R={math.degrees(dr):.1f} deg   "
              f"asymmetry={math.degrees(abs(dl-dr)):.1f} deg")
        mean_delta = 0.5 * (dl + dr)
        print(f"-> recommended sim delta_max for retrain: ~{mean_delta:.3f} rad "
              f"({math.degrees(mean_delta):.1f} deg)")
        print(f"   (vs current {SIM_DELTA_MAX} rad / {math.degrees(SIM_DELTA_MAX):.1f} deg)")


if __name__ == "__main__":
    main()
