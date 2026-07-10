#!/usr/bin/env python3
"""Analyze an on-car RL rosbag: manual vs auton, obs, actions, mux chain."""

from __future__ import annotations

import argparse
import bisect
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any


def _read_bag_rosbags(bag_path: str) -> dict[str, list[tuple[float, Any]]]:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    out: dict[str, list[tuple[float, object]]] = {}
    with Reader(bag_path) as reader:
        for conn in reader.connections:
            out.setdefault(conn.topic, [])
        for conn, t_ns, raw in reader.messages():
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            out[conn.topic].append((t_ns * 1e-9, msg))
    return out


def _read_bag(bag_path: str) -> dict[str, list[tuple[float, Any]]]:
    try:
        import rosbag2_py  # noqa: F401
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", ""),
        )
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        msg_classes = {name: get_message(type_map[name]) for name in type_map}
        out: dict[str, list[tuple[float, object]]] = {name: [] for name in type_map}
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            if topic not in out:
                continue
            msg = deserialize_message(raw, msg_classes[topic])
            out[topic].append((t_ns * 1e-9, msg))
        return out
    except ImportError:
        return _read_bag_rosbags(bag_path)


def _zoh(series: list[tuple[float, object]], t: float):
    if not series:
        return None
    times = [s[0] for s in series]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        i = 0
    return series[i][1]


def _yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


@dataclass
class Sample:
    t: float
    mode: str
    vx: float
    speed: float
    pf_x: float
    pf_y: float
    pf_yaw: float
    ey: float
    contact: float
    theta_err: float
    throttle: float
    steer: float
    drive_speed: float
    drive_steer: float
    ack_speed: float
    ack_steer: float
    teleop_speed: float
    teleop_fresh: bool
    auton_button: bool
    slip_max: float
    opp_present: float
    obs_min: float
    obs_max: float


def _joy_auton(msg) -> bool:
    return len(msg.buttons) > 5 and int(msg.buttons[5]) == 1


def analyze(bag_path: str) -> int:
    topics = _read_bag(bag_path)
    actions = topics.get("/rl/action", [])
    if not actions:
        print("ERROR: no /rl/action in bag", file=sys.stderr)
        return 1

    teleop_series = topics.get("/teleop", [])
    joy_series = topics.get("/joy", [])
    t0 = actions[0][0]
    samples: list[Sample] = []

    for t, act in actions:
        rel_t = t - t0
        odom = _zoh(topics.get("/odom", []), t)
        pose = _zoh(topics.get("/pf/pose/odom", []), t)
        drive = _zoh(topics.get("/drive", []), t)
        ack = _zoh(topics.get("/ackermann_cmd", []), t)
        obs = _zoh(topics.get("/rl/observation", []), t)
        teleop = _zoh(teleop_series, t)
        joy = _zoh(joy_series, t)

        teleop_t = teleop_series[bisect.bisect_right([s[0] for s in teleop_series], t) - 1][0] if teleop_series else None
        teleop_fresh = teleop_t is not None and (t - teleop_t) < 0.25
        auton_button = _joy_auton(joy) if joy is not None else False

        vx = vy = 0.0
        if odom is not None:
            vx = float(odom.twist.twist.linear.x)
            vy = float(odom.twist.twist.linear.y)
        speed = math.hypot(vx, vy)

        pf_x = pf_y = pf_yaw = float("nan")
        if pose is not None:
            pf_x = float(pose.pose.pose.position.x)
            pf_y = float(pose.pose.pose.position.y)
            pf_yaw = _yaw_from_quat(pose.pose.pose.orientation)

        throttle = float(act.data[0]) if len(act.data) >= 1 else 0.0
        steer = float(act.data[1]) if len(act.data) >= 2 else 0.0

        drive_speed = float(drive.drive.speed) if drive is not None else 0.0
        drive_steer = float(drive.drive.steering_angle) if drive is not None else 0.0
        ack_speed = float(ack.drive.speed) if ack is not None else 0.0
        ack_steer = float(ack.drive.steering_angle) if ack is not None else 0.0
        teleop_speed = float(teleop.drive.speed) if teleop is not None else 0.0

        obs_data = list(obs.data) if obs is not None and hasattr(obs, "data") else []
        ey = obs_data[10] if len(obs_data) > 10 else float("nan")
        contact = obs_data[11] if len(obs_data) > 11 else float("nan")
        theta_err = obs_data[9] if len(obs_data) > 9 else float("nan")
        slip = obs_data[372:380] if len(obs_data) >= 380 else []
        slip_max = max((abs(x) for x in slip), default=0.0)
        opp_present = obs_data[380] if len(obs_data) > 380 else 0.0
        obs_min = min(obs_data) if obs_data else float("nan")
        obs_max = max(obs_data) if obs_data else float("nan")

        if auton_button or (not teleop_fresh and throttle != 0.0):
            mode = "auton"
        elif teleop_fresh and abs(teleop_speed) > 0.02:
            mode = "manual"
        elif speed > 0.15:
            mode = "moving"
        else:
            mode = "idle"

        samples.append(
            Sample(
                t=rel_t,
                mode=mode,
                vx=vx,
                speed=speed,
                pf_x=pf_x,
                pf_y=pf_y,
                pf_yaw=pf_yaw,
                ey=ey,
                contact=contact,
                theta_err=theta_err,
                throttle=throttle,
                steer=steer,
                drive_speed=drive_speed,
                drive_steer=drive_steer,
                ack_speed=ack_speed,
                ack_steer=ack_steer,
                teleop_speed=teleop_speed,
                teleop_fresh=teleop_fresh,
                auton_button=auton_button,
                slip_max=slip_max,
                opp_present=opp_present,
                obs_min=obs_min,
                obs_max=obs_max,
            )
        )

    duration = samples[-1].t if samples else 0.0
    print(f"bag={bag_path}")
    print(f"duration={duration:.1f}s  action_samples={len(samples)}")
    print()

    # Global throttle / mux summary
    neg_th = sum(1 for s in samples if s.throttle < -0.05)
    pos_th = sum(1 for s in samples if s.throttle > 0.05)
    zero_drive = sum(1 for s in samples if s.drive_speed < 0.05)
    mux_block = sum(
        1 for s in samples
        if s.drive_speed > 0.5 and s.ack_speed < 0.05 and s.teleop_fresh
    )
    brake_cmd = sum(1 for s in samples if s.throttle < -0.5 and s.drive_speed < 0.05)

    print("=== ROOT CAUSE SUMMARY ===")
    print(f"  policy throttle < -0.05 (brake): {neg_th}/{len(samples)} ({100*neg_th/len(samples):.0f}%)")
    print(f"  policy throttle > +0.05 (accel):  {pos_th}/{len(samples)} ({100*pos_th/len(samples):.0f}%)")
    print(f"  /drive speed ~0:                  {zero_drive}/{len(samples)}")
    print(f"  RL /drive>0.5 but mux ack=0 (teleop override): {mux_block}/{len(samples)}")
    print(f"  brake action -> zero /drive speed: {brake_cmd}/{len(samples)}")
    if neg_th > len(samples) * 0.5:
        te = [s.theta_err for s in samples if math.isfinite(s.theta_err)]
        ey = [s.ey for s in samples if math.isfinite(s.ey)]
        print(f"  -> Policy mostly BRAKING. obs[9] heading_err: mean={statistics.fmean(te):.2f} rad "
              f"({math.degrees(statistics.fmean(te)):.0f}°)  max|err|={max(abs(x) for x in te):.2f} rad")
        print(f"     obs[10] ey: mean={statistics.fmean(ey):+.2f}m  max|ey|={max(abs(x) for x in ey):.2f}m")
        opp_frac = statistics.fmean(s.opp_present for s in samples)
        print(f"     opponent present flag mean={opp_frac:.2f}")
    elif mux_block > 0:
        print("  -> Mux teleop likely overriding RL /drive (hold R1 not L1).")
    print()

    for mode in ("manual", "auton", "moving", "idle"):
        grp = [s for s in samples if s.mode == mode]
        if not grp:
            continue
        print(f"=== {mode.upper()} ({len(grp)} samples, t={grp[0].t:.1f}-{grp[-1].t:.1f}s) ===")
        throttles = [s.throttle for s in grp]
        print(
            f"  throttle: mean={statistics.fmean(throttles):+.3f}  "
            f"min={min(throttles):+.3f}  max={max(throttles):+.3f}"
        )
        ds = [s.drive_speed for s in grp]
        ack = [s.ack_speed for s in grp]
        spd = [s.speed for s in grp]
        print(
            f"  /drive speed: mean={statistics.fmean(ds):.2f}  "
            f"/ackermann speed: mean={statistics.fmean(ack):.2f}  "
            f"odom speed: mean={statistics.fmean(spd):.2f}"
        )
        te = [s.theta_err for s in grp if math.isfinite(s.theta_err)]
        eys = [s.ey for s in grp if math.isfinite(s.ey)]
        if te:
            print(
                f"  obs[9] heading_err: mean={statistics.fmean(te):+.2f} rad "
                f"({math.degrees(statistics.fmean(te)):+.0f}°)  "
                f"max|err|={max(abs(x) for x in te):.2f} rad"
            )
        if eys:
            print(f"  obs[10] ey: mean={statistics.fmean(eys):+.2f}m  max|ey|={max(abs(x) for x in eys):.2f}m")
        auton_btn = sum(1 for s in grp if s.auton_button)
        teleop_on = sum(1 for s in grp if s.teleop_fresh)
        print(f"  R1 held: {auton_btn}/{len(grp)}  teleop fresh: {teleop_on}/{len(grp)}")
        print()

    auton = [s for s in samples if s.mode == "auton"]
    if auton:
        print("=== AUTON timeline (every ~1.5s) ===")
        last = -999.0
        for s in auton:
            if s.t - last < 1.5:
                continue
            last = s.t
            print(
                f"  t={s.t:5.1f}s  spd={s.speed:.2f}  th={s.throttle:+.2f}  "
                f"drv={s.drive_speed:.2f}  ack={s.ack_speed:.2f}  "
                f"hdg={s.theta_err:+.2f}  ey={s.ey:+.2f}  "
                f"R1={int(s.auton_button)}  teleop={s.teleop_speed:.2f}  opp={s.opp_present:.0f}"
            )

    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bag", help="rosbag2 directory")
    args = p.parse_args()
    return analyze(args.bag)


if __name__ == "__main__":
    raise SystemExit(main())
