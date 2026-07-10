"""Parse an on-car rosbag2 into the canonical 10 Hz profile CSV.

Reads ``/rl/action`` (the master clock, published at the control rate by
``profile_maneuver_node``), ``/odom`` (body-frame twist), ``/pf/pose/odom``
(map-frame pose), ``/drive`` (commanded Ackermann), and the ``/calib/maneuver``
and ``/calib/role`` labels. Everything is resampled onto the action timeline with
zero-order hold so the output schema matches the Genesis profiler exactly.

ROS 2 (``rosbag2_py`` + message packages) is only imported inside the reader, so
this module imports fine on a workstation without ROS; the parsing step itself
must run where ROS is available (typically on or near the car).
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import run_dir  # noqa: E402
from .maneuvers import load_schedule, load_settings  # noqa: E402
from .metrics import summarize  # noqa: E402
from .schema import ProfileTable  # noqa: E402

_TOPIC_TYPES = {
    "/rl/action": "std_msgs/msg/Float32MultiArray",
    "/odom": "nav_msgs/msg/Odometry",
    "/pf/pose/odom": "nav_msgs/msg/Odometry",
    "/drive": "ackermann_msgs/msg/AckermannDriveStamped",
    "/calib/maneuver": "std_msgs/msg/String",
    "/calib/role": "std_msgs/msg/String",
}


def _read_bag(
    bag_path: str, extra_topics: tuple[str, ...] = ()
) -> dict[str, list[tuple[float, object]]]:
    """Return ``{topic: [(t_sec, msg), ...]}`` for the topics we care about."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id = "mcap" if _looks_like_mcap(bag_path) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_classes = {name: get_message(type_map[name]) for name in type_map}

    wanted = set(_TOPIC_TYPES) | set(extra_topics)
    out: dict[str, list[tuple[float, object]]] = {t: [] for t in wanted}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if topic not in out:
            continue
        msg = deserialize_message(raw, msg_classes[topic])
        out[topic].append((t_ns * 1e-9, msg))
    return out


def _looks_like_mcap(bag_path: str) -> bool:
    if os.path.isdir(bag_path):
        return any(f.endswith(".mcap") for f in os.listdir(bag_path))
    return bag_path.endswith(".mcap")


def _zoh(series: list[tuple[float, object]], t: float):
    """Zero-order hold: last sample at or before ``t`` (else first)."""
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


def parse_bag(bag_path: str, twist_in_world_frame: bool = False) -> ProfileTable:
    topics = _read_bag(bag_path)
    actions = topics["/rl/action"]
    if not actions:
        raise ValueError("bag has no /rl/action messages (was the profiler running?)")

    rows: list[dict] = []
    cur_man = None
    man_t0 = None
    prev_vx = prev_vy = prev_t = None
    for t, act in actions:
        man_msg = _zoh(topics["/calib/maneuver"], t)
        role_msg = _zoh(topics["/calib/role"], t)
        man = man_msg.data if man_msg is not None else "UNKNOWN"
        role = role_msg.data if role_msg is not None else "measure"
        if man in ("DONE", "ABORT"):
            continue

        odom = _zoh(topics["/odom"], t)
        pose = _zoh(topics["/pf/pose/odom"], t)
        drive = _zoh(topics["/drive"], t)

        if man != cur_man:
            cur_man = man
            man_t0 = t
            prev_vx = prev_vy = prev_t = None
        man_t = t - man_t0

        vx = vy = omega_z = math.nan
        if odom is not None:
            vx = float(odom.twist.twist.linear.x)
            vy = float(odom.twist.twist.linear.y)
            omega_z = float(odom.twist.twist.angular.z)
            if twist_in_world_frame and pose is not None:
                yaw = _yaw_from_quat(pose.pose.pose.orientation)
                c, s = math.cos(-yaw), math.sin(-yaw)
                vx, vy = c * vx - s * vy, s * vx + c * vy

        x = y = yaw = math.nan
        if pose is not None:
            x = float(pose.pose.pose.position.x)
            y = float(pose.pose.pose.position.y)
            yaw = _yaw_from_quat(pose.pose.pose.orientation)

        speed = math.hypot(vx, vy) if math.isfinite(vx) and math.isfinite(vy) else math.nan
        ax = ay = math.nan
        if prev_t is not None and (t - prev_t) > 1e-6:
            ax = (vx - prev_vx) / (t - prev_t)
            ay = (vy - prev_vy) / (t - prev_t)
        prev_vx, prev_vy, prev_t = vx, vy, t

        throttle = float(act.data[0]) if len(act.data) >= 1 else math.nan
        steer = float(act.data[1]) if len(act.data) >= 2 else math.nan

        rows.append(
            {
                "t": man_t,
                "maneuver": man,
                "role": role,
                "throttle": throttle,
                "steer": steer,
                "x": x,
                "y": y,
                "yaw": yaw,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "ax": ax,
                "ay": ay,
                "omega_z": omega_z,
                "steer_state": math.nan,  # not observable on the car
                "source": "irl",
            }
        )
    return ProfileTable.from_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rosbag2 -> calibration profile CSV")
    parser.add_argument("bag", help="path to the rosbag2 directory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--twist-in-world-frame",
        action="store_true",
        help="set if /odom twist is map-frame (matches vehicle_obs config).",
    )
    args = parser.parse_args(argv)

    table = parse_bag(args.bag, twist_in_world_frame=args.twist_in_world_frame)

    out = run_dir(args.run_id, create=True)
    csv_path = os.path.join(out, "profile_irl.csv")
    table.to_csv(csv_path)

    schedule = load_schedule()
    settings = load_settings()
    summary = summarize(table, schedule, float(settings["v_fit_min"]))
    summary_path = os.path.join(out, "summary_irl.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[parse_bag] wrote {csv_path} ({len(table)} rows)")
    print(f"[parse_bag] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
