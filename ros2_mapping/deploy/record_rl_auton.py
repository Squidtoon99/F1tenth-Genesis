#!/usr/bin/env python3
"""Record a short rosbag around the manual -> autonomous handoff.

Sequence:
  1. Wait until the car is moving (motor or ackermann speed above threshold).
  2. Start rosbag2 recording (teleop + RL obs/action/drive + mux + motor + joy).
  3. Wait until autonomous mode is detected (R1 deadman / stale teleop + RL drive).
  4. Stop recording 5 seconds after autonomous mode starts.

Usage on car:
  python3 ~/deploy/record_rl_auton.py
  python3 ~/deploy/record_rl_auton.py --out-dir ~/bags --auton-hold-s 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, Float32MultiArray


class DriveMonitor(Node):
    def __init__(self) -> None:
        super().__init__("record_rl_auton_monitor")
        self.motor_speed = 0.0
        self.ack_speed = 0.0
        self.teleop_speed = 0.0
        self.teleop_stamp: float | None = None
        self.drive_speed = 0.0
        self.drive_stamp: float | None = None
        self.rl_throttle = 0.0
        self.rl_stamp: float | None = None
        self.joy_buttons: list[int] = []
        self.joy_stamp: float | None = None

        self.create_subscription(Float32, "/commands/motor/speed", self._motor_cb, 10)
        self.create_subscription(AckermannDriveStamped, "/ackermann_cmd", self._ack_cb, 10)
        self.create_subscription(AckermannDriveStamped, "/teleop", self._teleop_cb, 10)
        self.create_subscription(AckermannDriveStamped, "/drive", self._drive_cb, 10)
        self.create_subscription(Float32MultiArray, "/rl/action", self._rl_cb, 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _motor_cb(self, msg: Float32) -> None:
        self.motor_speed = float(msg.data)

    def _ack_cb(self, msg: AckermannDriveStamped) -> None:
        self.ack_speed = float(msg.drive.speed)

    def _teleop_cb(self, msg: AckermannDriveStamped) -> None:
        self.teleop_speed = float(msg.drive.speed)
        self.teleop_stamp = self._now()

    def _drive_cb(self, msg: AckermannDriveStamped) -> None:
        self.drive_speed = float(msg.drive.speed)
        self.drive_stamp = self._now()

    def _rl_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 1:
            self.rl_throttle = float(msg.data[0])
            self.rl_stamp = self._now()

    def _joy_cb(self, msg: Joy) -> None:
        self.joy_buttons = list(msg.buttons)
        self.joy_stamp = self._now()

    def is_moving(self, motor_thresh: float, ack_thresh: float) -> bool:
        return (
            abs(self.motor_speed) > motor_thresh
            or abs(self.ack_speed) > ack_thresh
        )

    def is_autonomous(self, teleop_timeout_s: float) -> bool:
        now = self._now()
        # R1 (button 5) = autonomous deadman in stock joy_teleop.yaml
        if (
            self.joy_stamp is not None
            and (now - self.joy_stamp) < 0.5
            and len(self.joy_buttons) > 5
            and self.joy_buttons[5] == 1
        ):
            return True

        # Fallback: RL stack active and teleop mux input is stale/idle
        rl_fresh = self.rl_stamp is not None and (now - self.rl_stamp) < 0.5
        teleop_stale = (
            self.teleop_stamp is None
            or (now - self.teleop_stamp) > teleop_timeout_s
        )
        return rl_fresh and teleop_stale


def spin_once(node: DriveMonitor, timeout_s: float = 0.05) -> None:
    rclpy.spin_once(node, timeout_sec=timeout_s)


def wait_until_moving(
    node: DriveMonitor,
    motor_thresh: float,
    ack_thresh: float,
    stable_count: int,
    timeout_s: float,
) -> bool:
    hits = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        spin_once(node)
        if node.is_moving(motor_thresh, ack_thresh):
            hits += 1
            if hits >= stable_count:
                return True
        else:
            hits = 0
        time.sleep(0.05)
    return False


def wait_until_autonomous(
    node: DriveMonitor,
    teleop_timeout_s: float,
    timeout_s: float,
) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        spin_once(node)
        if node.is_autonomous(teleop_timeout_s):
            return True
        time.sleep(0.05)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.expanduser("~/bags"))
    parser.add_argument("--auton-hold-s", type=float, default=5.0)
    parser.add_argument("--drive-wait-s", type=float, default=300.0)
    parser.add_argument("--auton-wait-s", type=float, default=120.0)
    parser.add_argument("--motor-thresh", type=float, default=200.0)
    parser.add_argument("--ack-thresh", type=float, default=0.05)
    parser.add_argument("--teleop-timeout-s", type=float, default=0.25)
    args = parser.parse_args()

    topics = [
        "/teleop",
        "/joy",
        "/rl/observation",
        "/rl/action",
        "/drive",
        "/ackermann_cmd",
        "/commands/servo/position",
        "/pf/pose/odom",
        "/odom",
        "/rl/opponent/odom",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_uri = os.path.join(args.out_dir, f"rl_auton_{stamp}")

    rclpy.init()
    node = DriveMonitor()

    print(
        f"[1/4] Waiting for car to move "
        f"(motor>{args.motor_thresh} erpm or ack>{args.ack_thresh} m/s)...",
        flush=True,
    )
    if not wait_until_moving(
        node,
        args.motor_thresh,
        args.ack_thresh,
        stable_count=3,
        timeout_s=args.drive_wait_s,
    ):
        print("Timed out waiting for motion. Aborting.", flush=True)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f"[2/4] Car moving — starting bag: {bag_uri}", flush=True)
    bag_cmd = ["ros2", "bag", "record", "-o", bag_uri] + topics
    bag_proc = subprocess.Popen(bag_cmd)

    print(
        "[3/4] Recording... Switch to autonomous (hold R1 / button 5). "
        "Waiting for auton mode...",
        flush=True,
    )
    if not wait_until_autonomous(
        node,
        args.teleop_timeout_s,
        timeout_s=args.auton_wait_s,
    ):
        print("Timed out waiting for autonomous mode. Stopping bag.", flush=True)
        bag_proc.send_signal(signal.SIGINT)
        bag_proc.wait(timeout=15)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    auton_t = time.monotonic()
    print(
        f"[3/4] Autonomous mode detected at t=0. "
        f"Recording for {args.auton_hold_s:.1f}s more...",
        flush=True,
    )
    while time.monotonic() - auton_t < args.auton_hold_s:
        spin_once(node)
        time.sleep(0.05)

    print(f"[4/4] Stopping bag after {args.auton_hold_s:.1f}s in auton.", flush=True)
    bag_proc.send_signal(signal.SIGINT)
    try:
        bag_proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        bag_proc.kill()
        bag_proc.wait()

    print(f"Done. Bag saved to: {bag_uri}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
