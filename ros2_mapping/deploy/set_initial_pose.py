#!/usr/bin/env python3
"""Publish /initialpose for the particle filter at a centerline point (map frame)."""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


def yaw_from_centerline(csv_path: str, index: int = 0) -> tuple[float, float, float]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, comments="#")
    x0, y0 = float(data["x_m"][index]), float(data["y_m"][index])
    i1 = (index + 1) % len(data)
    dx = float(data["x_m"][i1]) - x0
    dy = float(data["y_m"][i1]) - y0
    return x0, y0, math.atan2(dy, dx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/shereef/maps/f1tenth_map_centerline.csv")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frame", default="map")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    x, y, yaw = yaw_from_centerline(args.csv, args.index)
    half = 0.5 * yaw
    qw, qz = math.cos(half), math.sin(half)

    rclpy.init()
    node = Node("set_initial_pose")
    pub = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = args.frame
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.w = qw
    msg.pose.pose.orientation.z = qz
    msg.pose.covariance[0] = 0.25
    msg.pose.covariance[7] = 0.25
    msg.pose.covariance[35] = 0.07

    for _ in range(args.repeat):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        node.get_logger().info(
            f"initialpose map=({x:.3f},{y:.3f}) yaw={yaw:.3f} rad idx={args.index}"
        )
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
