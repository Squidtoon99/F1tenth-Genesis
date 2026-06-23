#!/usr/bin/env python3
"""Compare /rl/observation values to raw /odom and /pf/pose/odom sources."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def quat_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_centerline(csv_path: str):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, comments="#")
    xs = data["x_m"].astype(float)
    ys = data["y_m"].astype(float)
    wl = data["w_tr_left_m"].astype(float)
    wr = data["w_tr_right_m"].astype(float)
    return xs, ys, wl, wr


def frenet(xs, ys, wl, wr, px, py):
    best = None
    for i in range(len(xs) - 1):
        x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / max(l2, 1e-12)))
        cx, cy = x1 + t * dx, y1 + t * dy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if best is None or d2 < best[0]:
            ty = math.atan2(dy, dx)
            nx, ny = -math.sin(ty), math.cos(ty)
            ey = (px - cx) * nx + (py - cy) * ny
            best = (math.sqrt(d2), i, ey, ty, cx, cy, wl[i], wr[i])
    return best


class CompareNode(Node):
    def __init__(self) -> None:
        super().__init__("compare_obs_sources")
        self.pf: Odometry | None = None
        self.odom: Odometry | None = None
        self.obs: list[float] | None = None
        self.create_subscription(Odometry, "/pf/pose/odom", lambda m: setattr(self, "pf", m), 10)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10)
        self.create_subscription(
            Float32MultiArray, "/rl/observation", lambda m: setattr(self, "obs", list(m.data)), 10
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/shereef/maps/f1tenth_map_centerline.csv")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    xs, ys, wl, wr = load_centerline(args.csv)
    rclpy.init()
    node = CompareNode()

    rows: list[dict] = []
    t0 = time.time()
    while len(rows) < args.samples and time.time() - t0 < 8:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.pf is None or node.odom is None or node.obs is None:
            continue
        if len(node.obs) < 12:
            continue

        o = node.obs
        pf, od = node.pf, node.odom
        px, py = pf.pose.pose.position.x, pf.pose.pose.position.y
        pf_yaw = quat_yaw(pf.pose.pose.orientation)
        od_yaw = quat_yaw(od.pose.pose.orientation)
        odom_vx = od.twist.twist.linear.x
        odom_vy = od.twist.twist.linear.y
        odom_wz = od.twist.twist.angular.z

        dist, seg, ey_pf, track_yaw, cx, cy, w_l, w_r = frenet(xs, ys, wl, wr, px, py)
        theta_pf = math.atan2(
            math.sin(pf_yaw - track_yaw), math.cos(pf_yaw - track_yaw)
        )

        # If odom yaw were expressed in map frame via PF-implied offset (same base_link):
        map_odom_offset = pf_yaw - od_yaw
        od_yaw_as_map = od_yaw + map_odom_offset  # == pf_yaw by construction
        theta_odom_via_pf = math.atan2(
            math.sin(od_yaw_as_map - track_yaw), math.cos(od_yaw_as_map - track_yaw)
        )

        # If user trusts odom heading: estimate map yaw = odom_yaw + offset where offset
        # makes heading match track (what odom would imply if facing track):
        offset_to_track = track_yaw - od_yaw
        odom_yaw_faces_track = od_yaw + offset_to_track
        theta_odom_faces_track = 0.0

        rows.append(
            {
                "obs0": o[0],
                "obs1": o[1],
                "obs2": o[2],
                "obs9": o[9],
                "obs10": o[10],
                "obs11": o[11],
                "odom_vx": odom_vx,
                "odom_vy": odom_vy,
                "odom_wz": odom_wz,
                "pf_yaw_deg": math.degrees(pf_yaw),
                "odom_yaw_deg": math.degrees(od_yaw),
                "track_yaw_deg": math.degrees(track_yaw),
                "recomputed9_pf": theta_pf,
                "recomputed10": ey_pf,
                "map_odom_offset_deg": math.degrees(map_odom_offset),
                "offset_to_track_deg": math.degrees(offset_to_track),
            }
        )
        node.obs = None
        time.sleep(0.15)

    rclpy.shutdown()
    if not rows:
        print("ERROR: no synchronized samples", file=sys.stderr)
        return 1

    r = rows[-1]
    print("=== OBS vs SOURCE COMPARISON (vehicle_obs wiring) ===")
    print("Pose xy,yaw -> /pf/pose/odom (map)   |   Twist vx,vy,wz -> /odom (body)\n")
    print(f"{'field':<22} {'obs':>10} {'raw source':>12} {'match?':>8}")
    print("-" * 56)
    for label, obs_k, raw, tol in [
        ("obs[0] vx", "obs0", "odom_vx", 0.02),
        ("obs[1] vy", "obs1", "odom_vy", 0.02),
        ("obs[2] wz", "obs2", "odom_wz", 0.05),
    ]:
        ok = abs(r[obs_k] - r[raw]) < tol
        print(f"{label:<22} {r[obs_k]:10.4f} {r[raw]:12.4f} {'OK' if ok else 'DIFF':>8}")

    print()
    print(f"{'field':<22} {'obs':>10} {'recomputed':>12} {'match?':>8}")
    print("-" * 56)
    ok9 = abs(r["obs9"] - r["recomputed9_pf"]) < 0.02
    ok10 = abs(r["obs10"] - r["recomputed10"]) < 0.05
    print(
        f"{'obs[9] heading err':<22} {math.degrees(r['obs9']):10.1f}° "
        f"{math.degrees(r['recomputed9_pf']):10.1f}° {'OK' if ok9 else 'DIFF':>8}"
    )
    print(
        f"{'obs[10] ey':<22} {r['obs10']:10.3f}m {r['recomputed10']:10.3f}m {'OK' if ok10 else 'DIFF':>8}"
    )

    print("\n=== HEADING: PF vs ODOM vs TRACK ===")
    print(f"  track tangent (map):     {r['track_yaw_deg']:+.1f}°")
    print(f"  PF yaw (map):            {r['pf_yaw_deg']:+.1f}°  -> obs[9] = {math.degrees(r['obs9']):+.1f}°")
    print(f"  odom yaw (odom frame):   {r['odom_yaw_deg']:+.1f}°")
    print(f"  PF-odom yaw delta:       {r['map_odom_offset_deg']:+.1f}°  (frame rotation, not used in obs[9])")
    print(
        f"  offset so odom->track:   {r['offset_to_track_deg']:+.1f}°  "
        f"(if odom heading is truth, map yaw should be odom+this)"
    )
    print(
        f"  -> odom-implied map yaw facing track would give obs[9] ~ 0°, "
        f"but actual obs[9] = {math.degrees(r['obs9']):+.1f}° because obs[9] uses PF yaw only"
    )

    print("\n=== TAKEAWAY ===")
    print("  obs[0:2] come from /odom twist — should match odom (you said odom is right).")
    print("  obs[9:10] come from /pf/pose/odom xy+yaw — odom yaw is NOT in obs[9].")
    if abs(r["obs9"]) > 0.5:
        print(
            f"  PF yaw ({r['pf_yaw_deg']:+.0f}°) vs track ({r['track_yaw_deg']:+.0f}°) "
            f"drives obs[9]={math.degrees(r['obs9']):+.0f}° regardless of odom."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
