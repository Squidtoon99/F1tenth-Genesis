#!/usr/bin/env python3
"""Animate a car mapping a fresh track: a simulated LiDAR drive that builds the
occupancy grid from scratch, rendered as a GIF.

This is the offline / no-ROS counterpart to watching ``mapping_sim.launch.py`` in
RViz: a virtual car follows the track centerline, a 2D LiDAR raycasts against the
"truth" map each step, and the discovered free/occupied cells accumulate into a
grid that starts entirely unknown. Reuses the raycast helpers from
``offline_mapping_e2e`` so the mapping math is identical to the validated pipeline.

Example:
    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. python animate_mapping.py \
        --out ../output/mapping_anim.gif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from centerline_extractor import load_map_yaml, occupancy_grid_to_gray
from offline_mapping_e2e import _drive_poses, _stamp_disk, _world_to_col_row
from track_reference import load_reference_track

UNKNOWN, FREE, OCC = 2, 0, 1


def simulate_frames(
    truth_yaml: Path,
    reference_csv: Path,
    laps: float,
    drive_step_m: float,
    n_rays: int,
    max_range_m: float,
    ray_step_mult: float,
    frame_every: int,
):
    """Drive the reference line, raycast a LiDAR each pose, and snapshot the growing
    grid every ``frame_every`` poses. Returns (truth, frames) where each frame is
    (grid_copy, pose, yaw, ray_endpoints)."""
    truth = load_map_yaml(truth_yaml)
    ref_cl, *_ = load_reference_track(reference_csv)
    poses, yaws = _drive_poses(ref_cl, step_m=drive_step_m)

    grid = np.full(truth.grid.shape, UNKNOWN, dtype=np.uint8)
    h, w = truth.grid.shape
    step = truth.resolution * ray_step_mult
    angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
    n_poses = int(len(poses) * laps)

    frames = []
    for idx in range(n_poses):
        pose = poses[idx % len(poses)]
        yaw = float(yaws[idx % len(poses)])
        pr, pc = _world_to_col_row(pose, truth)
        if 0 <= pr < h and 0 <= pc < w and grid[pr, pc] != OCC:
            grid[pr, pc] = FREE

        endpoints = []
        for da in angles:
            angle = yaw + float(da)
            dx, dy = math.cos(angle) * step, math.sin(angle) * step
            x, y, dist = float(pose[0]), float(pose[1]), 0.0
            while dist < max_range_m:
                x += dx
                y += dy
                dist += step
                row, col = _world_to_col_row(np.array([x, y]), truth)
                if row < 0 or col < 0 or row >= h or col >= w:
                    break
                if truth.grid[row, col] == OCC:
                    _stamp_disk(grid, row, col, OCC, 1)
                    break
                if grid[row, col] != OCC:
                    grid[row, col] = FREE
            endpoints.append((x, y))

        if idx % frame_every == 0 or idx == n_poses - 1:
            frames.append((grid.copy(), pose.copy(), yaw, np.asarray(endpoints)))
    return truth, frames


def render_gif(truth, frames, out_path: Path, ref_cl: np.ndarray, fps: int) -> None:
    res = truth.resolution
    h, w = truth.grid.shape
    extent = [
        truth.origin_x,
        truth.origin_x + w * res,
        truth.origin_y,
        truth.origin_y + h * res,
    ]

    def to_display(grid: np.ndarray) -> np.ndarray:
        disp = np.full(grid.shape, 0.5, dtype=np.float32)  # unknown -> gray
        disp[grid == FREE] = 1.0  # free -> white
        disp[grid == OCC] = 0.0  # occupied -> black
        return disp

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        to_display(frames[0][0]), cmap="gray", origin="lower",
        extent=extent, vmin=0.0, vmax=1.0,
    )
    ax.plot(ref_cl[:, 0], ref_cl[:, 1], color="#2ca02c", lw=1.0, alpha=0.35,
            label="true centerline")
    (trail_line,) = ax.plot([], [], color="#1f77b4", lw=1.5, label="car path")
    (rays,) = ax.plot([], [], color="#ff7f0e", lw=0.3, alpha=0.5)
    (car_dot,) = ax.plot([], [], "o", color="#d62728", ms=8, label="car")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    trail_x: list[float] = []
    trail_y: list[float] = []

    def update(i):
        grid, pose, yaw, endpoints = frames[i]
        im.set_data(to_display(grid))
        trail_x.append(float(pose[0]))
        trail_y.append(float(pose[1]))
        trail_line.set_data(trail_x, trail_y)
        car_dot.set_data([pose[0]], [pose[1]])
        # LiDAR fan: car -> each endpoint, NaN-separated into one polyline.
        rx, ry = [], []
        for ex, ey in endpoints[::4]:
            rx += [pose[0], ex, np.nan]
            ry += [pose[1], ey, np.nan]
        rays.set_data(rx, ry)
        mapped = float(np.mean(grid != UNKNOWN)) * 100.0
        title.set_text(f"Mapping a fresh track  -  frame {i + 1}/{len(frames)}  -  {mapped:4.1f}% explored")
        return im, trail_line, rays, car_dot, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth-yaml", type=Path, default=repo / "ros2_deploy/assets/Oschersleben_map.yaml")
    p.add_argument("--reference-csv", type=Path, default=repo / "ros2_deploy/assets/Oschersleben_centerline.csv")
    p.add_argument("--out", type=Path, default=repo / "ros2_mapping/output/mapping_anim.gif")
    p.add_argument("--laps", type=float, default=1.15)
    p.add_argument("--drive-step-m", type=float, default=0.9)
    p.add_argument("--n-rays", type=int, default=180)
    p.add_argument("--max-range-m", type=float, default=8.0)
    p.add_argument("--ray-step-mult", type=float, default=2.0)
    p.add_argument("--frame-every", type=int, default=6)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    truth, frames = simulate_frames(
        args.truth_yaml, args.reference_csv, args.laps, args.drive_step_m,
        args.n_rays, args.max_range_m, args.ray_step_mult, args.frame_every,
    )
    ref_cl, *_ = load_reference_track(args.reference_csv)
    render_gif(truth, frames, args.out, ref_cl, args.fps)
    final_explored = float(np.mean(frames[-1][0] != UNKNOWN)) * 100.0
    print(f"wrote {args.out}  ({len(frames)} frames, {final_explored:.1f}% explored)")


if __name__ == "__main__":
    main()
