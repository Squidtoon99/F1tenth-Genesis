#!/usr/bin/env python3
"""Animate autonomous DFS frontier exploration mapping a fresh track (offline, no ROS).

Unlike ``animate_mapping.py`` (which drives the known centerline), this composes the
*actual* mapping stack from ``f1tenth_mapping``:

- ``mapping_math.cluster_frontiers`` + ``DfsGoalStack`` choose the next frontier goal
  (depth-first: nearest cluster popped first),
- ``mapping_math.astar`` plans through currently-known free space (inflated),
- ``pure_pursuit.compute_steering`` + ``SpeedPID`` track the path,
- a vectorized 2D LiDAR raycasts the "truth" map each step to grow the occupancy grid.

The car never sees the centerline; it explores from scratch. Outputs an exploration
GIF and the final generated occupancy map (the thing SLAM would have produced).

Example:
    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=.:../f1tenth_mapping python animate_dfs_exploration.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "ros2_mapping" / "f1tenth_mapping"))

from centerline_extractor import load_map_yaml  # noqa: E402  (postprocess)
from track_reference import load_reference_track  # noqa: E402

from f1tenth_mapping.mapping_math import (  # noqa: E402
    MapMeta,
    astar,
    cluster_frontiers,
    coverage_ratio,
    free_bounding_box,
)
# ROS occupancy convention used by mapping_math.
UNKNOWN, FREE, OCC = -1, 0, 100
# centerline_extractor MapData.grid convention.
TRUTH_OCC = 1


def _order_frontiers(frontiers, pose, yaw, min_goal_dist, fwd_cone=-0.25):
    """Order frontier centroids so the car commits to the FORWARD direction and maps the
    loop in a single lap instead of ping-ponging between the two open ends of the corridor.

    A wall-bounded circuit is one continuous corridor. If we always chase the nearest
    frontier, the trailing open end (behind us, where we came from) keeps winning once the
    leading edge recedes, so the car repeatedly turns around. Instead we strictly prefer
    frontiers ahead of the current heading and only fall back to ones behind when nothing
    ahead is reachable (a genuine dead end, or the lap has closed in front). A* still plans
    only through known-free space, so "forward" can never cut across an unmapped wall.

    Priority is "real motion" first, direction second, so the car always commits to a
    goal far enough to build speed (>= min_goal_dist) instead of nibbling sub-tolerance
    specks every tick (which freezes it in place):
      1. forward & far   - leading edge of the corridor (the normal case),
      2. behind & far    - only reached when nothing far is open ahead (true dead end /
                           lap closed in front), so we don't ping-pong on every step,
      3. forward near    - small specks ahead (rounding hairpins),
      4. behind near     - last resort.
    A* still plans only through known-free space, so a "forward" pick can never cut
    across an unmapped wall."""
    hx, hy = math.cos(yaw), math.sin(yaw)
    fwd_far, beh_far, fwd_near, beh_near = [], [], [], []
    for c in frontiers:
        v = c.centroid_xy - pose
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            continue
        forward = (v[0] * hx + v[1] * hy) / d  # +1 dead ahead, -1 behind
        if d >= min_goal_dist:
            (fwd_far if forward > fwd_cone else beh_far).append((d, c.centroid_xy))
        else:
            (fwd_near if forward > fwd_cone else beh_near).append((d, c.centroid_xy))
    for bucket in (fwd_far, beh_far, fwd_near, beh_near):
        bucket.sort(key=lambda t: t[0])
    return [xy for _, xy in fwd_far + beh_far + fwd_near + beh_near]


def frontiers_in_roi(known, meta, roi, min_cluster_size, pad=6):
    """Run cluster_frontiers on just the explored ROI (cropped) for speed; centroids
    come back in world coords so the result is identical to clustering the full grid."""
    if roi is None:
        return cluster_frontiers(known, meta, min_cluster_size=min_cluster_size)
    r0, r1, c0, c1 = roi
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(meta.height - 1, r1 + pad)
    c1 = min(meta.width - 1, c1 + pad)
    sub = known[r0 : r1 + 1, c0 : c1 + 1]
    submeta = MapMeta(
        width=sub.shape[1], height=sub.shape[0], resolution=meta.resolution,
        origin_x=meta.origin_x + c0 * meta.resolution,
        origin_y=meta.origin_y + r0 * meta.resolution,
    )
    return cluster_frontiers(sub, submeta, min_cluster_size=min_cluster_size)


def coarsen(grid: np.ndarray, f: int) -> np.ndarray:
    """Block-reduce a {0 free, 1 occ, 2 unknown} truth grid by factor ``f`` (occ wins,
    then free). Speeds up the Python frontier clustering / raycast without changing
    track geometry."""
    if f <= 1:
        return grid
    H, W = grid.shape
    H2, W2 = H // f, W // f
    g = grid[: H2 * f, : W2 * f].reshape(H2, f, W2, f)
    occ = (g == 1).any(axis=(1, 3))
    free = (g == 0).any(axis=(1, 3))
    out = np.full((H2, W2), 2, dtype=np.uint8)
    out[free] = 0
    out[occ] = 1
    return out


def vectorized_scan(truth_occ, known, meta, px, py, yaw, angles, distances, free_radius=1):
    """Raycast all LiDAR beams at once and stamp free/occupied into ``known``.
    Returns ray endpoints (K, 2) for visualization."""
    H, W = known.shape
    res, ox, oy = meta.resolution, meta.origin_x, meta.origin_y
    ca = np.cos(yaw + angles)
    sa = np.sin(yaw + angles)
    xs = px + np.outer(ca, distances)
    ys = py + np.outer(sa, distances)
    cols = np.floor((xs - ox) / res).astype(np.int64)
    rows = np.floor((ys - oy) / res).astype(np.int64)
    inb = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rc = np.clip(rows, 0, H - 1)
    cc = np.clip(cols, 0, W - 1)

    M = distances.shape[0]
    occ = (truth_occ[rc, cc]) & inb
    oob = ~inb
    occ_any = occ.any(axis=1)
    oob_any = oob.any(axis=1)
    first_occ = np.where(occ_any, occ.argmax(axis=1), M)
    first_oob = np.where(oob_any, oob.argmax(axis=1), M)
    first_stop = np.minimum(first_occ, first_oob)

    step_idx = np.arange(M)[None, :]
    before = (step_idx < first_stop[:, None]) & inb
    fr = rc[before]
    fc = cc[before]
    # Stamp free as a small disk around each beam sample so the swept corridor is a
    # solid connected free region (otherwise angular gaps between beams at range leave
    # unknown speckles that break A* connectivity).
    for dr in range(-free_radius, free_radius + 1):
        for dc in range(-free_radius, free_radius + 1):
            rr = np.clip(fr + dr, 0, H - 1)
            kc = np.clip(fc + dc, 0, W - 1)
            known[rr, kc] = np.where(known[rr, kc] == OCC, OCC, FREE)

    hit = occ_any & (first_occ <= first_oob)
    if hit.any():
        krange = np.arange(angles.shape[0])
        hr = rc[krange[hit], first_occ[hit]]
        hc = cc[krange[hit], first_occ[hit]]
        # Single-cell walls: thickening occupied cells would seal narrow corridors and
        # trap the car in an enclosed free pocket with no reachable frontier.
        known[hr, hc] = OCC

    # Ray endpoints (for the LiDAR fan overlay) and per-beam free range (for follow-the-gap).
    end_d = np.where(first_stop < M, distances[np.clip(first_stop, 0, M - 1)], distances[-1])
    ex = px + ca * end_d
    ey = py + sa * end_d
    return np.stack([ex, ey], axis=1), end_d


def _setup_world(truth_yaml: Path, reference_csv: Path, cfg):
    """Shared setup for both exploration modes: load + orient the truth grid, thicken
    walls, build map meta, blank known grid, LiDAR angle/range tables, and a valid free
    start pose/heading taken from the reference centerline (used ONLY to spawn, never to
    drive)."""
    truth = load_map_yaml(truth_yaml)
    # load_map_yaml returns the PNG in image order (row 0 = top). world_to_grid and the
    # reference centerline use ROS convention (row 0 = bottom / world y_min), so flip to
    # align the simulated track with the world/centerline frame.
    grid_truth = coarsen(np.flipud(truth.grid), cfg.downsample)
    resolution = truth.resolution * cfg.downsample
    truth_occ = grid_truth == TRUTH_OCC
    # Give walls real thickness so they are 8-connected and continuous; a 1-cell-thick
    # wall has diagonal gaps that LiDAR beams thread through, flooding "free" into the
    # exterior and spawning spurious frontiers.
    if cfg.wall_thickness > 0:
        from scipy.ndimage import binary_dilation

        truth_occ = binary_dilation(truth_occ, iterations=cfg.wall_thickness)
    # Optional fully-known truth grid for planning (debug only). Deployment-faithful
    # behavior plans A* on the discovered map (--plan-on-truth disabled by default),
    # exactly like the real navigator_node planning on slam_toolbox's /map.
    truth_ros = np.where(truth_occ, OCC, FREE).astype(np.int16)
    H, W = grid_truth.shape
    meta = MapMeta(
        width=W, height=H, resolution=resolution,
        origin_x=truth.origin_x, origin_y=truth.origin_y,
    )
    ref_cl, *_ = load_reference_track(reference_csv)
    pose = ref_cl[0].astype(np.float64).copy()
    tangent = ref_cl[1] - ref_cl[0]
    yaw = float(math.atan2(tangent[1], tangent[0]))
    known = np.full((H, W), UNKNOWN, dtype=np.int16)
    angles = np.linspace(-math.pi, math.pi, cfg.n_rays, endpoint=False)
    distances = np.arange(resolution, cfg.max_range_m, resolution * cfg.ray_step_mult)
    return truth, grid_truth, truth_occ, truth_ros, meta, known, pose, yaw, angles, distances


def _follow_the_gap(angles, ranges, max_range, arc_half, bubble_m):
    """Reactive steering: pick a target angle (relative to current heading) toward the
    center of the widest deep gap in the forward arc, after carving a safety bubble around
    the closest obstacle. This follows the corridor and structurally cannot jump a wall."""
    m = np.abs(angles) <= arc_half
    a = angles[m]
    r = ranges[m].copy()
    if a.size == 0:
        return 0.0
    # Safety bubble: blank beams within an angular window of the nearest obstacle so we
    # don't steer toward a gap that clips a wall corner.
    inear = int(np.argmin(r))
    rn = max(float(r[inear]), 1e-3)
    dtheta = math.atan2(bubble_m, rn)
    r[np.abs(a - a[inear]) <= dtheta] = 0.0
    # Widest contiguous run of "deep" beams (corridor continues there).
    deep = r >= 0.6 * max_range
    if not deep.any():
        return float(a[int(np.argmax(ranges[m]))])
    best_lo, best_len, lo = 0, 0, None
    for i, f in enumerate(deep):
        if f and lo is None:
            lo = i
        if (not f or i == deep.size - 1) and lo is not None:
            hi = i + 1 if f else i
            if hi - lo > best_len:
                best_lo, best_len = lo, hi - lo
            lo = None
    return float(a[best_lo + best_len // 2])


def explore_reactive(truth_yaml: Path, reference_csv: Path, cfg) -> tuple:
    """Recon-lap exploration: drive the corridor reactively with follow-the-gap, never
    reversing, mapping with the LiDAR as we go. A wall-bounded loop is mapped in a single
    clean lap, and loop closure (return near start after a real lap) is reliable because
    the car is always physically inside the corridor (no global planning, no shortcuts)."""
    truth, grid_truth, truth_occ, truth_ros, meta, known, pose, yaw, angles, distances = \
        _setup_world(truth_yaml, reference_csv, cfg)
    H, W = known.shape
    dt = 1.0 / cfg.control_hz
    speed = 0.0
    arc_half = math.radians(cfg.fov_deg * 0.5)
    max_dyaw = cfg.cruise * dt / max(cfg.min_turn_radius, 1e-3)  # curvature-limited turn rate

    start_pose = pose.copy()
    prev_pose = pose.copy()
    traveled = 0.0
    loop_closed = False
    frames = []
    trail = [pose.copy()]
    for tick in range(cfg.max_ticks):
        endpoints, ranges = vectorized_scan(truth_occ, known, meta, pose[0], pose[1], yaw,
                                            angles, distances, free_radius=cfg.free_radius)
        cr = int((pose[1] - meta.origin_y) / meta.resolution)
        cc0 = int((pose[0] - meta.origin_x) / meta.resolution)
        if 0 <= cr < H and 0 <= cc0 < W and known[cr, cc0] != OCC:
            known[cr, cc0] = FREE

        target = _follow_the_gap(angles, ranges, cfg.max_range_m, arc_half, cfg.bubble_m)
        yaw += float(np.clip(target, -max_dyaw, max_dyaw))
        speed = min(cfg.cruise, speed + 4.0 * dt)
        pose = pose + speed * dt * np.array([math.cos(yaw), math.sin(yaw)])

        traveled += float(np.linalg.norm(pose - prev_pose))
        prev_pose = pose.copy()
        trail.append(pose.copy())
        if (not loop_closed and traveled > cfg.min_lap_m
                and float(np.linalg.norm(pose - start_pose)) < cfg.closure_radius_m):
            loop_closed = True

        if tick % cfg.frame_every == 0:
            frames.append((known.copy(), pose.copy(), yaw, endpoints, None, None, 0.0, -1))
        if loop_closed:
            if cfg.debug:
                print(f"[t{tick}] DONE (loop closed): traveled={traveled:.0f}m "
                      f"pose=({pose[0]:.1f},{pose[1]:.1f})", flush=True)
            frames.append((known.copy(), pose.copy(), yaw, endpoints, None, None, 0.0, -1))
            break

    return truth, meta, known, frames, 0, truth_occ, grid_truth, np.asarray(trail)


def explore(truth_yaml: Path, reference_csv: Path, cfg) -> tuple:
    truth, grid_truth, truth_occ, truth_ros, meta, known, pose, yaw, angles, distances = \
        _setup_world(truth_yaml, reference_csv, cfg)
    H, W = known.shape
    speed = 0.0

    active_goal = None
    path = None
    path_i = 0
    dt = 1.0 / cfg.control_hz
    speed = 0.0
    idle_ticks = 0
    start_pose = pose.copy()
    prev_pose = pose.copy()
    traveled = 0.0
    loop_closed = False

    frames = []
    trail = [pose.copy()]  # full traveled path (every tick), for the static-map overlay
    goals_reached = 0
    for tick in range(cfg.max_ticks):
        endpoints, _ = vectorized_scan(truth_occ, known, meta, pose[0], pose[1], yaw, angles,
                                        distances, free_radius=cfg.free_radius)
        # Car footprint is free.
        cr = int((pose[1] - meta.origin_y) / meta.resolution)
        cc0 = int((pose[0] - meta.origin_x) / meta.resolution)
        if 0 <= cr < H and 0 <= cc0 < W and known[cr, cc0] != OCC:
            known[cr, cc0] = FREE

        free_mask = (known >= 0) & (known <= 99)
        roi = free_bounding_box(free_mask)
        cov = coverage_ratio(known, roi) if roi is not None else 0.0

        # Pick a new frontier goal when idle. The track is a wall-bounded loop, so we
        # follow the open corridor FORWARD (prefer the reachable frontier furthest
        # ahead of the current heading) rather than nearest-first, which would oscillate
        # on a fine grid. Exploration is complete only once the lap closes and the fully
        # enclosed corridor leaves no frontiers.
        if active_goal is None:
            frontiers = frontiers_in_roi(known, meta, roi, cfg.min_cluster_size)
            frontier_count = len(frontiers)
            if not frontiers:
                if tick > 5:
                    if cfg.debug:
                        status = "loop closed" if loop_closed else "no frontiers"
                        print(f"[t{tick}] DONE ({status}): traveled={traveled:.0f}m "
                              f"pose=({pose[0]:.1f},{pose[1]:.1f}) goals={goals_reached}", flush=True)
                    frames.append((known.copy(), pose.copy(), yaw, endpoints, None, path, cov, 0))
                    break
            else:
                cand = _order_frontiers(frontiers, pose, yaw, cfg.min_goal_dist)
                tries = 0
                astar_fail = 0
                plan_grid = truth_ros if cfg.plan_on_truth else known
                for goal_xy in cand[: cfg.max_goal_tries]:
                    p = astar(plan_grid, meta, pose, goal_xy, inflation_cells=cfg.inflation)
                    if p is not None and p.shape[0] >= 2:
                        active_goal, path, path_i = goal_xy, p, 0
                        pose = p[0].astype(np.float64).copy()
                        break
                    astar_fail += 1
                    tries += 1
                if cfg.debug and active_goal is None:
                    print(f"[t{tick}] idle: frontiers={frontier_count} "
                          f"astar_fail={astar_fail} pose=({pose[0]:.1f},{pose[1]:.1f})", flush=True)
            # Loop has closed but a few unreachable speckle frontiers remain: finish.
            if loop_closed:
                if cfg.debug:
                    print(f"[t{tick}] DONE (loop closed): traveled={traveled:.0f}m "
                          f"goals={goals_reached}", flush=True)
                frames.append((known.copy(), pose.copy(), yaw, endpoints, None, path, cov, frontier_count))
                break
            # Genuinely stuck (all frontiers unreachable, lap not yet closed).
            if active_goal is None and frontier_count > 0:
                idle_ticks += 1
                if idle_ticks > cfg.idle_stop:
                    frames.append((known.copy(), pose.copy(), yaw, endpoints, None, path, cov, frontier_count))
                    break
        else:
            frontier_count = -1
            idle_ticks = 0

        # Follow the collision-free A* plan (perfect path tracking). The plan is
        # produced by the real astar() through currently-known free space toward the
        # DFS-selected frontier, so this trajectory IS the actual exploration path.
        if active_goal is not None and path is not None:
            speed = min(cfg.cruise, speed + 4.0 * dt)
            step_len = speed * dt
            while step_len > 1e-6 and path_i < path.shape[0] - 1:
                seg = path[path_i + 1] - pose
                seglen = float(np.linalg.norm(seg))
                if seglen < 1e-9:
                    path_i += 1
                    continue
                yaw = math.atan2(float(seg[1]), float(seg[0]))
                if seglen <= step_len:
                    pose = path[path_i + 1].astype(np.float64).copy()
                    path_i += 1
                    step_len -= seglen
                else:
                    pose = pose + seg / seglen * step_len
                    step_len = 0.0
            if float(np.linalg.norm(active_goal - pose)) <= cfg.goal_tol or path_i >= path.shape[0] - 1:
                active_goal, path, goals_reached = None, None, goals_reached + 1
                speed = 0.0

        # Track distance traveled and detect loop closure (returned near start after a
        # substantial lap), which is the true completion condition for a closed circuit.
        traveled += float(np.linalg.norm(pose - prev_pose))
        prev_pose = pose.copy()
        trail.append(pose.copy())
        if (not loop_closed and traveled > cfg.min_lap_m
                and float(np.linalg.norm(pose - start_pose)) < cfg.closure_radius_m):
            loop_closed = True

        if tick % cfg.frame_every == 0:
            frames.append((known.copy(), pose.copy(), yaw, endpoints,
                           None if active_goal is None else active_goal.copy(),
                           path, cov, frontier_count))

    return truth, meta, known, frames, goals_reached, truth_occ, grid_truth, np.asarray(trail)


def to_display(grid):
    disp = np.full(grid.shape, 0.5, dtype=np.float32)
    disp[(grid >= 0) & (grid <= 99)] = 1.0
    disp[grid >= OCC] = 0.0
    return disp


def render_gif(truth, meta, frames, out_path: Path, fps: int, label: str = "Exploration") -> None:
    res, H, W = meta.resolution, meta.height, meta.width
    extent = [meta.origin_x, meta.origin_x + W * res, meta.origin_y, meta.origin_y + H * res]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(to_display(frames[0][0]), cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    (trail_line,) = ax.plot([], [], color="#1f77b4", lw=1.4, label="car path")
    (path_line,) = ax.plot([], [], color="#17becf", lw=1.2, ls="--", label="A* plan")
    (rays,) = ax.plot([], [], color="#ff7f0e", lw=0.3, alpha=0.45)
    (goal_dot,) = ax.plot([], [], "*", color="#9467bd", ms=13, label="frontier goal")
    (car_dot,) = ax.plot([], [], "o", color="#d62728", ms=8, label="car")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    tx, ty = [], []

    cell_area = res * res

    def update(i):
        grid, pose, yaw, endpoints, goal, path, cov, fc = frames[i]
        im.set_data(to_display(grid))
        mapped_area = float(((grid >= 0) & (grid <= 99)).sum()) * cell_area
        tx.append(float(pose[0]))
        ty.append(float(pose[1]))
        trail_line.set_data(tx, ty)
        car_dot.set_data([pose[0]], [pose[1]])
        if path is not None:
            path_line.set_data(path[:, 0], path[:, 1])
        else:
            path_line.set_data([], [])
        if goal is not None:
            goal_dot.set_data([goal[0]], [goal[1]])
        else:
            goal_dot.set_data([], [])
        rx, ry = [], []
        for ex, ey in endpoints[::4]:
            rx += [pose[0], ex, np.nan]
            ry += [pose[1], ey, np.nan]
        rays.set_data(rx, ry)
        fc_str = "--" if fc < 0 else str(fc)
        frontier_txt = "" if fc < 0 else f"  -  frontiers: {fc_str}"
        title.set_text(f"{label}  -  frame {i+1}/{len(frames)}  -  "
                       f"mapped {mapped_area:5.0f} m²{frontier_txt}")
        return im, trail_line, path_line, rays, goal_dot, car_dot, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_final_map(meta, known, out_path: Path, trail=None, label="exploration") -> None:
    res, H, W = meta.resolution, meta.height, meta.width
    extent = [meta.origin_x, meta.origin_x + W * res, meta.origin_y, meta.origin_y + H * res]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(to_display(known), cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    title = f"Generated occupancy map (from {label})"
    if trail is not None and len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], color="#1f77b4", lw=1.6, alpha=0.9,
                label="robot path")
        ax.plot([trail[0, 0]], [trail[0, 1]], "o", color="#2ca02c", ms=9, label="start")
        ax.plot([trail[-1, 0]], [trail[-1, 1]], "s", color="#d62728", ms=8, label="end")
        ax.legend(loc="upper right", fontsize=8)
        title += "  -  blue = path the robot drove"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def true_track_mask(meta, ref_cl, ref_wl, ref_wr) -> np.ndarray:
    """Boolean mask of the true track corridor, built by stamping the local track
    width along the centerline. Robust to any track shape (hairpins, folds), unlike an
    inner/outer-polygon annulus."""
    res, H, W = meta.resolution, meta.height, meta.width
    mask = np.zeros((H, W), dtype=bool)
    half = (np.asarray(ref_wl) + np.asarray(ref_wr)) * 0.5  # per-point half-width (m)
    for (x, y), hw in zip(ref_cl, half):
        rad = int(math.ceil(hw / res))
        cr = int((y - meta.origin_y) / res)
        cc = int((x - meta.origin_x) / res)
        r0, r1 = max(0, cr - rad), min(H, cr + rad + 1)
        c0, c1 = max(0, cc - rad), min(W, cc + rad + 1)
        rr, ccg = np.mgrid[r0:r1, c0:c1]
        mask[r0:r1, c0:c1] |= ((rr - cr) ** 2 + (ccg - cc) ** 2) <= rad * rad
    return mask


def save_overlay(meta, known, ref_cl, ref_wl, ref_wr, out_path: Path) -> float:
    """Overlay the generated map on the TRUE track corridor. Red = track the robot
    never mapped. Returns the fraction of the true track that was mapped."""
    res, H, W = meta.resolution, meta.height, meta.width
    extent = [meta.origin_x, meta.origin_x + W * res, meta.origin_y, meta.origin_y + H * res]
    track = true_track_mask(meta, ref_cl, ref_wl, ref_wr)
    disc_free = (known >= 0) & (known <= 99)
    disc_occ = known >= OCC
    missing = track & ~disc_free

    rgb = np.full((H, W, 3), 0.5, dtype=np.float32)
    rgb[track] = (0.80, 0.88, 1.0)        # pale blue: true track corridor
    rgb[disc_free] = (1.0, 1.0, 1.0)      # white: mapped free
    rgb[disc_occ] = (0.0, 0.0, 0.0)       # black: mapped walls
    rgb[missing] = (0.90, 0.10, 0.10)     # red: unmapped track

    covered = 1.0 - (missing.sum() / max(track.sum(), 1))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb, origin="lower", extent=extent)
    ax.set_title(f"Mapped vs true track  -  {covered*100:.1f}% of track mapped (red = missed)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return covered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth-yaml", type=Path, default=_REPO / "ros2_deploy/assets/Oschersleben_map.yaml")
    p.add_argument("--reference-csv", type=Path, default=_REPO / "ros2_deploy/assets/Oschersleben_centerline.csv")
    p.add_argument("--gif", type=Path, default=_REPO / "ros2_mapping/output/dfs_exploration.gif")
    p.add_argument("--map-out", type=Path, default=_REPO / "ros2_mapping/output/dfs_generated_map.png")
    p.add_argument("--overlay-out", type=Path, default=_REPO / "ros2_mapping/output/dfs_map_vs_truth.png")
    p.add_argument("--control-hz", type=float, default=15.0)
    p.add_argument("--cruise", type=float, default=2.0)
    p.add_argument("--speed-limit", type=float, default=2.5)
    p.add_argument("--goal-tol", type=float, default=0.6)
    p.add_argument("--inflation", type=int, default=1)
    p.add_argument("--free-radius", type=int, default=1)
    p.add_argument("--wall-thickness", type=int, default=1,
                   help="Dilate truth walls by this many cells so beams can't leak through gaps.")
    p.add_argument("--downsample", type=int, default=2,
                   help="Coarsen the truth grid by this factor for speed (geometry preserved).")
    p.add_argument("--n-rays", type=int, default=540)
    p.add_argument("--max-range-m", type=float, default=10.0)
    p.add_argument("--ray-step-mult", type=float, default=1.5)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--min-goal-dist", type=float, default=2.5)
    p.add_argument("--max-goal-tries", type=int, default=60)
    p.add_argument("--idle-stop", type=int, default=25)
    p.add_argument("--min-lap-m", type=float, default=120.0,
                   help="Minimum distance traveled before loop closure can be declared.")
    p.add_argument("--closure-radius-m", type=float, default=4.0,
                   help="Return within this radius of the start (after a lap) = loop closed.")
    p.add_argument("--plan-on-truth", action="store_true",
                   help="Debug only: plan A* on ground truth. Default plans on the "
                        "discovered map (deployment-faithful; no ground truth at deploy).")
    p.add_argument("--mode", choices=["reactive", "frontier"], default="reactive",
                   help="reactive: one-lap follow-the-gap recon (maps a loop in a single "
                        "pass, no shortcuts). frontier: DFS frontier + A* exploration.")
    p.add_argument("--fov-deg", type=float, default=200.0,
                   help="Reactive: forward field-of-view arc considered for gap following.")
    p.add_argument("--bubble-m", type=float, default=0.6,
                   help="Reactive: safety bubble radius carved around the nearest obstacle.")
    p.add_argument("--min-turn-radius", type=float, default=0.4,
                   help="Reactive: minimum turn radius (m); caps the per-tick heading change.")
    p.add_argument("--stuck-ticks", type=int, default=60)
    p.add_argument("--max-ticks", type=int, default=4000)
    p.add_argument("--frame-every", type=int, default=8)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--debug", action="store_true")
    cfg = p.parse_args()

    driver = explore_reactive if cfg.mode == "reactive" else explore
    label = "follow-the-gap recon lap" if cfg.mode == "reactive" else "DFS frontier exploration"
    truth, meta, known, frames, goals, truth_occ, grid_truth, trail = driver(
        cfg.truth_yaml, cfg.reference_csv, cfg)
    render_gif(truth, meta, frames, cfg.gif, cfg.fps, label=label)
    save_final_map(meta, known, cfg.map_out, trail=trail, label=label)
    ref_cl, ref_wl, ref_wr, _, _ = load_reference_track(cfg.reference_csv)
    covered = save_overlay(meta, known, ref_cl, ref_wl, ref_wr, cfg.overlay_out)
    print(f"goals reached: {goals}  frames: {len(frames)}  drivable-area mapped: {covered*100:.1f}%")
    print(f"wrote {cfg.gif}")
    print(f"wrote {cfg.map_out}")
    print(f"wrote {cfg.overlay_out}")


if __name__ == "__main__":
    main()
