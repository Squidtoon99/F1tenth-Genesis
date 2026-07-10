#!/usr/bin/env python3
"""Clean a noisy SLAM map down to a single track corridor and extract its centerline.

The gym-tuned ``centerline_extractor.extract_centerline`` rejects small real tracks
(its quality score requires >180 m loops) and is thrown off by SLAM spray / specks.
This tool:

  1. denoises occupied pixels (drops tiny connected components = lidar specks),
  2. morphologically closes small wall gaps so the corridor is fully enclosed,
  3. isolates the drivable ring (interior free space within a distance band),
  4. extracts + orders the medial-axis centerline of that ring (no length gate),
  5. ray-casts left/right widths, writes the centerline CSV and a cleaned map,
  6. renders the walls + centerline + left/right boundaries for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.morphology import binary_closing, binary_erosion, disk

import centerline_extractor as ce


def clean_grid(
    data: ce.MapData,
    min_wall_px: int,
    close_radius: int,
) -> ce.MapData:
    grid = data.grid.copy()
    occ = grid == 1

    # 1. drop tiny occupied components (specks, the small ring artifact, spray dots).
    labeled, n = ndimage.label(occ)
    if n:
        sizes = ndimage.sum(occ, labeled, range(1, n + 1))
        keep = {lab for lab, sz in enumerate(sizes, start=1) if sz >= min_wall_px}
        occ = np.isin(labeled, list(keep)) if keep else np.zeros_like(occ)

    # 2. close small wall gaps so the corridor is bounded.
    if close_radius > 0:
        occ = binary_closing(occ, disk(close_radius))

    new = grid.copy()
    new[(grid == 1) & ~occ] = 2  # demoted occupied -> unknown
    new[occ] = 1
    data.grid = new
    return data


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


def _resample_loop(pts: np.ndarray, spacing_m: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    s = np.arange(0.0, total, spacing_m)
    x = np.interp(s, arc, closed[:, 0])
    y = np.interp(s, arc, closed[:, 1])
    return np.stack([x, y], axis=1)


def _smooth_loop(pts: np.ndarray, window: int) -> np.ndarray:
    if len(pts) < window:
        return pts
    window = window if window % 2 == 1 else window + 1
    pad = window // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    sx = savgol_filter(ext[:, 0], window, 3)
    sy = savgol_filter(ext[:, 1], window, 3)
    return np.stack([sx[pad : pad + len(pts)], sy[pad : pad + len(pts)]], axis=1)


def _tangents_normals(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(pts)
    tang = np.stack([pts[(np.arange(n) + 1) % n] - pts[np.arange(n) - 1]])[0]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    return tang, normal


def _raycast(
    x0: float, y0: float, nx: float, ny: float, occupied: np.ndarray,
    data: ce.MapData, cap_m: float,
) -> float:
    step = data.resolution * 0.5
    h, w = occupied.shape
    n = int(cap_m / step)
    for i in range(1, n + 1):
        x, y = x0 + nx * step * i, y0 + ny * step * i
        row, col = ce.world_to_grid(x, y, data)
        if not (0 <= row < h and 0 <= col < w):
            return step * (i - 1)
        if occupied[row, col]:
            return step * (i - 1)
    return cap_m


def centerline_from_outer_offset(
    data: ce.MapData,
    offset_m: float,
    max_half_m: float,
    snap_iters: int,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centerline = clean outer-wall loop offset inward, then snapped to the midline.

    Works when the inner wall is incomplete: the inward ray-cast is *capped* at
    ``max_half_m`` so missing-inner-wall sections assume a nominal half-width
    instead of shooting across the open infield.
    """
    free = data.grid == 0
    interior = ce._interior_free_mask(free)
    blob = _largest_component(interior)
    solid = ndimage.binary_fill_holes(blob)

    k = max(1, int(round(offset_m / data.resolution)))
    eroded = _largest_component(binary_erosion(solid, disk(k)))

    cnts, _ = cv2.findContours(
        eroded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        raise ValueError("Outer-offset contour empty; reduce --offset-m")
    contour = max(cnts, key=lambda c: cv2.arcLength(c, True)).reshape(-1, 2)
    world = np.array(
        [ce._grid_to_world(float(r), float(c), data) for c, r in contour],
        dtype=np.float64,
    )
    center = _resample_loop(world, spacing_m)

    occupied = data.grid == 1
    for _ in range(snap_iters):
        _, normal = _tangents_normals(center)
        shifted = center.copy()
        for i, (x0, y0) in enumerate(center):
            nx, ny = normal[i]
            d_l = _raycast(x0, y0, nx, ny, occupied, data, max_half_m)
            d_r = _raycast(x0, y0, -nx, -ny, occupied, data, max_half_m)
            shift = np.clip((d_l - d_r) / 2.0, -max_half_m, max_half_m)
            shifted[i] = [x0 + nx * shift, y0 + ny * shift]
        center = _resample_loop(_smooth_loop(shifted, 11), spacing_m)

    _, normal = _tangents_normals(center)
    w_left = np.zeros(len(center))
    w_right = np.zeros(len(center))
    for i, (x0, y0) in enumerate(center):
        nx, ny = normal[i]
        w_left[i] = min(_raycast(x0, y0, nx, ny, occupied, data, max_half_m), max_half_m)
        w_right[i] = min(_raycast(x0, y0, -nx, -ny, occupied, data, max_half_m), max_half_m)
    w_left = ce._smooth_periodic(np.clip(w_left, 0.3, max_half_m))
    w_right = ce._smooth_periodic(np.clip(w_right, 0.3, max_half_m))

    corridor = solid & ~binary_erosion(solid, disk(1))  # outline for viz only
    return center, w_left, w_right, blob


def render(
    data: ce.MapData,
    centerline: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    corridor: np.ndarray,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, ox, oy = data.resolution, data.origin_x, data.origin_y
    h, w = data.grid.shape

    def to_world(rc):
        r, c = rc
        return ce._grid_to_world(float(r), float(c), data)

    n = len(centerline)
    tang = np.zeros_like(centerline)
    for i in range(n):
        tang[i] = centerline[(i + 1) % n] - centerline[i - 1]
    tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left = centerline + normal * w_left[:, None]
    right = centerline - normal * w_right[:, None]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    extent = [ox, ox + w * res, oy, oy + h * res]
    ax[0].imshow(data.grid == 1, cmap="Greys", origin="lower", extent=extent, alpha=0.5)
    cor_rc = np.argwhere(corridor)
    if len(cor_rc):
        cw = np.array([to_world(rc) for rc in cor_rc[::12]])
        ax[0].scatter(cw[:, 0], cw[:, 1], s=0.5, c="#e6f2ff", label="drivable")
    ax[0].plot(centerline[:, 0], centerline[:, 1], "b-", lw=2, label="centerline")
    ax[0].plot(left[:, 0], left[:, 1], color="orange", lw=1, label="left bound")
    ax[0].plot(right[:, 0], right[:, 1], color="magenta", lw=1, label="right bound")
    ax[0].scatter([centerline[0, 0]], [centerline[0, 1]], c="lime", s=60, zorder=5, label="start")
    ax[0].set_aspect("equal")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"cleaned track ({n} pts, loop {ce.loop_length(centerline):.1f} m)")

    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))])
    ax[1].plot(s, w_left, color="green", label="left half-width")
    ax[1].plot(s, w_right, color="purple", label="right half-width")
    ax[1].set_xlabel("arc length (m)")
    ax[1].set_ylabel("half-width (m)")
    ax[1].legend()
    ax[1].set_title("track widths")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-yaml", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--track-name", default="f1tenth_map")
    p.add_argument("--min-wall-px", type=int, default=40)
    p.add_argument("--close-radius", type=int, default=2)
    p.add_argument("--offset-m", type=float, default=0.6, help="inward offset from outer wall")
    p.add_argument("--max-half-m", type=float, default=1.3, help="capped half-width (m)")
    p.add_argument("--snap-iters", type=int, default=4)
    p.add_argument("--spacing-m", type=float, default=0.1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = ce.load_map_yaml(args.map_yaml)
    data = clean_grid(data, args.min_wall_px, args.close_radius)

    centerline, w_left, w_right, blob = centerline_from_outer_offset(
        data, args.offset_m, args.max_half_m, args.snap_iters, args.spacing_m
    )

    result = ce.CenterlineResult(centerline=centerline, w_tr_left=w_left, w_tr_right=w_right)
    csv_path = args.out_dir / f"{args.track_name}_centerline.csv"
    ce.write_centerline_csv(result, csv_path)
    ce.export_cleaned_map(data, args.out_dir, args.track_name)
    render(data, centerline, w_left, w_right, blob, args.out_dir / f"{args.track_name}_overlay.png")

    print(f"loop length {ce.loop_length(centerline):.1f} m, {len(centerline)} pts -> {csv_path}")


if __name__ == "__main__":
    main()
