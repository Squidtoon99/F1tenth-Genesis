#!/usr/bin/env python3
"""Extract a geometric track centerline and per-point widths from a map PNG.

The IV 2026 ``*_smooth.csv`` raceline is an *optimal racing line*: on the
start/finish straight it runs outbound and inbound on parallel offsets. Using
that polyline as the centerline draws overlapping 2.2 m corridors near the
start line.

This script ray-casts from the raceline seed to the map walls, snaps each
sample to the local midline, deduplicates spatial revisits (single pass around
the track), resamples uniformly, and rotates the loop to start at the gym ego
pose.

Output columns: ``x_m, y_m, w_tr_right_m, w_tr_left_m`` (same as f1tenth
convention). Heading angles for training are derived from segment tangents in
``f1tenth_env/utils.py``, not stored in the CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load_raceline(path: Path) -> np.ndarray:
    text = path.read_text()
    delimiter = ";" if ";" in text.splitlines()[0] else ","
    data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float64)
    if data.dtype.names is None or "x_m" not in data.dtype.names:
        raise ValueError(f"{path}: expected named columns including x_m, y_m")
    return data


def load_map(map_yaml: Path) -> tuple[np.ndarray, dict]:
    meta = yaml.safe_load(map_yaml.read_text())
    image_path = map_yaml.parent / meta["image"]
    gray = np.array(Image.open(image_path))
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    return gray, meta


def world_to_pix(
    x: float, y: float, height: int, origin: tuple[float, float, float], res: float
) -> tuple[int, int]:
    ox, oy, _ = origin
    col = int(round((x - ox) / res))
    row = int(round(height - 1 - (y - oy) / res))
    return row, col


def ray_cast(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    occ: np.ndarray,
    height: int,
    origin: tuple[float, float, float],
    res: float,
    max_m: float = 6.0,
) -> float:
    step = res * 0.5
    n = int(max_m / step)
    for i in range(1, n + 1):
        x = x0 + dx * step * i
        y = y0 + dy * step * i
        row, col = world_to_pix(x, y, height, origin, res)
        if not (0 <= row < occ.shape[0] and 0 <= col < occ.shape[1]):
            return step * (i - 1)
        if occ[row, col]:
            return step * (i - 1)
    return max_m


def raceline_tangents(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    tangent = np.zeros_like(xy)
    for i in range(n):
        j = (i + 1) % n
        delta = xy[j] - xy[i]
        tangent[i] = delta / (np.linalg.norm(delta) + 1e-9)
    return tangent


def centerline_tangents(centers: np.ndarray) -> np.ndarray:
    """Forward tangents matching ``compute_track_boundaries`` (roll -1 semantics)."""
    n = len(centers)
    tangent = np.zeros_like(centers)
    for i in range(n):
        nxt = centers[(i + 1) % n]
        delta = nxt - centers[i]
        norm = np.linalg.norm(delta)
        if norm < 1e-8:
            prv = centers[i - 1]
            delta = centers[i] - prv
            norm = np.linalg.norm(delta)
        tangent[i] = delta / (norm + 1e-9)
    return tangent


def is_spatial_duplicate(
    pt: np.ndarray,
    kept_pts: list[np.ndarray],
    min_sep_m: float,
) -> bool:
    if not kept_pts:
        return False
    return bool(np.linalg.norm(np.stack(kept_pts) - pt, axis=1).min() <= min_sep_m)


def build_ordered_loop(
    centers: np.ndarray,
    start_xy: tuple[float, float],
    min_sep_m: float = 0.28,
) -> np.ndarray:
    """Single-pass loop from the start pose, skipping spatial revisits.

    Raceline index steps are preserved so we never chord across skipped gaps.
    """
    n = len(centers)
    i0 = int(np.linalg.norm(centers - np.asarray(start_xy), axis=1).argmin())
    order = [(i0 + k) % n for k in range(n)]

    kept_idx: list[int] = [order[0]]
    kept_pts: list[np.ndarray] = [centers[order[0]]]

    for target in order[1:]:
        pos = kept_idx[-1]
        while pos != target:
            pos = (pos + 1) % n
            pt = centers[pos]
            if is_spatial_duplicate(pt, kept_pts, min_sep_m):
                continue
            kept_idx.append(pos)
            kept_pts.append(pt)

    return centers[np.asarray(kept_idx, dtype=int)]


def measure_widths(
    centers: np.ndarray,
    occ: np.ndarray,
    meta: dict,
    max_half_width_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast wall distances using centerline tangents (same normal as OOB code)."""
    res = float(meta["resolution"])
    origin = tuple(meta["origin"])
    height = occ.shape[0]
    tangents = centerline_tangents(centers)

    w_left = np.zeros(len(centers), dtype=np.float64)
    w_right = np.zeros(len(centers), dtype=np.float64)

    for i, (x0, y0) in enumerate(centers):
        tx, ty = tangents[i]
        nx, ny = -ty, tx
        d_left = ray_cast(x0, y0, nx, ny, occ, height, origin, res, max_half_width_m)
        d_right = ray_cast(x0, y0, -nx, -ny, occ, height, origin, res, max_half_width_m)
        w_left[i] = min(d_left, max_half_width_m)
        w_right[i] = min(d_right, max_half_width_m)

    return w_left, w_right


def snap_to_midline(
    xy: np.ndarray,
    occ: np.ndarray,
    meta: dict,
    max_half_width_m: float = 6.0,
) -> np.ndarray:
    """Snap raceline seed points onto the local midline (positions only)."""
    res = float(meta["resolution"])
    origin = tuple(meta["origin"])
    height = occ.shape[0]
    tangents = raceline_tangents(xy)

    centers = np.zeros_like(xy)
    for i, (x0, y0) in enumerate(xy):
        tx, ty = tangents[i]
        nx, ny = -ty, tx
        d_left = ray_cast(x0, y0, nx, ny, occ, height, origin, res, max_half_width_m)
        d_right = ray_cast(x0, y0, -nx, -ny, occ, height, origin, res, max_half_width_m)
        centers[i, 0] = x0 + nx * (min(d_left, max_half_width_m) - min(d_right, max_half_width_m)) / 2.0
        centers[i, 1] = y0 + ny * (min(d_left, max_half_width_m) - min(d_right, max_half_width_m)) / 2.0

    return centers


def remove_backtracks(
    centers: np.ndarray,
    cos_thresh: float = -0.15,
) -> np.ndarray:
    """Drop points that reverse direction (corner artifacts near start/finish)."""
    if len(centers) < 3:
        return centers

    keep: list[int] = [0, 1]
    for i in range(2, len(centers)):
        a = centers[keep[-2]]
        b = centers[keep[-1]]
        c = centers[i]
        v1 = b - a
        v2 = c - b
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-6:
            continue
        cos_angle = float(np.dot(v1, v2) / (n1 * n2 + 1e-9))
        if cos_angle < cos_thresh and len(keep) >= 2:
            keep.pop()
        keep.append(i)

    return centers[np.asarray(keep, dtype=int)]


def resample_loop(centers: np.ndarray, n_points: int) -> np.ndarray:
    if len(centers) < 3:
        raise ValueError("Need at least 3 centerline points after deduplication")

    seg = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    close_seg = float(np.linalg.norm(centers[0] - centers[-1]))
    arc = np.concatenate([[0.0], np.cumsum(seg), [float(np.sum(seg)) + close_seg]])
    pts = np.vstack([centers, centers[0:1]])
    total = arc[-1]
    if total < 1e-3:
        raise ValueError("Degenerate centerline length")

    s = np.linspace(0.0, total, n_points, endpoint=False)
    x = np.interp(s, arc, pts[:, 0])
    y = np.interp(s, arc, pts[:, 1])
    return np.stack([x, y], axis=1)


def smooth_widths(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(ext, kernel, mode="valid")
    return smoothed[: len(values)]


def write_centerline(
    centers: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    out_path: Path,
) -> None:
    header = "# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
    lines = [header]
    for (x, y), wl, wr in zip(centers, w_left, w_right):
        lines.append(f"{x}, {y}, {wr}, {wl}\n")
    out_path.write_text("".join(lines))


def extract_centerline(
    map_yaml: Path,
    raceline_csv: Path,
    n_points: int = 671,
    start_xy: tuple[float, float] = (0.0, 2.0),
    min_sep_m: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray, meta = load_map(map_yaml)
    occ = gray < int(meta["occupied_thresh"] * 255)

    raceline = load_raceline(raceline_csv)
    xy = np.stack([raceline["x_m"], raceline["y_m"]], axis=1)

    centers = snap_to_midline(xy, occ, meta)
    centers = build_ordered_loop(centers, start_xy=start_xy, min_sep_m=min_sep_m)
    centers = remove_backtracks(centers)
    centers = resample_loop(centers, n_points)
    w_left, w_right = measure_widths(centers, occ, meta)
    w_left = smooth_widths(w_left, window=5)
    w_right = smooth_widths(w_right, window=5)
    return centers, w_left, w_right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_yaml", type=Path, help="Map YAML (with PNG sibling)")
    parser.add_argument("raceline_csv", type=Path, help="Seed raceline CSV")
    parser.add_argument("centerline_csv", type=Path, help="Output centerline CSV")
    parser.add_argument("--n-points", type=int, default=671)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=2.0)
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0.30,
        help="Min spacing when dropping spatial revisits (m)",
    )
    args = parser.parse_args()

    centers, w_left, w_right = extract_centerline(
        args.map_yaml,
        args.raceline_csv,
        n_points=args.n_points,
        start_xy=(args.start_x, args.start_y),
        min_sep_m=args.min_sep,
    )
    write_centerline(centers, w_left, w_right, args.centerline_csv)

    seg = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    length = float(np.sum(seg) + np.linalg.norm(centers[0] - centers[-1]))
    print(
        f"Wrote {len(centers)} points -> {args.centerline_csv} "
        f"(length ~{length:.1f} m, start=({centers[0,0]:.3f}, {centers[0,1]:.3f}))"
    )


if __name__ == "__main__":
    main()
