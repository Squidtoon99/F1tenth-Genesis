#!/usr/bin/env python3
"""Render a clean gym occupancy PNG from a centerline CSV.

f1tenth_gym PNG convention (after load + threshold):
  - pixel > 128  -> internal bitmap 255 = free (drivable)
  - pixel <= 128 -> internal bitmap 0   = obstacle (wall)

The bundled IV_2026_SIM.png is a thin, jagged black scribble with speckle noise.
This script fills the corridor between track boundaries as solid white (free)
on a black (wall) background.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Reuse boundary math from the ROS package when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "ros2_deploy" / "f1tenth_rl_agent"))

from f1tenth_rl_agent.track_io import compute_track_boundaries, load_track_csv  # noqa: E402


def _world_to_rc(
    x: float, y: float, origin: tuple[float, float, float], resolution: float
) -> tuple[int, int]:
    ox, oy, otheta = origin
    oc, os = np.cos(otheta), np.sin(otheta)
    x_rot = (x - ox) * oc + (y - oy) * os
    y_rot = -(x - ox) * os + (y - oy) * oc
  # Match f1tenth_gym ``xy_2_rc`` (truncate, do not round).
    return int(y_rot / resolution), int(x_rot / resolution)


def render_corridor_map(
    centerline: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    origin: tuple[float, float, float],
    resolution: float,
    width_px: int,
    height_px: int,
    margin_m: float = 0.05,
) -> np.ndarray:
    """Return uint8 PNG pixel values: 255 = free corridor, 0 = occupied (wall).

    f1tenth_gym loads the image then builds an internal bitmap where 0 is obstacle
  and 255 is free (see ``get_dt``). Pixels > 128 become free, <= 128 become walls.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python (cv2) required") from exc

    left, right = compute_track_boundaries(centerline, w_left, w_right)
    ring = np.vstack([left, right[::-1]])
    pts = np.zeros((len(ring), 1, 2), dtype=np.int32)
    for i, (x, y) in enumerate(ring):
        r, c = _world_to_rc(float(x), float(y), origin, resolution)
        pts[i, 0, 0] = c
        pts[i, 0, 1] = r

    # Black = wall in PNG -> gym bitmap 0 = obstacle.
    img = np.zeros((height_px, width_px), dtype=np.uint8)
    cv2.fillPoly(img, [pts], color=255)

    if margin_m > 0.0:
        k = max(1, int(round(margin_m / resolution)))
        kernel = np.ones((k, k), np.uint8)
        free = (img > 128).astype(np.uint8)
        free = cv2.erode(free, kernel, iterations=1)
        img = np.where(free > 0, 255, 0).astype(np.uint8)

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_yaml", type=Path, help="Existing map yaml (origin/resolution/size)")
    parser.add_argument("centerline_csv", type=Path)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG (default: same dir as yaml, image name from yaml)",
    )
    parser.add_argument("--margin-m", type=float, default=0.0)
    args = parser.parse_args()

    with args.map_yaml.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)

    from PIL import Image

    image_name = meta["image"]
    ref_png = args.map_yaml.parent / image_name
    if ref_png.is_file():
        ref = Image.open(ref_png)
        width_px, height_px = ref.size
    else:
        raise FileNotFoundError(f"Reference image not found: {ref_png}")

    origin = tuple(float(v) for v in meta["origin"])
    resolution = float(meta["resolution"])
    centerline, w_left, w_right = load_track_csv(str(args.centerline_csv))

    # Draw in the same orientation f1tenth_gym uses after FLIP_TOP_BOTTOM on load.
    height_px = int(ref.size[1])
    width_px = int(ref.size[0])
    img_gym = render_corridor_map(
        centerline,
        w_left,
        w_right,
        origin,
        resolution,
        width_px,
        height_px,
        margin_m=args.margin_m,
    )
    img_disk = np.flipud(img_gym)

    out_png = args.out_png or ref_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_disk, mode="L").save(out_png)
    free_frac = (img_gym > 128).mean()
    print(f"Wrote {out_png} ({width_px}x{height_px}) free_fraction={free_frac:.3f}")


if __name__ == "__main__":
    main()
