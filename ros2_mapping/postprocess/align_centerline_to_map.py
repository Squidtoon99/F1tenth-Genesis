#!/usr/bin/env python3
"""Re-express centerline CSV in map_server world coordinates.

Older extractions treated cv2 image row 0 as minimum world-y.  nav2 ``map_server``
and RViz ``/map`` use the ROS convention (image row 0 = maximum world-y).  This
script either:

  * ``--mode flip-y`` — flip existing CSV y about the map vertical midline (fast), or
  * ``--mode reextract`` — re-run ``process_real_map`` centerline on the map PGM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

import centerline_extractor as ce


def _load_map_height(yaml_path: Path) -> tuple[float, float, float, int]:
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    origin = meta["origin"]
    ox, oy = float(origin[0]), float(origin[1])
    res = float(meta["resolution"])
    image_path = yaml_path.parent / meta["image"]
    if not image_path.is_file():
        image_path = yaml_path.parent / Path(meta["image"]).name
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"map image not found: {image_path}")
    return ox, oy, res, int(img.shape[0])


def flip_y_csv(
    in_csv: Path,
    out_csv: Path,
    map_yaml: Path,
) -> ce.CenterlineResult:
    """Convert legacy extractor y into map_server y."""
    ox, oy, res, height = _load_map_height(map_yaml)
    raw = np.genfromtxt(in_csv, delimiter=",", names=True, dtype=np.float64)
    xs = np.asarray(raw["x_m"], dtype=np.float64).copy()
    ys = np.asarray(raw["y_m"], dtype=np.float64).copy()
    w_right = np.asarray(raw["w_tr_right_m"], dtype=np.float64).copy()
    w_left = np.asarray(raw["w_tr_left_m"], dtype=np.float64).copy()

    # y_map = 2*origin_y + height*res - y_legacy
    ys = 2.0 * oy + height * res - ys
    centerline = np.stack([xs, ys], axis=1)
    return ce.CenterlineResult(centerline=centerline, w_tr_left=w_left, w_tr_right=w_right)


def verify_on_map(result: ce.CenterlineResult, map_yaml: Path) -> float:
    """Return fraction of centerline points on free map pixels (map_server indexing)."""
    ox, oy, res, height = _load_map_height(map_yaml)
    import cv2

    image_path = map_yaml.parent / yaml.safe_load(map_yaml.open())["image"]
    if not image_path.is_file():
        image_path = map_yaml.parent / Path(image_path).name
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    xs, ys = result.centerline[:, 0], result.centerline[:, 1]
    px = ((xs - ox) / res).astype(int)
    py = (h - 1 - ((ys - oy) / res)).astype(int)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not ok.all():
        return 0.0
    return float(np.mean(img[py, px] > 200))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-yaml", type=Path, required=True)
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("flip-y", "reextract"),
        default="flip-y",
        help="flip-y: transform legacy CSV; reextract: run process_real_map pipeline",
    )
    args = p.parse_args()

    if args.mode == "reextract":
        # Delegate to process_real_map on the same map.
        from process_real_map import centerline_from_outer_offset, clean_grid

        data = ce.load_map_yaml(args.map_yaml)
        data = clean_grid(data, min_wall_px=40, close_radius=2)
        centerline, w_left, w_right, _ = centerline_from_outer_offset(
            data, offset_m=0.6, max_half_m=1.3, snap_iters=4, spacing_m=0.1
        )
        result = ce.CenterlineResult(centerline=centerline, w_tr_left=w_left, w_tr_right=w_right)
    else:
        result = flip_y_csv(args.in_csv, args.out_csv, args.map_yaml)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    ce.write_centerline_csv(result, args.out_csv)
    free_frac = verify_on_map(result, args.map_yaml)
    n = result.centerline.shape[0]
    xs, ys = result.centerline[:, 0], result.centerline[:, 1]
    print(f"wrote {args.out_csv} ({n} pts)")
    print(f"  x=[{xs.min():.2f},{xs.max():.2f}] y=[{ys.min():.2f},{ys.max():.2f}]")
    print(f"  map_server free_pixel_frac={free_frac:.3f} (expect >= 0.95)")
    if free_frac < 0.90:
        print("WARN: alignment still poor — check map_yaml matches the deployed /map", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
