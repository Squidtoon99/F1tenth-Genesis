#!/usr/bin/env python3
"""Sanity-check centerline CSV is in the map frame (overlay on PGM)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> int:
    maps_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/shereef/maps")
    yaml_path = maps_dir / "f1tenth_map.yaml"
    pgm_path = maps_dir / "f1tenth_map.pgm"
    png_path = maps_dir / "f1tenth_map.png"
    map_img = pgm_path if pgm_path.exists() else png_path
    csv_path = maps_dir / "f1tenth_map_centerline.csv"

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    origin = meta["origin"]
    res = float(meta["resolution"])
    ox, oy = float(origin[0]), float(origin[1])

    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        comments="#",
    )
    xs, ys = data["x_m"], data["y_m"]

    overlay = maps_dir / "f1tenth_map_overlay.png"
    if str(map_img).endswith(".png"):
        import cv2
        img = cv2.imread(str(map_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"FAIL: could not read {map_img}")
            return 1
        h, w = img.shape
    else:
        with open(map_img, "rb") as f:
            assert f.readline().strip() == b"P5"
            dims = f.readline().strip().split()
            w, h = int(dims[0]), int(dims[1])
            f.readline()
            img = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    # map_server / RViz: image row 0 is the north (max-y) edge.
    px = ((xs - ox) / res).astype(int)
    py = (h - 1 - ((ys - oy) / res)).astype(int)
    if overlay.exists():
        print(f"(checked against {map_img.name}; overlay present at {overlay.name})")
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not ok.all():
        bad = np.where(~ok)[0][:5]
        print(f"FAIL: {np.sum(~ok)} points out of bounds; examples idx={bad}")
        return 1

    vals = img[py, px]
    free_frac = np.mean(vals > 200)
    print(f"map origin=({ox},{oy}) res={res} size={w}x{h}")
    print(f"centerline N={len(xs)} x=[{xs.min():.2f},{xs.max():.2f}] y=[{ys.min():.2f},{ys.max():.2f}]")
    print(f"in_bounds=100% free_pixel_frac={free_frac:.3f} (expect ~1.0 on corridor)")
    if free_frac < 0.95:
        print("WARN: some centerline points sit on occupied pixels — check frame alignment")
        return 2
    print("PASS: centerline aligns with map frame (no transform applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
