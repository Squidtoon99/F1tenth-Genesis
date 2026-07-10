#!/usr/bin/env python3
"""Plot a centerline CSV with left/right boundaries on the occupancy map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros2_deploy" / "f1tenth_rl_agent"))
from f1tenth_rl_agent.track_io import compute_track_boundaries, load_track_csv


def load_map_extent(map_yaml: Path) -> tuple[np.ndarray, float, float, float, float]:
    meta = yaml.safe_load(map_yaml.read_text())
    image_path = map_yaml.parent / meta["image"]
    gray = np.array(Image.open(image_path))
    if gray.ndim == 3:
        gray = gray[:, :, 0]

    res = float(meta["resolution"])
    ox, oy, _ = meta["origin"]
    height, width = gray.shape
    x_max = ox + width * res
    y_max = oy + height * res
    return gray, ox, oy, x_max, y_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centerline",
        type=Path,
        default=REPO_ROOT
        / "ros2_deploy/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=REPO_ROOT / "ros2_deploy/assets/IV_2026_SIM.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/IV_2026_SIM_track_map.png",
    )
    args = parser.parse_args()

    cl, wl, wr = load_track_csv(str(args.centerline))
    left, right = compute_track_boundaries(cl, wl, wr)

    gray, ox, oy, x_max, y_max = load_map_extent(args.map_yaml)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.imshow(
        gray,
        cmap="gray",
        origin="upper",
        extent=[ox, x_max, oy, y_max],
        alpha=0.55,
    )

    corridor_x = np.concatenate([left[:, 0], right[::-1, 0], left[:1, 0]])
    corridor_y = np.concatenate([left[:, 1], right[::-1, 1], left[:1, 1]])
    ax.fill(corridor_x, corridor_y, color="#2ecc71", alpha=0.25, label="drivable corridor")
    ax.plot(left[:, 0], left[:, 1], color="#7bed9f", linewidth=1.0, label="left boundary")
    ax.plot(right[:, 0], right[:, 1], color="#1e8449", linewidth=1.0, label="right boundary")
    ax.plot(cl[:, 0], cl[:, 1], color="white", linewidth=1.5, label="centerline")
    ax.scatter(
        [cl[0, 0]],
        [cl[0, 1]],
        color="#f1c40f",
        s=60,
        zorder=5,
        label=f"start ({cl[0,0]:.2f}, {cl[0,1]:.2f})",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("IV 2026 SIM — centerline and track widths")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
