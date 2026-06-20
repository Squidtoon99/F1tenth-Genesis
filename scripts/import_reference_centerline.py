#!/usr/bin/env python3
"""Import a reference centerline CSV in map-offset coordinates into repo assets.

The Discord / gym export stores ``x_m, y_m`` shifted by the map yaml origin.
World coordinates: ``world = csv_xy + origin``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "ros2_deploy/assets/IV_2026_SIM.yaml"
DEFAULT_OUT = [
    REPO_ROOT / "ros2_deploy/assets/IV_2026_SIM_centerline.csv",
    REPO_ROOT / "ros2_deploy/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
]


def import_centerline(
    source_csv: Path,
    map_yaml: Path,
    out_paths: list[Path],
) -> int:
    meta = yaml.safe_load(map_yaml.read_text())
    ox, oy, _ = meta["origin"]

    raw = np.genfromtxt(source_csv, delimiter=",", names=True, dtype=np.float64)
    required = {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"}
    if raw.dtype.names is None or not required.issubset(set(raw.dtype.names)):
        raise ValueError(f"{source_csv}: expected columns {sorted(required)}")

    header = "# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
    lines = [header]
    for row in raw:
        x = row["x_m"] + ox
        y = row["y_m"] + oy
        lines.append(
            f"{x}, {y}, {row['w_tr_right_m']}, {row['w_tr_left_m']}\n"
        )

    text = "".join(lines)
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    return len(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=DEFAULT_MAP,
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        dest="outputs",
        help="Output path (repeatable; defaults to both asset copies)",
    )
    args = parser.parse_args()
    outputs = args.outputs or DEFAULT_OUT
    n = import_centerline(args.source_csv, args.map_yaml, outputs)
    for path in outputs:
        print(f"Wrote {n} points -> {path}")


if __name__ == "__main__":
    main()
