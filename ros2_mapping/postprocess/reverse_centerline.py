#!/usr/bin/env python3
"""Reverse centerline vertex order and swap left/right widths for opposite travel direction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reverse_centerline_csv(src: Path, dst: Path) -> None:
    data = np.genfromtxt(src, delimiter=",", names=True, comments="#")
    names = data.dtype.names
    if names is None or "x_m" not in names or "y_m" not in names:
        raise ValueError(f"{src}: expected columns x_m, y_m, w_tr_left_m, w_tr_right_m")

    rows = []
    for i in range(len(data) - 1, -1, -1):
        x = float(data["x_m"][i])
        y = float(data["y_m"][i])
        wl = float(data["w_tr_left_m"][i]) if "w_tr_left_m" in names else 0.0
        wr = float(data["w_tr_right_m"][i]) if "w_tr_right_m" in names else 0.0
        # Swap widths: left/right are defined relative to travel direction.
        rows.append((x, y, wl, wr))

    header = "# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
    lines = [header]
    for x, y, old_left, old_right in rows:
        lines.append(f"{x:.6f}, {y:.6f}, {old_left:.6f}, {old_right:.6f}\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines))


def signed_area(path: Path) -> float:
    data = np.genfromtxt(path, delimiter=",", names=True, comments="#")
    xs, ys = data["x_m"], data["y_m"]
    n = len(xs)
    return 0.5 * sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)",
    )
    args = parser.parse_args()
    out = args.output or args.input_csv
    before = signed_area(args.input_csv)
    reverse_centerline_csv(args.input_csv, out)
    after = signed_area(out)
    print(f"Wrote {len(np.genfromtxt(out, delimiter=',', names=True, comments='#'))} points -> {out}")
    print(f"Signed area: {before:.1f} -> {after:.1f} ({'CW' if after < 0 else 'CCW'})")


if __name__ == "__main__":
    main()
