"""Plot a bundled F1tenth centerline CSV (x_m, y_m, w_tr_right_m, w_tr_left_m).

Renders the centerline plus the left/right track boundaries derived from the
per-point widths, exactly as the training env interprets them. Read-only.

Usage:
    python scripts/plot_centerline.py --track IV_2026_SIM
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def bundled_csv(track: str) -> str:
    for root in (
        os.path.join(REPO, "ros2_deploy", "f1tenth_rl_agent", "assets"),
        os.path.join(REPO, "ros2_deploy", "assets"),
    ):
        p = os.path.join(root, f"{track}_centerline.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No bundled centerline for {track}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="IV_2026_SIM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = bundled_csv(args.track)
    data = np.genfromtxt(path, delimiter=",", names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
    cx, cy = data["x_m"], data["y_m"]
    w_l, w_r = data["w_tr_left_m"], data["w_tr_right_m"]

    # Drop an explicit duplicate closing vertex (first == last) before computing
    # tangents. Otherwise np.roll produces a zero-length segment at the seam, the
    # boundary normal collapses to zero, and the track appears to pinch/gap at the
    # start. The loop is re-closed implicitly by the wrap-around roll below.
    if np.allclose([cx[0], cy[0]], [cx[-1], cy[-1]]):
        cx, cy, w_l, w_r = cx[:-1], cy[:-1], w_l[:-1], w_r[:-1]

    # closed-loop tangents -> left normal n_hat = [-t_y, t_x]
    nxt = np.roll(np.stack([cx, cy], -1), -1, axis=0)
    cur = np.stack([cx, cy], -1)
    t = nxt - cur
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], -1)
    left = cur + w_l[:, None] * n
    right = cur - w_r[:, None] * n

    length = float(np.linalg.norm(np.diff(np.vstack([cur, cur[:1]]), axis=0), axis=-1).sum())

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.fill(
        np.r_[left[:, 0], right[::-1, 0]],
        np.r_[left[:, 1], right[::-1, 1]],
        color="0.85", zorder=0, label="track surface",
    )
    # append the first vertex so the closed loop renders without a seam gap
    def _close(a):
        return np.r_[a, a[:1]]

    ax.plot(_close(left[:, 0]), _close(left[:, 1]), "-", lw=1.2, color="tab:red", label="left boundary")
    ax.plot(_close(right[:, 0]), _close(right[:, 1]), "-", lw=1.2, color="tab:blue", label="right boundary")
    ax.plot(_close(cx), _close(cy), "--", lw=1.0, color="0.3", label="centerline")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{args.track} centerline ({len(cx)} pts, loop length {length:.1f} m)\n{path}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)

    out = args.out or os.path.join(REPO, "outputs", f"{args.track}_centerline.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"points={len(cx)}  loop_length_m={length:.3f}")
    print(f"x_range=[{cx.min():.2f},{cx.max():.2f}]  y_range=[{cy.min():.2f},{cy.max():.2f}]")
    print(f"width_left[min/max]={w_l.min():.2f}/{w_l.max():.2f}  "
          f"width_right[min/max]={w_r.min():.2f}/{w_r.max():.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
