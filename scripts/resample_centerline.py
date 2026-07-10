"""Resample + lightly smooth a bundled F1tenth centerline CSV.

Fits a periodic smoothing spline to the centerline (x_m, y_m), resamples it to
uniform arc-length spacing, and smooths the per-point track half-widths
(w_tr_right_m, w_tr_left_m) along arc length. This removes the SLAM-extraction
jitter and the start/finish seam kink that make the offset boundaries spike and
self-cross.

By default this is a DRY RUN: it writes a preview CSV and a before/after
comparison plot to outputs/ and does NOT modify the bundled (deployed) CSVs.
Pass --deploy to overwrite the bundled CSVs once you've verified the preview.

Usage:
    # preview only (safe)
    python scripts/resample_centerline.py --track IV_2026_SIM
    # write the verified result back to the bundled CSVs
    python scripts/resample_centerline.py --track IV_2026_SIM --deploy
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d, minimum_filter1d  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BUNDLED_DIRS = (
    os.path.join(REPO, "ros2_deploy", "f1tenth_rl_agent", "assets"),
    os.path.join(REPO, "ros2_deploy", "assets"),
)


def bundled_csvs(track: str) -> list[str]:
    found = [
        os.path.join(d, f"{track}_centerline.csv")
        for d in BUNDLED_DIRS
        if os.path.exists(os.path.join(d, f"{track}_centerline.csv"))
    ]
    if not found:
        raise FileNotFoundError(f"No bundled centerline for {track}")
    return found


def load_centerline(path: str):
    data = np.genfromtxt(
        path, delimiter=",", names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]
    )
    cx, cy = data["x_m"], data["y_m"]
    w_r, w_l = data["w_tr_right_m"], data["w_tr_left_m"]
    if np.allclose([cx[0], cy[0]], [cx[-1], cy[-1]]):
        cx, cy, w_r, w_l = cx[:-1], cy[:-1], w_r[:-1], w_l[:-1]
    return cx, cy, w_r, w_l


def closed_boundaries(cx, cy, w_l, w_r):
    cur = np.stack([cx, cy], -1)
    nxt = np.roll(cur, -1, axis=0)
    t = nxt - cur
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], -1)
    return cur + w_l[:, None] * n, cur - w_r[:, None] * n


def geom_stats(cx, cy, w_l, w_r):
    P = np.stack([cx, cy], -1)
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    t = (np.roll(P, -1, axis=0) - P)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    dots = np.clip((t * np.roll(t, 1, axis=0)).sum(1), -1, 1)
    ang = np.degrees(np.arccos(dots))
    ang_r = np.radians(ang)
    R = np.where(ang_r > 1e-6, seg / ang_r, np.inf)
    folds = int((R < np.maximum(w_l, w_r)).sum())
    return {
        "points": len(cx),
        "length_m": float(seg.sum()),
        "spacing_min": float(seg.min()),
        "spacing_max": float(seg.max()),
        "spacing_std": float(seg.std()),
        "turn_max_deg": float(ang.max()),
        "turn_gt30": int((ang > 30).sum()),
        "folds": folds,
    }


def _uniform_resample(cx, cy, w_r, w_l, ds: float):
    """Linear resample of the closed polyline to uniform arc-length step `ds`."""
    P = np.stack([cx, cy], -1)
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]  # len n+1, arc[-1] == total
    total = float(arc[-1])
    xc = np.r_[cx, cx[0]]
    yc = np.r_[cy, cy[0]]
    wrc = np.r_[w_r, w_r[0]]
    wlc = np.r_[w_l, w_l[0]]
    m = max(8, int(round(total / ds)))
    t = np.linspace(0.0, total, m, endpoint=False)
    return (
        np.interp(t, arc, xc),
        np.interp(t, arc, yc),
        np.interp(t, arc, wrc),
        np.interp(t, arc, wlc),
        total,
    )


def clamp_inner_width(cx, cy, w_r, w_l, spacing, k: float = 0.9,
                      min_half: float = 0.3, taper_m: float = 1.5):
    """Cap the *inner* half-width to k * local corner radius so a corner's inner
    wall can never cross itself (physically: a tube corridor's inner radius stays
    positive). The cap is spread over a ~taper_m arc (running minimum) and then
    smoothed, so the inner edge tapers gently into a tight corner instead of
    notching at a single apex point. Straights/wide corners are untouched."""
    P = np.stack([cx, cy], -1)
    fwd = np.roll(P, -1, axis=0) - P
    seg = np.linalg.norm(fwd, axis=1)
    t = fwd / np.maximum(seg[:, None], 1e-9)
    tp = np.roll(t, 1, axis=0)
    ang = np.arccos(np.clip((t * tp).sum(1), -1, 1))
    cross = tp[:, 0] * t[:, 1] - tp[:, 1] * t[:, 0]  # >0 left turn, <0 right turn
    seg_avg = 0.5 * (seg + np.roll(seg, 1))
    R = np.where(ang > 1e-6, seg_avg / ang, np.inf)
    cap = np.maximum(min_half, k * R)

    win = max(1, int(round(taper_m / spacing)) | 1)
    sig = max(taper_m / (2.0 * spacing), 1e-6)

    def _apply(w, inner_mask):
        cap_side = np.where(inner_mask, cap, np.inf)
        cap_side = minimum_filter1d(cap_side, win, mode="wrap")  # spread the narrowing
        out = np.minimum(w, cap_side)
        out = gaussian_filter1d(out, sig, mode="wrap")           # gentle taper
        return np.minimum(out, np.where(inner_mask, cap, np.inf))  # never exceed cap

    wl2 = _apply(w_l, cross > 0)
    wr2 = _apply(w_r, cross < 0)
    return wr2, wl2


def resample(cx, cy, w_r, w_l, spacing: float, smooth_m: float,
             clamp: bool = False, clamp_k: float = 0.9):
    """Round corners with a periodic Gaussian filter (length scale `smooth_m`),
    then resample to uniform `spacing`. Larger smooth_m => rounder corners and
    larger minimum radius (matches smooth HVAC-tube walls)."""
    # 1) dense uniform resample so the Gaussian sigma maps to a physical length
    ds = min(0.05, spacing / 4.0)
    fx, fy, fwr, fwl, total = _uniform_resample(cx, cy, w_r, w_l, ds)
    sigma = max(smooth_m / ds, 1e-6)

    # 2) periodic Gaussian smoothing of geometry + widths
    fx = gaussian_filter1d(fx, sigma, mode="wrap")
    fy = gaussian_filter1d(fy, sigma, mode="wrap")
    fwr = gaussian_filter1d(fwr, sigma, mode="wrap")
    fwl = gaussian_filter1d(fwl, sigma, mode="wrap")

    # 3) re-measure arc length on the smoothed curve, resample to target spacing
    P = np.stack([fx, fy], -1)
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    arc = np.r_[0.0, np.cumsum(seg)[:-1]]
    total = float(arc[-1] + seg[-1])
    n_out = max(8, int(round(total / spacing)))
    targets = np.linspace(0.0, total, n_out, endpoint=False)
    ox = np.interp(targets, arc, fx, period=total)
    oy = np.interp(targets, arc, fy, period=total)
    owr = np.interp(targets, arc, fwr, period=total)
    owl = np.interp(targets, arc, fwl, period=total)

    if clamp:
        owr, owl = clamp_inner_width(ox, oy, owr, owl, spacing, k=clamp_k)
    return ox, oy, owr, owl


def write_csv(path: str, cx, cy, w_r, w_l):
    lines = ["# x_m, y_m, w_tr_right_m, w_tr_left_m"]
    for x, y, wr, wl in zip(cx, cy, w_r, w_l):
        lines.append(f"{x:.6f}, {y:.6f}, {wr:.6f}, {wl:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_compare(track, src_path, before, after, out_png):
    bx, by, bwr, bwl = before
    ax_, ay_, awr, awl = after
    bl, br = closed_boundaries(bx, by, bwl, bwr)
    al, ar = closed_boundaries(ax_, ay_, awl, awr)

    def _c(a):
        return np.r_[a, a[:1]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=True, sharey=True)
    for ax, (cx, cy, L, R, title) in zip(
        axes,
        [
            (bx, by, bl, br, "current (deployed)"),
            (ax_, ay_, al, ar, "resampled (preview)"),
        ],
    ):
        ax.fill(
            np.r_[L[:, 0], R[::-1, 0]], np.r_[L[:, 1], R[::-1, 1]],
            color="0.85", zorder=0,
        )
        ax.plot(_c(L[:, 0]), _c(L[:, 1]), "-", lw=1.0, color="tab:red")
        ax.plot(_c(R[:, 0]), _c(R[:, 1]), "-", lw=1.0, color="tab:blue")
        ax.plot(_c(cx), _c(cy), "--", lw=0.8, color="0.3")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("y [m]")
    fig.suptitle(f"{track} centerline resample — verify before deploy\n{src_path}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="IV_2026_SIM")
    ap.add_argument("--spacing", type=float, default=0.4, help="uniform arc-length spacing [m]")
    ap.add_argument("--smooth-m", type=float, default=0.30,
                    help="Gaussian smoothing length scale [m]; larger => rounder corners")
    ap.add_argument("--clamp", action="store_true",
                    help="cap inner half-width to the local corner radius (kills folds)")
    ap.add_argument("--clamp-k", type=float, default=0.9,
                    help="inner half-width <= clamp_k * corner radius")
    ap.add_argument("--deploy", action="store_true", help="overwrite the bundled CSVs")
    args = ap.parse_args()

    csvs = bundled_csvs(args.track)
    src = csvs[0]
    cx, cy, w_r, w_l = load_centerline(src)
    before = (cx, cy, w_r, w_l)
    after = resample(cx, cy, w_r, w_l, args.spacing, args.smooth_m,
                     clamp=args.clamp, clamp_k=args.clamp_k)

    sb = geom_stats(cx, cy, w_l, w_r)
    sa = geom_stats(after[0], after[1], after[3], after[2])
    print(f"{'metric':<16}{'current':>12}{'resampled':>12}")
    for k in ("points", "length_m", "spacing_min", "spacing_max", "spacing_std",
              "turn_max_deg", "turn_gt30", "folds"):
        print(f"{k:<16}{sb[k]:>12.3f}{sa[k]:>12.3f}")

    preview_csv = os.path.join(REPO, "outputs", f"{args.track}_centerline_resampled.csv")
    os.makedirs(os.path.dirname(preview_csv), exist_ok=True)
    write_csv(preview_csv, after[0], after[1], after[2], after[3])
    out_png = os.path.join(REPO, "outputs", f"{args.track}_centerline_resample_compare.png")
    plot_compare(args.track, src, before, after, out_png)
    print(f"\npreview csv : {preview_csv}")
    print(f"compare png : {out_png}")

    if args.deploy:
        for c in csvs:
            write_csv(c, after[0], after[1], after[2], after[3])
        print(f"\nDEPLOYED to {len(csvs)} bundled CSV(s):")
        for c in csvs:
            print(f"  {c}")
    else:
        print("\nDRY RUN — bundled CSVs unchanged. Re-run with --deploy to apply.")


if __name__ == "__main__":
    main()
