"""Compare a Genesis profile against an IRL profile and write an alignment report.

Produces per-maneuver overlay plots (speed and yaw rate vs time, plus XY path for
cornering maneuvers) with the sub-``v_fit_min`` region shaded as out-of-fit, and a
scalar diff table restricted to fit-band metrics. Low-speed diagnostics are
reported separately and never gate pass/fail, so IRL crawl "crunching" can not
fail an otherwise good high-speed match.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import run_dir  # noqa: E402
from .maneuvers import load_schedule, load_settings  # noqa: E402
from .metrics import FIT_KEYS, flatten_fit_targets, summarize  # noqa: E402
from .schema import ProfileTable  # noqa: E402

# Fit-band relative-error thresholds (pass/fail) per scalar key.
_THRESHOLDS = {
    "accel_band": 0.15,
    "steady_speed": 0.10,
    "brake_decel": 0.20,
    "turn_radius": 0.15,
    "max_lat_acc": 0.15,
    "tau_steer": 0.20,
}


def _rel_err(sim: float, irl: float) -> float:
    if irl is None or not math.isfinite(irl) or abs(irl) < 1e-9:
        return math.nan
    return abs(sim - irl) / abs(irl)


def build_report(
    sim_summary: dict, irl_summary: dict, v_fit_min: float
) -> dict:
    sim_flat = flatten_fit_targets(sim_summary, v_fit_min)
    irl_flat = flatten_fit_targets(irl_summary, v_fit_min)

    rows = []
    n_pass = n_total = 0
    for key in sorted(set(sim_flat) | set(irl_flat)):
        sim_v = sim_flat.get(key, math.nan)
        irl_v = irl_flat.get(key, math.nan)
        rel = _rel_err(sim_v, irl_v)
        scalar = key.split(".", 1)[1]
        thr = _THRESHOLDS.get(scalar, 0.20)
        ok = math.isfinite(rel) and rel <= thr
        if math.isfinite(rel):
            n_total += 1
            n_pass += int(ok)
        rows.append(
            {
                "key": key,
                "sim": sim_v,
                "irl": irl_v,
                "rel_err": rel,
                "threshold": thr,
                "pass": bool(ok),
            }
        )
    return {
        "v_fit_min": v_fit_min,
        "n_pass": n_pass,
        "n_total": n_total,
        "passed": n_total > 0 and n_pass == n_total,
        "fit_band_metrics": rows,
        "diagnostics": _collect_diagnostics(sim_summary, irl_summary),
    }


def _collect_diagnostics(sim_summary: dict, irl_summary: dict) -> list[dict]:
    out = []
    for mid, m in sim_summary["maneuvers"].items():
        fit_keys = set(FIT_KEYS.get(m["metric"], []))
        irl_vals = irl_summary["maneuvers"].get(mid, {}).get("values", {})
        for key, sim_v in m["values"].items():
            if key in fit_keys and not m["diagnostic_only"]:
                continue  # already in the fit-band table
            out.append(
                {
                    "key": f"{mid}.{key}",
                    "sim": sim_v,
                    "irl": irl_vals.get(key, math.nan),
                }
            )
    return out


def _shade_low_speed(ax, t, speed, v_fit_min) -> None:
    below = np.asarray(speed) < v_fit_min
    if below.any():
        ax.fill_between(
            t, 0, 1, where=below, transform=ax.get_xaxis_transform(),
            color="0.85", alpha=0.6, step="mid", label=f"v < {v_fit_min:g} (out-of-fit)",
        )


def _plot_maneuver(sim: ProfileTable, irl: ProfileTable, mid: str, metric: str,
                   v_fit_min: float, out_dir: str) -> None:
    s = sim.select(mid, role="measure")
    has_irl = mid in irl.maneuver_ids()
    r = irl.select(mid, role="measure") if has_irl else None

    if metric == "corner":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(s["t"], s["speed"], label="genesis")
        if r is not None:
            axes[0].plot(r["t"], r["speed"], label="irl")
        _shade_low_speed(axes[0], s["t"], s["speed"], v_fit_min)
        axes[0].set(xlabel="t (s)", ylabel="speed (m/s)", title=f"{mid}: speed")
        axes[0].legend(fontsize=8)
        axes[1].plot(s["x"], s["y"], label="genesis")
        if r is not None:
            axes[1].plot(r["x"], r["y"], label="irl")
        axes[1].set(xlabel="x (m)", ylabel="y (m)", title=f"{mid}: path")
        axes[1].axis("equal")
        axes[1].legend(fontsize=8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(s["t"], s["speed"], label="genesis")
        if r is not None:
            axes[0].plot(r["t"], r["speed"], label="irl")
        _shade_low_speed(axes[0], s["t"], s["speed"], v_fit_min)
        axes[0].set(xlabel="t (s)", ylabel="speed (m/s)", title=f"{mid}: speed")
        axes[0].legend(fontsize=8)
        axes[1].plot(s["t"], np.abs(s["omega_z"]), label="genesis")
        if r is not None:
            axes[1].plot(r["t"], np.abs(r["omega_z"]), label="irl")
        _shade_low_speed(axes[1], s["t"], s["speed"], v_fit_min)
        axes[1].set(xlabel="t (s)", ylabel="|yaw rate| (rad/s)", title=f"{mid}: yaw rate")
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{mid}.png"), dpi=110)
    plt.close(fig)


def run_compare(run_id: str) -> dict:
    out = run_dir(run_id, create=True)
    sim_csv = os.path.join(out, "profile_genesis.csv")
    irl_csv = os.path.join(out, "profile_irl.csv")
    if not os.path.isfile(sim_csv):
        raise FileNotFoundError(f"missing {sim_csv}; run `profile genesis` first")

    schedule = load_schedule()
    settings = load_settings()
    v_fit_min = float(settings["v_fit_min"])

    sim = ProfileTable.read_csv(sim_csv)
    sim_summary = summarize(sim, schedule, v_fit_min)

    if os.path.isfile(irl_csv):
        irl = ProfileTable.read_csv(irl_csv)
    else:
        print(f"[compare] no {irl_csv}; plotting Genesis only, empty IRL side")
        irl = ProfileTable.from_rows([])
    irl_summary = summarize(irl, schedule, v_fit_min)

    plots_dir = os.path.join(out, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for man in schedule.maneuvers:
        if man.id in sim.maneuver_ids():
            _plot_maneuver(sim, irl, man.id, man.metric, v_fit_min, plots_dir)

    report = build_report(sim_summary, irl_summary, v_fit_min)
    report_path = os.path.join(out, "alignment_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    _print_report(report)
    print(f"[compare] plots -> {plots_dir}")
    print(f"[compare] report -> {report_path}")
    return report


def _print_report(report: dict) -> None:
    print(f"\n=== fit-band alignment ({report['n_pass']}/{report['n_total']} pass) ===")
    print(f"{'metric':28s} {'genesis':>10s} {'irl':>10s} {'rel_err':>8s}  ok")
    for row in report["fit_band_metrics"]:
        rel = row["rel_err"]
        rel_s = f"{rel*100:6.1f}%" if math.isfinite(rel) else "    n/a"
        print(
            f"{row['key']:28s} {row['sim']:10.3f} {row['irl']:10.3f} "
            f"{rel_s:>8s}  {'Y' if row['pass'] else '.'}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Genesis vs IRL profiles")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    report = run_compare(args.run_id)
    return 0 if report["n_total"] == 0 or report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
