"""Fit Genesis physics parameters to IRL data in the high-speed band.

Searches a small set of ``config.py["env"]`` parameters (drive force, drag, power
cap, brake force, tire friction, steering lag) to minimize a weighted relative
error against the IRL fit-band metrics. Low-speed crunching is already excluded
upstream: only ``measure`` samples inside each maneuver's ``fit_band`` reach the
objective, and ``c_roll``/stiction/speed-tracking knobs are intentionally not in
the search space.

Each candidate is profiled in an **isolated subprocess** (``profile genesis``):
Genesis is not robust to building and tearing down many scenes in one process, so
process isolation keeps the search stable. The winning parameters are re-checked
against ``scripts/physics_check.py`` (also in a subprocess) before being written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import run_dir  # noqa: E402
from .maneuvers import load_schedule, load_settings  # noqa: E402
from .metrics import flatten_fit_targets, summarize  # noqa: E402
from .schema import ProfileTable  # noqa: E402


def _metric_weight(key: str, schedule, fit_weights: dict) -> float:
    mid = key.split(".", 1)[0]
    try:
        metric = schedule.get(mid).metric
    except KeyError:
        return 1.0
    return float(fit_weights.get(metric, 1.0))


def _objective(sim_flat, irl_flat, schedule, fit_weights) -> tuple[float, int]:
    """Weighted RMS of relative errors over shared fit-band keys."""
    num = den = 0.0
    n = 0
    for key, irl_v in irl_flat.items():
        sim_v = sim_flat.get(key)
        if sim_v is None or not math.isfinite(sim_v) or abs(irl_v) < 1e-9:
            continue
        w = _metric_weight(key, schedule, fit_weights)
        rel = (sim_v - irl_v) / irl_v
        num += w * rel * rel
        den += w
        n += 1
    if den == 0:
        return math.inf, 0
    return math.sqrt(num / den), n


def _decode(z, names, bounds):
    """Map an unconstrained vector to bounded params via clamped [0,1] scaling."""
    out = {}
    for zi, name in zip(z, names):
        lo, hi = bounds[name]
        frac = min(1.0, max(0.0, float(zi)))
        out[name] = lo + frac * (hi - lo)
    return out


def _encode(params, names, bounds):
    return [
        (params[name] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])
        for name in names
    ]


def _profile_subprocess(params: dict, maneuver_ids, eval_run_id: str, backend: str):
    """Profile one candidate in an isolated process; return its ProfileTable."""
    cmd = [
        sys.executable, "-m", "vehicle_calibration", "profile", "genesis",
        "--run-id", eval_run_id, "--backend", backend,
        "--overrides", json.dumps(params),
    ]
    if maneuver_ids:
        cmd += ["--maneuvers", ",".join(maneuver_ids)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    csv_path = os.path.join(run_dir(eval_run_id), "profile_genesis.csv")
    if proc.returncode != 0 or not os.path.isfile(csv_path):
        raise RuntimeError(
            f"profile subprocess failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    return ProfileTable.read_csv(csv_path)


def _physics_check_subprocess(params: dict, backend: str) -> tuple[bool, str | None]:
    code = (
        "import json,sys,genesis as gs;"
        "from scripts.physics_check import headless_gs_init, run_physics_check;"
        f"headless_gs_init(gs.{backend});"
        "run_physics_check(extra_overrides=json.loads(sys.argv[1]), verbose=False)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(params)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return True, None
    return False, proc.stderr.strip().splitlines()[-1] if proc.stderr else "unknown"


def fit(
    run_id: str,
    target_summary: dict,
    settings: dict,
    schedule,
    max_iter: int = 40,
    backend: str = "cpu",
) -> dict:
    v_fit_min = float(settings["v_fit_min"])
    irl_flat = flatten_fit_targets(target_summary, v_fit_min)
    if not irl_flat:
        raise ValueError("target summary has no fit-band metrics to match")

    fit_params = settings["fit_params"]
    fit_weights = settings["fit_weights"]
    fit_maneuvers = settings.get("fit_maneuvers")
    names = list(fit_params.keys())
    bounds = {n: (float(fit_params[n]["min"]), float(fit_params[n]["max"])) for n in names}
    init = {n: float(fit_params[n]["init"]) for n in names}
    eval_run_id = f"{run_id}/_fit_eval"

    history: list[dict] = []

    def evaluate(params: dict) -> float:
        table = _profile_subprocess(params, fit_maneuvers, eval_run_id, backend)
        summ = summarize(table, schedule, v_fit_min)
        sim_flat = flatten_fit_targets(summ, v_fit_min)
        cost, n = _objective(sim_flat, irl_flat, schedule, fit_weights)
        history.append({"params": dict(params), "cost": cost, "n_matched": n})
        print(f"[fit]   cost={cost:.4f} (n={n})  " +
              " ".join(f"{k}={v:.3g}" for k, v in params.items()))
        return cost

    best = _search(evaluate, names, bounds, init, max_iter)

    physics_ok, physics_err = _physics_check_subprocess(best["params"], backend)

    result = {
        "v_fit_min": v_fit_min,
        "fit_maneuvers": fit_maneuvers,
        "best_params": best["params"],
        "best_cost": best["cost"],
        "n_matched": best["n_matched"],
        "physics_check_passed": physics_ok,
        "physics_check_error": physics_err,
        "history": history,
    }
    out = run_dir(run_id, create=True)
    path = os.path.join(out, "fitted_env.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[fit] best cost={best['cost']:.4f}  "
          f"physics_check={'PASS' if physics_ok else 'FAIL'}")
    print(f"[fit] wrote {path}")
    return result


def _search(evaluate, names, bounds, init, max_iter) -> dict:
    """Nelder-Mead if SciPy is available, else a coarse coordinate search."""
    best = {"params": dict(init), "cost": math.inf, "n_matched": 0}

    def track(params):
        cost = evaluate(params)
        if cost < best["cost"]:
            best.update(params=dict(params), cost=cost)
        return cost

    try:
        from scipy.optimize import minimize

        z0 = _encode(init, names, bounds)

        def f(z):
            return track(_decode(z, names, bounds))

        res = minimize(
            f, z0, method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-2, "fatol": 1e-3},
        )
        best["params"] = _decode(res.x, names, bounds)
        best["cost"] = float(res.fun)
    except ImportError:
        track(dict(init))
        for name in names:
            lo, hi = bounds[name]
            for frac in (0.25, 0.5, 0.75):
                cand = dict(best["params"])
                cand[name] = lo + frac * (hi - lo)
                track(cand)
    return best


def load_target_summary(run_id: str, target_path: str | None) -> dict:
    if target_path:
        with open(target_path) as fh:
            return json.load(fh)
    irl_path = os.path.join(run_dir(run_id), "summary_irl.json")
    if not os.path.isfile(irl_path):
        raise FileNotFoundError(
            f"no IRL summary at {irl_path}; run `parse bag ... --run-id {run_id}` "
            "first, or pass --target-summary"
        )
    with open(irl_path) as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Genesis physics to IRL data")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--target-summary",
        default=None,
        help="explicit summary JSON of fit targets (default: runs/<run-id>/summary_irl.json)",
    )
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args(argv)

    schedule = load_schedule()
    settings = load_settings()
    target = load_target_summary(args.run_id, args.target_summary)

    fit(args.run_id, target, settings, schedule,
        max_iter=args.max_iter, backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
