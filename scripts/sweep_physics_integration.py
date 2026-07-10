#!/usr/bin/env python3
"""Sweep physics-integration settings, gating each candidate on the physics
stability check and measuring env throughput.

For each candidate (sim_substeps / solver iterations), this runs the full
``physics_check`` stability gate (accel, top-speed, brake, lateral<=mu*g,
upright, no-NaN) and, only if it passes, a short throughput benchmark. It then
reports a table and recommends the fastest stable configuration.

``sim_dt`` and ``control_interval`` are held fixed so the physics_check timing
constants stay valid; only the per-step integration cost is swept.

Example:
    python scripts/sweep_physics_integration.py --bench-envs 256 --steps 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_check import headless_gs_init, run_physics_check  # noqa: E402
from bench_env import bench_once, _make_sync, _select_backend  # noqa: E402


# (label, override dict) candidates. sim_dt / control_interval fixed.
def default_candidates() -> list[tuple[str, dict]]:
    candidates: list[tuple[str, dict]] = []
    for substeps in (10, 8, 6, 5, 4, 3, 2):
        candidates.append(
            (f"substeps={substeps}", {"sim_substeps": substeps})
        )
    # A couple of solver-iteration reductions at a stable substep count.
    for iters in (30, 20):
        candidates.append(
            (
                f"substeps=5,iters={iters}",
                {"sim_substeps": 5, "solver_iterations": iters,
                 "solver_ls_iterations": iters},
            )
        )
    return candidates


def run_single_candidate(override: dict, backend: str, bench_envs: int,
                         steps: int, warmup: int) -> dict:
    """Run one candidate (stability gate + bench) in the current process."""
    row = {"stable": False, "reason": "", "steps_per_s": 0.0,
           "top_speed": None, "brake_time": None}
    headless_gs_init(_select_backend(backend))
    try:
        stats = run_physics_check(extra_overrides=override, verbose=False)
        row["stable"] = True
        row["top_speed"] = stats["top_speed"]
        row["brake_time"] = stats["brake_time"]
    except AssertionError as exc:
        row["reason"] = str(exc)[:70]
        return row
    except Exception as exc:  # noqa: BLE001
        row["reason"] = f"{type(exc).__name__}: {str(exc)[:55]}"
        return row

    try:
        res = bench_once(
            num_envs=bench_envs, warmup=warmup, steps=steps,
            control_interval=10, sync=_make_sync(), overrides=override,
        )
        row["steps_per_s"] = res["env_steps_per_s"]
    except Exception as exc:  # noqa: BLE001
        row["reason"] = f"bench failed: {type(exc).__name__}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Physics integration sweep")
    parser.add_argument(
        "--backend", type=str, default="cpu",
        choices=["auto", "cpu", "gpu", "cuda", "metal"],
    )
    parser.add_argument("--bench-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=6)
    # Worker mode: evaluate a single candidate and emit one JSON line. Genesis
    # cannot rebuild many scenes per process reliably, so the driver spawns one
    # isolated subprocess per candidate.
    parser.add_argument("--single", type=str, default=None,
                        help="JSON override dict for worker mode")
    args = parser.parse_args()

    if args.single is not None:
        override = json.loads(args.single)
        row = run_single_candidate(
            override, args.backend, args.bench_envs, args.steps, args.warmup
        )
        print("RESULT_JSON:" + json.dumps(row))
        return 0

    candidates = default_candidates()
    rows: list[dict] = []

    print(f"=== Physics integration sweep (backend={args.backend}, "
          f"bench_envs={args.bench_envs}) ===\n")

    for label, override in candidates:
        proc = subprocess.run(
            [sys.executable, __file__,
             "--single", json.dumps(override),
             "--backend", args.backend,
             "--bench-envs", str(args.bench_envs),
             "--steps", str(args.steps),
             "--warmup", str(args.warmup)],
            capture_output=True, text=True,
            env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/mpl")},
        )
        row = {"label": label, "stable": False, "reason": "crash",
               "steps_per_s": 0.0, "top_speed": None, "brake_time": None}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                row.update(json.loads(line[len("RESULT_JSON:"):]))
                row["label"] = label
                break

        rows.append(row)
        status = "OK  " if row["stable"] else "FAIL"
        ts = f"top={row['top_speed']:.2f}" if row["top_speed"] is not None else "top=---"
        bt = f"brake={row['brake_time']}" if row["brake_time"] is not None else ""
        print(f"[{status}] {label:24s} env_steps/s={row['steps_per_s']:>10,.0f} "
              f"{ts} {bt} {row['reason']}")

    stable = [r for r in rows if r["stable"] and r["steps_per_s"] > 0]
    print("\n=== Summary ===")
    if not stable:
        print("No stable candidate found.")
        return 1

    baseline = next((r for r in rows if r["label"] == "substeps=10"), None)
    best = max(stable, key=lambda r: r["steps_per_s"])
    print(f"Fastest stable: {best['label']} "
          f"({best['steps_per_s']:,.0f} env_steps/s, top={best['top_speed']:.2f} m/s)")
    if baseline and baseline["steps_per_s"] > 0:
        speedup = best["steps_per_s"] / baseline["steps_per_s"]
        print(f"Speedup vs substeps=10 baseline: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
