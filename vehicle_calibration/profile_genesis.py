"""Drive Genesis through the shared maneuver schedule and log per-step state.

Reuses the headless init and the env-construction conventions from
``scripts/physics_check.py`` (terminations disabled so a single rollout stays
continuous). Produces a :class:`ProfileTable` with the canonical schema so the
output is directly comparable to a parsed rosbag from the real car.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import genesis as gs  # noqa: E402
import torch  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from scripts.physics_check import headless_gs_init  # noqa: E402

from . import DEFAULT_PROFILE_YAML, run_dir  # noqa: E402
from .maneuvers import Maneuver, Schedule, load_schedule, load_settings  # noqa: E402
from .metrics import summarize  # noqa: E402
from .schema import ProfileTable  # noqa: E402

# Terminations/jitter disabled: we want a clean, continuous open-loop rollout.
_CONTINUOUS_OVERRIDES = {
    "reset_spawn_margin_m": 0.0,
    "reset_yaw_jitter_rad": 0.0,
    "reset_along_track_jitter_m": 0.0,
    "reset_speed_min_mps": 0.0,
    "reset_speed_max_mps": 0.0,
    "simulate_action_latency": False,
    "launch_strategy": "fixed",
    "car_spawn_pos": (0.0, 0.0, 0.05),
    "car_spawn_rot": (0.0, 0.0, 0.0),
    "term_oob_max_consecutive": 10**9,
    "term_oob_margin_m": -100.0,
    "term_not_moving_time_s": 10**9,
    "term_heading_error_rad": 10.0,
}


def build_env(env_overrides: dict | None = None) -> tuple[F1tenthEnv, dict]:
    """Construct a single-env F1tenthEnv for profiling. Assumes gs is initialized."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = cfg["env"]
    env_cfg.update(_CONTINUOUS_OVERRIDES)
    if env_overrides:
        env_cfg.update(env_overrides)
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
    )
    return env, cfg


def _yaw_from_quat(quat: torch.Tensor) -> float:
    w, x, y, z = (float(quat[0, i].item()) for i in range(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _reset(env: F1tenthEnv) -> None:
    env.reset(torch.tensor([True], dtype=gs.tc_bool, device=gs.device))


def _run_maneuver(env: F1tenthEnv, man: Maneuver, control_hz: float) -> list[dict]:
    control_interval = int(env.control_interval)
    rows: list[dict] = []
    t = 0.0
    dt = 1.0 / control_hz
    if man.reset_before:
        _reset(env)
    for seg in man.segments:
        action = torch.tensor(
            [[seg.throttle, seg.steer]], dtype=gs.tc_float, device=gs.device
        )
        for _ in range(round(seg.duration_s * control_hz)):
            env.step(action, n_steps=control_interval)
            vx = float(env.base_lin_vel[0, 0].item())
            vy = float(env.base_lin_vel[0, 1].item())
            rows.append(
                {
                    "t": t,
                    "maneuver": man.id,
                    "role": seg.role,
                    "throttle": seg.throttle,
                    "steer": seg.steer,
                    "x": float(env.base_pos[0, 0].item()),
                    "y": float(env.base_pos[0, 1].item()),
                    "yaw": _yaw_from_quat(env.base_quat),
                    "vx": vx,
                    "vy": vy,
                    "speed": math.hypot(vx, vy),
                    "ax": float(env.base_lin_acc[0, 0].item()),
                    "ay": float(env.base_lin_acc[0, 1].item()),
                    "omega_z": float(env.base_ang_vel[0, 2].item()),
                    "steer_state": float(env.steer_state[0].item()),
                    "source": "genesis",
                }
            )
            t += dt
    return rows


def run_profile(
    schedule: Schedule,
    env_overrides: dict | None = None,
    maneuver_ids: list[str] | None = None,
) -> ProfileTable:
    """Run the (optionally filtered) schedule and return the logged table.

    Assumes Genesis is already initialized (see :func:`headless_gs_init`).
    """
    env, _ = build_env(env_overrides)
    expected_hz = 1.0 / env.control_dt
    if abs(expected_hz - schedule.control_hz) > 1e-6:
        env.close()
        raise ValueError(
            f"schedule control_hz={schedule.control_hz} != env {expected_hz:.3f} Hz "
            "(sim_dt * control_interval mismatch)"
        )
    rows: list[dict] = []
    try:
        for man in schedule.maneuvers:
            if maneuver_ids is not None and man.id not in maneuver_ids:
                continue
            rows.extend(_run_maneuver(env, man, schedule.control_hz))
    finally:
        env.close()
    return ProfileTable.from_rows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genesis open-loop maneuver profiler")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE_YAML)
    parser.add_argument(
        "--maneuvers", default=None, help="comma-separated subset of maneuver ids"
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="JSON dict of config.py env overrides (e.g. '{\"f_drive_max\": 18}')",
    )
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    args = parser.parse_args(argv)

    schedule = load_schedule(args.profile)
    settings = load_settings()
    maneuver_ids = args.maneuvers.split(",") if args.maneuvers else None
    env_overrides = json.loads(args.overrides) if args.overrides else None

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    headless_gs_init(backend)

    table = run_profile(schedule, env_overrides, maneuver_ids)

    out = run_dir(args.run_id, create=True)
    csv_path = os.path.join(out, "profile_genesis.csv")
    table.to_csv(csv_path)

    summary = summarize(table, schedule, float(settings["v_fit_min"]))
    summary["env_overrides"] = env_overrides or {}
    summary_path = os.path.join(out, "summary_genesis.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[profile_genesis] wrote {csv_path} ({len(table)} rows)")
    print(f"[profile_genesis] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
