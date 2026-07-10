"""Policy-lap comparison between Genesis and the real car (high-speed segments).

Three subcommands:
  * ``profile-genesis`` rolls out a checkpoint in Genesis and logs a lap trace.
  * ``parse-irl`` turns a lap rosbag into the same lap trace schema.
  * ``compare`` scores speed and lateral-error agreement, gated to ``lap_v_gate``.

This is the end-to-end (closed-loop) validation that complements the open-loop
maneuver fit. It is only meaningful once the real track centerline is aligned to
the localization frame (see ``track_align.py`` / Phase 0), so the IRL side is run
on the car after that gate passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from . import run_dir  # noqa: E402
from .maneuvers import load_settings  # noqa: E402

LAP_COLUMNS = ["t", "s", "speed", "lateral_error", "throttle", "steer", "source"]
# Index of signed lateral error inside the deployed observation vector.
OBS_LATERAL_ERROR_IDX = 10


def _write_lap(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_lap(path: str) -> dict[str, np.ndarray]:
    cols = {c: [] for c in LAP_COLUMNS}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            for c in LAP_COLUMNS:
                v = row.get(c, "")
                if c == "source":
                    cols[c].append(v)
                else:
                    cols[c].append(float(v) if v not in ("", None) else math.nan)
    return {
        c: (np.asarray(cols[c], dtype=np.float64) if c != "source"
            else np.asarray(cols[c], dtype=object))
        for c in LAP_COLUMNS
    }


def profile_genesis(run_id: str, ckpt: str, steps: int, opponent: str,
                    opponent_ckpt: str | None, precision: str) -> str:
    import genesis as gs
    import torch
    from pathlib import Path

    from scripts.eval_record import build_cfg, load_actor_and_norm

    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        precision=precision,
        performance_mode=True,
    )
    device = gs.device

    from f1tenth_env import F1tenthEnv

    cfg = build_cfg(opponent, opponent_ckpt=opponent_ckpt)
    actor, norm_mean, norm_var, step = load_actor_and_norm(Path(ckpt), cfg, device)
    env = F1tenthEnv(
        num_envs=1,
        env_cfg={"launch_strategy": "uniform_jittered",
                 "launch_strategy_data": {"num_cars": 1}, **cfg["env"]},
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
    )
    control_interval = int(cfg["env"].get("control_interval", 10))
    clip = float(cfg["env"]["clip_actions"])
    dt = env.control_dt

    obs, _ = env.reset()
    rows: list[dict] = []
    s = 0.0
    px = float(env.base_pos[0, 0].item())
    py = float(env.base_pos[0, 1].item())
    for i in range(steps):
        model_obs = obs.to(dtype=torch.float32, device=device)
        if norm_mean is not None:
            model_obs = torch.clamp(
                (model_obs - norm_mean) / torch.sqrt(norm_var + 1e-8), -10.0, 10.0
            )
        with torch.no_grad():
            action, _ = actor(model_obs, deterministic=True, with_logprob=False)
        action = torch.clamp(action, -clip, clip).to(dtype=obs.dtype, device=obs.device)
        obs, _, _, extras = env.step(action, n_steps=control_interval)

        nx = float(env.base_pos[0, 0].item())
        ny = float(env.base_pos[0, 1].item())
        s += math.hypot(nx - px, ny - py)
        px, py = nx, ny
        metrics = extras.get("metrics", {})
        speed = _scalar(metrics.get("speed_xy"))
        lat = _scalar(metrics.get("lateral_error"))
        rows.append({
            "t": i * dt, "s": s, "speed": speed, "lateral_error": lat,
            "throttle": float(action[0, 0].item()),
            "steer": float(action[0, 1].item()), "source": "genesis",
        })
    env.close()

    out = run_dir(run_id, create=True)
    path = os.path.join(out, "lap_genesis.csv")
    _write_lap(path, rows)
    print(f"[lap_compare] wrote {path} ({len(rows)} steps, ckpt step={step})")
    return path


def _scalar(value) -> float:
    if value is None:
        return math.nan
    try:
        return float(value[0]) if hasattr(value, "__len__") else float(value)
    except (TypeError, ValueError):
        try:
            return float(value.reshape(-1)[0])
        except Exception:
            return math.nan


def parse_irl(run_id: str, bag_path: str) -> str:
    from .parse_bag import _read_bag, _zoh

    topics = _read_bag(bag_path, extra_topics=("/rl/observation",))
    # Master clock: /rl/observation if present, else /odom.
    obs_series = topics.get("/rl/observation") or []
    if not obs_series:
        # parse_bag does not read /rl/observation by default; pull odom timeline.
        odom = topics["/odom"]
        clock = [(t, None) for t, _ in odom]
    else:
        clock = obs_series

    rows: list[dict] = []
    s = 0.0
    px = py = None
    t0 = None
    for t, obs_msg in clock:
        if t0 is None:
            t0 = t
        odom = _zoh(topics["/odom"], t)
        pose = _zoh(topics["/pf/pose/odom"], t)
        drive = _zoh(topics["/drive"], t)
        speed = math.nan
        if odom is not None:
            speed = math.hypot(odom.twist.twist.linear.x, odom.twist.twist.linear.y)
        if pose is not None:
            x = float(pose.pose.pose.position.x)
            y = float(pose.pose.pose.position.y)
            if px is not None:
                s += math.hypot(x - px, y - py)
            px, py = x, y
        lat = math.nan
        if obs_msg is not None and len(obs_msg.data) > OBS_LATERAL_ERROR_IDX:
            lat = float(obs_msg.data[OBS_LATERAL_ERROR_IDX])
        throttle = steer = math.nan
        rows.append({
            "t": t - t0, "s": s, "speed": speed, "lateral_error": lat,
            "throttle": throttle, "steer": steer, "source": "irl",
        })

    out = run_dir(run_id, create=True)
    path = os.path.join(out, "lap_irl.csv")
    _write_lap(path, rows)
    print(f"[lap_compare] wrote {path} ({len(rows)} samples)")
    return path


def compare(run_id: str, v_gate: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run_dir(run_id, create=True)
    sim = _read_lap(os.path.join(out, "lap_genesis.csv"))
    irl_path = os.path.join(out, "lap_irl.csv")
    irl = _read_lap(irl_path) if os.path.isfile(irl_path) else None

    def gated_stats(lap):
        m = lap["speed"] >= v_gate
        lat = np.abs(lap["lateral_error"][m])
        lat = lat[np.isfinite(lat)]
        spd = lap["speed"][m]
        spd = spd[np.isfinite(spd)]
        return {
            "n_gated": int(m.sum()),
            "mean_speed": float(np.mean(spd)) if spd.size else math.nan,
            "mean_abs_lat_err": float(np.mean(lat)) if lat.size else math.nan,
            "p95_abs_lat_err": float(np.percentile(lat, 95)) if lat.size else math.nan,
        }

    report = {"v_gate": v_gate, "genesis": gated_stats(sim)}
    if irl is not None:
        report["irl"] = gated_stats(irl)
        irl_lat = report["irl"]["mean_abs_lat_err"]
        sim_lat = report["genesis"]["mean_abs_lat_err"]
        if math.isfinite(irl_lat) and irl_lat > 1e-6:
            report["lat_err_ratio"] = sim_lat / irl_lat
            report["passed"] = report["lat_err_ratio"] <= 2.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(sim["s"], sim["speed"], label="genesis")
    if irl is not None:
        ax[0].plot(irl["s"], irl["speed"], label="irl")
    ax[0].axhline(v_gate, color="0.6", ls="--", lw=0.8)
    ax[0].set(xlabel="arc length s (m)", ylabel="speed (m/s)", title="lap speed")
    ax[0].legend(fontsize=8)
    ax[1].plot(sim["s"], np.abs(sim["lateral_error"]), label="genesis")
    if irl is not None:
        ax[1].plot(irl["s"], np.abs(irl["lateral_error"]), label="irl")
    ax[1].set(xlabel="arc length s (m)", ylabel="|lateral error| (m)", title="lap tracking")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(out, "plots", "lap_compare.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)

    report_path = os.path.join(out, "lap_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[lap_compare] plot -> {plot_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy-lap sim vs IRL comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("profile-genesis", help="rollout a checkpoint in Genesis")
    pg.add_argument("--run-id", required=True)
    pg.add_argument("--ckpt", required=True)
    pg.add_argument("--steps", type=int, default=600)
    pg.add_argument("--opponent", default="none", choices=["none", "scripted", "policy"])
    pg.add_argument("--opponent-ckpt", default=None)
    pg.add_argument("--precision", default="32", choices=["32", "64"])

    pi = sub.add_parser("parse-irl", help="parse a lap rosbag")
    pi.add_argument("--run-id", required=True)
    pi.add_argument("bag")

    cp = sub.add_parser("compare", help="score gated speed / lateral error")
    cp.add_argument("--run-id", required=True)
    cp.add_argument("--v-gate", type=float, default=None)

    args = parser.parse_args(argv)
    settings = load_settings()

    if args.cmd == "profile-genesis":
        profile_genesis(args.run_id, args.ckpt, args.steps, args.opponent,
                        args.opponent_ckpt, args.precision)
    elif args.cmd == "parse-irl":
        parse_irl(args.run_id, args.bag)
    elif args.cmd == "compare":
        v_gate = args.v_gate if args.v_gate is not None else float(settings["lap_v_gate"])
        compare(args.run_id, v_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
