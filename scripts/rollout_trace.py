#!/usr/bin/env python3
"""Headless 1v0 rollout that logs the driven path and overlays it on the centerline.

Faster than eval_record (no camera/video); used to confirm a policy completes a
full lap of an extracted centerline.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import genesis as gs
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from qrsac import SquashedGaussianMLPActor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponent", default="none", choices=["none", "scripted"])
    args = ap.parse_args()

    gs.init(backend=gs.cpu, precision="32", performance_mode=True)
    device = gs.device
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    if args.opponent != "none":
        cfg["env"]["opponent_strategy"] = args.opponent
        cfg["obs"]["enable_opponent_obs"] = True
        cfg["obs"]["num_obs"] = 380 + int(cfg["obs"]["opponent_obs_dim"])

    actor = SquashedGaussianMLPActor(
        obs_dim=cfg["obs"]["num_obs"],
        act_dim=cfg["env"]["num_actions"],
        hidden_sizes=cfg["model"]["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor"])
    actor.to(device=device, dtype=torch.float32).eval()
    mean = payload["obs_norm"]["mean"].to(device, torch.float32)
    var = payload["obs_norm"]["var"].to(device, torch.float32)

    env = F1tenthEnv(
        env_cfg={
            "launch_strategy": "uniform_jittered",
            "launch_strategy_data": {"num_cars": 1},
            **cfg["env"],
        },
        num_envs=1,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    ci = int(cfg["env"].get("control_interval", 10))
    clip = float(cfg["env"]["clip_actions"])

    obs, _ = env.reset()
    has_opp = env.opponent is not None
    trace = []
    opp_trace = []
    for _ in range(args.steps):
        mo = obs.to(dtype=torch.float32, device=device)
        mo = torch.clamp((mo - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
        with torch.no_grad():
            action, _ = actor(mo, deterministic=True, with_logprob=False)
        action = torch.clamp(action, -clip, clip).to(dtype=obs.dtype, device=obs.device)
        obs, _, _, _ = env.step(action, n_steps=ci)
        trace.append(env.base_pos[0, :2].detach().cpu().numpy().copy())
        if has_opp:
            opp_trace.append(env.opp_base_pos[0, :2].detach().cpu().numpy().copy())
    env.close()

    trace = np.asarray(trace)
    opp_trace = np.asarray(opp_trace) if has_opp else None
    np.savetxt(Path(args.out).with_suffix(".csv"), trace, delimiter=",", header="x_m,y_m")
    cl = np.genfromtxt(args.track, delimiter=",", names=True, dtype=np.float64)
    cx, cy = cl["x_m"], cl["y_m"]

    # break the polyline at episode resets (large one-step teleports) so respawn
    # chords are not drawn as if the car drove across the track.
    jumps = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    n_resets = int(np.sum(jumps > 2.0))
    seg = trace.copy().astype(np.float64)
    breaks = np.where(jumps > 2.0)[0]
    seg_plot = np.insert(seg, breaks + 1, np.nan, axis=0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ego_label = "ego (1v1)" if opp_trace is not None else "1v0 driven path"
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), "k--", lw=1.2, label="centerline")
    ax.plot(seg_plot[:, 0], seg_plot[:, 1], "-", color="#1f77b4", lw=2,
            label=f"{ego_label} (resets removed)")
    if opp_trace is not None:
        ojumps = np.linalg.norm(np.diff(opp_trace, axis=0), axis=1)
        obreaks = np.where(ojumps > 2.0)[0]
        opp_plot = np.insert(opp_trace.astype(np.float64), obreaks + 1, np.nan, axis=0)
        ax.plot(opp_plot[:, 0], opp_plot[:, 1], "-", color="#d62728", lw=2,
                label="scripted opponent (mobile)")
    ax.scatter([trace[0, 0]], [trace[0, 1]], c="lime", s=70, zorder=5, label="ego start")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    mode = "1v1 vs mobile opponent" if opp_trace is not None else "1v0"
    ax.set_title(f"{mode} on f1tenth_map ({args.steps} steps, {n_resets} ego resets)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  (path pts={len(trace)}, resets={n_resets})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
