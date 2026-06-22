#!/usr/bin/env python3
"""Evaluate a trained checkpoint in Genesis and record an MP4 of the rollout.

Loads a ``standalone_trainer`` checkpoint (``{"step", "actor", ..., "obs_norm"}``),
runs a deterministic (exploit) rollout with the camera following the ego car, and
writes a video to ``outputs/eval_videos/``.

The 1v1 checkpoints (e.g. ``iv2026_1v1_500k_v1``) were trained with the opponent
observation block enabled (num_obs = 380 + opponent_obs_dim), so the opponent is
on by default to match the observation layout the policy expects.

Usage:
    python scripts/eval_record.py \
        --ckpt outputs/standalone/iv2026_1v1_500k_v1/ckpt_500000.pt \
        --steps 300
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import genesis as gs
import rerun as rr
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from qrsac import SquashedGaussianMLPActor  # noqa: E402


def build_cfg(opponent: str, opponent_ckpt: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if opponent != "none":
        cfg["env"]["opponent_strategy"] = opponent
        if opponent == "policy":
            if not opponent_ckpt:
                raise ValueError("--opponent-ckpt is required when --opponent policy")
            cfg["env"]["opponent_ckpt"] = opponent_ckpt
        cfg["obs"]["enable_opponent_obs"] = True
        cfg["obs"]["num_obs"] = 380 + int(cfg["obs"]["opponent_obs_dim"])
        cfg["reward"]["reward_scales"]["passing"] = 0.5
    return cfg


def load_actor_and_norm(ckpt_path: Path, cfg: dict, device: torch.device):
    actor = SquashedGaussianMLPActor(
        obs_dim=cfg["obs"]["num_obs"],
        act_dim=cfg["env"]["num_actions"],
        hidden_sizes=cfg["model"]["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor"])
    actor.to(device=device, dtype=torch.float32).eval()

    mean = var = None
    if "obs_norm" in payload:
        mean = payload["obs_norm"]["mean"].to(device=device, dtype=torch.float32)
        var = payload["obs_norm"]["var"].to(device=device, dtype=torch.float32)
    return actor, mean, var, int(payload.get("step", -1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Genesis eval + video recorder")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/standalone/iv2026_1v1_500k_v1/ckpt_500000.pt",
    )
    parser.add_argument("--steps", type=int, default=300, help="control steps to roll out")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--opponent",
        type=str,
        default="scripted",
        choices=["scripted", "policy", "none"],
        help="match the checkpoint's training setup (1v1 -> scripted, self-play -> policy)",
    )
    parser.add_argument(
        "--opponent-ckpt",
        type=str,
        default=None,
        help="frozen policy checkpoint for --opponent policy (self-play eval)",
    )
    parser.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"[eval_record] checkpoint not found: {ckpt_path}")
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else Path("outputs/eval_videos") / f"{ckpt_path.parent.name}_{ckpt_path.stem}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # In-memory rerun recording so the env's per-frame rr.log() calls have a sink
    # (no viewer spawned). The MP4 itself comes from the Genesis camera, not rerun.
    rr.init("f1tenth-eval-record", spawn=False)

    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        precision=args.precision,
        performance_mode=True,
    )
    device = gs.device

    if args.opponent == "policy" and not args.opponent_ckpt:
        print("[eval_record] --opponent policy requires --opponent-ckpt")
        return 1

    cfg = build_cfg(args.opponent, opponent_ckpt=args.opponent_ckpt)
    actor, norm_mean, norm_var, step = load_actor_and_norm(ckpt_path, cfg, device)
    print(f"[eval_record] loaded checkpoint step={step} obs_dim={cfg['obs']['num_obs']}")
    if args.opponent == "policy":
        opp_path = Path(args.opponent_ckpt)
        print(f"[eval_record] policy opponent: {opp_path.name}")

    env = F1tenthEnv(
        env_cfg={
            "launch_strategy": "uniform_jittered",
            "launch_strategy_data": {"num_cars": args.num_envs},
            **cfg["env"],
        },
        num_envs=args.num_envs,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=True,
    )

    control_interval = int(cfg["env"].get("control_interval", 10))
    clip = float(cfg["env"]["clip_actions"])

    obs, _ = env.reset()
    print(
        f"[eval_record] rolling out {args.steps} control steps "
        f"({args.steps * control_interval} rendered frames)..."
    )
    for i in range(args.steps):
        model_obs = obs.to(dtype=torch.float32, device=device)
        if norm_mean is not None:
            model_obs = torch.clamp(
                (model_obs - norm_mean) / torch.sqrt(norm_var + 1e-8), -10.0, 10.0
            )
        with torch.no_grad():
            action, _ = actor(model_obs, deterministic=True, with_logprob=False)
        action = torch.clamp(action, -clip, clip).to(dtype=obs.dtype, device=obs.device)
        obs, _, _, _ = env.step(action, n_steps=control_interval)
        if (i + 1) % 50 == 0:
            print(f"[eval_record]   step {i + 1}/{args.steps}")

    print(f"[eval_record] saving video to {out_path} ...")
    env.cam1.stop_recording(save_to_filename=str(out_path), fps=args.fps)
    env.close()
    print(f"[eval_record] done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
