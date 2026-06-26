#!/usr/bin/env python3
"""In-sim assertion harness for the 1v1 racing feature (autonomous gate).

Runs a short Genesis rollout and hard-fails (non-zero exit) on any of:
- wrong observation shape (must be 380 + opponent_obs_dim when the opponent is on,
  and exactly 380 when it is off / 1v0),
- non-finite observations or rewards,
- the passing reward term missing from the reward breakdown,
- a discontinuous (spiky) per-step passing reward,
- zero reachable collisions over the whole 1v1 rollout (collisions must at least
  be possible so the termination is exercised).

It also dumps a per-term reward-vs-step plot to ``outputs/verify_1v1/`` when
matplotlib is available. This script is the smoke gate referenced by the plan and
is intended to be run in the Genesis environment (CPU is fine for the smoke run;
GPU for speed).

Usage:
    python scripts/verify_1v1.py --num-envs 16 --steps 400
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import genesis as gs
import torch

# Repo root on path so "config" / "f1tenth_env" import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402


def build_cfg(opponent: str, num_envs: int) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if opponent != "none":
        cfg["env"]["opponent_strategy"] = opponent
        if opponent == "mixed":
            # 50/50 so both behaviors are exercised within a small smoke batch.
            cfg["env"]["opponent_mix"] = {
                "scripted_weight": 0.5,
                "policy_weight": 0.5,
            }
        cfg["obs"]["enable_opponent_obs"] = True
        cfg["obs"]["num_obs"] = 380 + int(cfg["obs"]["opponent_obs_dim"])
        # Smoke wants the collision-termination path to be deterministically
        # reachable, so terminate on any overlap regardless of closing speed
        # (the speed-gated default is exercised by unit tests instead).
        cfg["env"]["collision_term_speed_mps"] = 0.0
        cfg["reward"]["reward_scales"]["passing"] = 0.5
        # Activate the GT Sophy rear-end penalty so the smoke can confirm it fires
        # when the chase ego rear-ends the slower opponent ahead.
        cfg["reward"]["reward_scales"]["rear_end"] = 1.0
    return cfg


def make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _actions(policy: str, num_envs: int, num_actions: int, clip: float, device) -> torch.Tensor:
    if policy == "chase":
        # Deterministically close the spawn gap: full throttle, hold heading. This
        # exercises the collision-termination path (the ego rear-ends the slower
        # scripted opponent), which a zero-mean random policy almost never does.
        act = torch.zeros(num_envs, num_actions, device=device)
        act[:, 0] = clip
        return act
    return torch.rand(num_envs, num_actions, device=device) * 2 * clip - clip


def run(
    opponent: str,
    num_envs: int,
    steps: int,
    control_interval: int,
    ego_policy: str = "chase",
) -> dict:
    cfg = build_cfg(opponent, num_envs)
    expected_obs = int(cfg["obs"]["num_obs"])
    env = make_env(cfg, num_envs)
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, expected_obs), (
            f"obs shape {tuple(obs.shape)} != {(num_envs, expected_obs)}"
        )

        clip = float(cfg["env"]["clip_actions"])
        collisions = 0
        passing_series: list[float] = []
        term_series: dict[str, list[float]] = {}
        any_nonfinite = False
        passing_present = opponent == "none"  # not expected when off
        rear_end_present = False
        rear_end_min = 0.0
        scripted_speed_max = 0.0
        policy_rows_seen = False

        for _ in range(steps):
            actions = _actions(
                ego_policy, num_envs, cfg["env"]["num_actions"], clip, obs.device
            )
            obs, reward, done, extras = env.step(
                actions.to(gs.tc_float), n_steps=control_interval
            )
            if not (torch.isfinite(obs).all() and torch.isfinite(reward).all()):
                any_nonfinite = True

            terms = extras.get("rewards", {}).get("terms", {})
            if "passing" in terms:
                passing_present = True
                passing_series.append(float(terms["passing"].mean()))
            if "rear_end" in terms:
                rear_end_present = True
                rear_end_min = min(rear_end_min, float(terms["rear_end"].min()))
            for name, val in extras.get("termination", {}).items():
                if torch.is_tensor(val):
                    term_series.setdefault(name, []).append(float(val.sum()))
            collisions += int(
                extras.get("termination", {}).get(
                    "collision", torch.zeros(1)
                ).sum()
            )

            # Mixed population: split opponent speed by the per-row mode so we can
            # confirm scripted rows actually drive and policy rows stay valid.
            if opponent == "mixed":
                mode_buf = getattr(env.opponent_ctrl, "mode_buf", None)
                opp_speed = extras.get("metrics", {}).get("opp_speed")
                if mode_buf is not None and opp_speed is not None:
                    scripted_rows = ~mode_buf
                    if scripted_rows.any():
                        scripted_speed_max = max(
                            scripted_speed_max,
                            float(opp_speed[scripted_rows].max().item()),
                        )
                    if bool(mode_buf.any()):
                        policy_rows_seen = True

        result = {
            "expected_obs": expected_obs,
            "obs_shape": tuple(obs.shape),
            "nonfinite": any_nonfinite,
            "passing_present": passing_present,
            "collisions": collisions,
            "passing_series": passing_series,
            "term_series": term_series,
            "ego_policy": ego_policy,
            "rear_end_present": rear_end_present,
            "rear_end_min": rear_end_min,
            "scripted_speed_max": scripted_speed_max,
            "policy_rows_seen": policy_rows_seen,
        }
        return result
    finally:
        env.close()


def check(result: dict, opponent: str) -> list[str]:
    failures: list[str] = []
    if result["obs_shape"][1] != result["expected_obs"]:
        failures.append(
            f"obs shape {result['obs_shape']} != expected dim {result['expected_obs']}"
        )
    if result["nonfinite"]:
        failures.append("non-finite obs/reward encountered")
    if opponent != "none":
        if not result["passing_present"]:
            failures.append("passing reward term missing from reward breakdown")
        # Only the deterministic chase policy is guaranteed to close the spawn gap; a
        # random policy almost never collides, so don't assert collisions there.
        if result.get("ego_policy") == "chase" and result["collisions"] == 0:
            failures.append("no collisions reachable over the rollout (term not exercised)")
        series = result["passing_series"]
        if len(series) > 2:
            max_jump = max(abs(b - a) for a, b in zip(series[:-1], series[1:]))
            if max_jump > 5.0:
                failures.append(f"passing reward discontinuous (max step jump {max_jump:.3f})")
        # Rr smoke: a chase ego that collides with the slower opponent ahead must
        # trigger the rear-end penalty at least once.
        if result.get("ego_policy") == "chase" and result["collisions"] > 0:
            if not result.get("rear_end_present"):
                failures.append("rear_end reward term missing from reward breakdown")
            elif result.get("rear_end_min", 0.0) >= 0.0:
                failures.append("rear_end penalty never fired on a chase-collision step")
    if opponent == "mixed":
        if not result.get("policy_rows_seen"):
            failures.append("mixed: no policy-mode opponent rows were assigned")
        if result.get("scripted_speed_max", 0.0) <= 1.0:
            failures.append("mixed: scripted-mode opponents never reached >1 m/s")
    return failures


def maybe_plot(result: dict, opponent: str) -> None:
    if opponent == "none" or not result["passing_series"]:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir = Path("outputs/verify_1v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.plot(result["passing_series"], label="passing (mean)")
    plt.xlabel("control step")
    plt.ylabel("reward term")
    plt.title("1v1 passing reward over a smoke rollout")
    plt.legend()
    path = out_dir / "passing_reward.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"[verify_1v1] wrote reward plot to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="1v1 in-sim verification harness")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    parser.add_argument(
        "--opponent",
        type=str,
        default="scripted",
        choices=["scripted", "mixed"],
        help="Opponent strategy to verify: 'scripted' (default) or 'mixed' "
        "(per-row scripted + policy population).",
    )
    parser.add_argument(
        "--skip-1v0", action="store_true", help="skip the 1v0 regression shape check"
    )
    parser.add_argument(
        "--ego-policy",
        type=str,
        default="chase",
        choices=["chase", "random"],
        help="chase = deterministic full-throttle (exercises collision term); "
        "random = stochastic finiteness sweep (no collision assertion)",
    )
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        precision=args.precision,
        performance_mode=True,
    )

    all_failures: list[str] = []

    print(
        f"[verify_1v1] running 1v1 ({args.opponent} opponent, ego={args.ego_policy}) ..."
    )
    res_1v1 = run(
        args.opponent, args.num_envs, args.steps, args.control_interval, args.ego_policy
    )
    print(
        f"[verify_1v1] 1v1: {res_1v1['obs_shape']=} collisions={res_1v1['collisions']} "
        f"rear_end_min={res_1v1.get('rear_end_min', 0.0):.3f}"
    )
    if args.opponent == "mixed":
        print(
            f"[verify_1v1] mixed: scripted_speed_max={res_1v1['scripted_speed_max']:.3f} "
            f"policy_rows_seen={res_1v1['policy_rows_seen']}"
        )
    all_failures += [f"[1v1] {m}" for m in check(res_1v1, args.opponent)]
    maybe_plot(res_1v1, args.opponent)

    if not args.skip_1v0:
        print("[verify_1v1] running 1v0 (no opponent) regression shape check ...")
        res_1v0 = run(
            "none", args.num_envs, min(args.steps, 100), args.control_interval, "random"
        )
        print(f"[verify_1v1] 1v0: {res_1v0['obs_shape']=}")
        if res_1v0["obs_shape"][1] != 380:
            all_failures.append(f"[1v0] obs dim {res_1v0['obs_shape'][1]} != 380")
        all_failures += [f"[1v0] {m}" for m in check(res_1v0, "none")]

    if all_failures:
        print("\n[verify_1v1] FAILED:")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print("\n[verify_1v1] PASS: all 1v1 verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
