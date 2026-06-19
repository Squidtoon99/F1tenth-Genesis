#!/usr/bin/env python3
"""Benchmark F1tenthEnv.step() throughput with a per-section breakdown.

Backend-agnostic (CUDA / Metal / CPU). The section breakdown is collected by
wrapping the env's step sub-methods and ``scene.step`` with timing shims, so it
requires no changes to the environment code.

Examples:
    python scripts/bench_env.py --backend auto --num-envs 1024 --steps 200
    python scripts/bench_env.py --num-envs 256 1024 4096 --steps 100
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy

import genesis as gs
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv


def _make_sync():
    """Return a callable that blocks until queued device work finishes."""
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.mps.synchronize
    return lambda: None


def _select_backend(name: str):
    if name == "cpu":
        return gs.cpu
    if name == "gpu":
        return gs.gpu
    if name == "metal":
        return getattr(gs, "metal", gs.gpu)
    if name == "cuda":
        return getattr(gs, "cuda", gs.gpu)
    # auto
    return gs.gpu if torch.cuda.is_available() else gs.cpu


def _patch_headless_rasterizer() -> None:
    import pyglet
    from genesis.vis.rasterizer import Rasterizer

    pyglet.options["headless"] = True

    def _headless_build(self):
        if self._context is None:
            return
        self.visualizer = self._context.visualizer

    Rasterizer.build = _headless_build


class SectionTimer:
    """Accumulates wall-clock time per labeled section (device-synced)."""

    def __init__(self, sync):
        self._sync = sync
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.enabled = False

    def wrap_method(self, obj, attr: str, label: str) -> None:
        fn = getattr(obj, attr)

        @functools.wraps(fn)
        def shim(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)
            self._sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self._sync()
            self.totals[label] += time.perf_counter() - t0
            self.counts[label] += 1
            return out

        setattr(obj, attr, shim)


def build_env(num_envs: int) -> F1tenthEnv:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = cfg["env"]
    env_cfg.update(
        {
            "launch_strategy": "uniform_jittered",
            "launch_strategy_data": {"num_cars": num_envs},
        }
    )
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
    )


def bench_once(
    num_envs: int,
    warmup: int,
    steps: int,
    control_interval: int,
    sync,
) -> dict[str, float]:
    env = build_env(num_envs)

    timer = SectionTimer(sync)
    # Wrap the step sub-methods and the physics step for a breakdown.
    timer.wrap_method(env, "_apply_actions", "apply_actions")
    timer.wrap_method(env, "_compute_dissipative_force", "drag_compute")
    timer.wrap_method(env, "_apply_dissipative_force", "drag_apply")
    timer.wrap_method(env.scene, "step", "scene.step")
    timer.wrap_method(env, "_update_state_buffers", "state_buffers")
    timer.wrap_method(env, "_compute_rewards", "rewards")
    timer.wrap_method(env, "_compute_terminations", "terminations")
    timer.wrap_method(env, "_update_observation", "observation")
    timer.wrap_method(env, "reset", "reset")

    act_dim = env.num_actions
    actions = torch.zeros((num_envs, act_dim), dtype=gs.tc_float, device=gs.device)

    def random_actions() -> torch.Tensor:
        actions.uniform_(-1.0, 1.0)
        return actions

    for _ in range(warmup):
        env.step(random_actions(), n_steps=control_interval)

    sync()
    timer.enabled = True
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(random_actions(), n_steps=control_interval)
    sync()
    elapsed = time.perf_counter() - t0
    timer.enabled = False

    env.close()

    control_steps_per_s = steps / elapsed
    env_steps_per_s = control_steps_per_s * num_envs
    sim_frames_per_s = env_steps_per_s * control_interval

    return {
        "num_envs": num_envs,
        "elapsed_s": elapsed,
        "control_steps_per_s": control_steps_per_s,
        "env_steps_per_s": env_steps_per_s,
        "sim_frames_per_s": sim_frames_per_s,
        "section_totals": dict(timer.totals),
    }


def print_result(res: dict[str, float], steps: int) -> None:
    n = res["num_envs"]
    print(f"\n=== num_envs={n} ===")
    print(f"wall={res['elapsed_s']:.3f}s for {steps} control steps")
    print(f"control_steps/s = {res['control_steps_per_s']:.1f}")
    print(f"env_steps/s     = {res['env_steps_per_s']:,.0f}")
    print(f"sim_frames/s    = {res['sim_frames_per_s']:,.0f}")
    totals = res["section_totals"]
    total_time = sum(totals.values()) or 1e-9
    print("section breakdown (% of measured step time):")
    for label, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {label:16s} {t:8.3f}s  {100.0 * t / total_time:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark F1tenthEnv throughput")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "metal"],
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    _patch_headless_rasterizer()
    gs.init(
        backend=_select_backend(args.backend),
        logging_level="warning",
        performance_mode=True,
    )
    sync = _make_sync()
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])

    print(f"backend={args.backend} device={gs.device} "
          f"control_interval={control_interval} steps={args.steps}")

    for num_envs in args.num_envs:
        res = bench_once(
            num_envs=num_envs,
            warmup=args.warmup,
            steps=args.steps,
            control_interval=control_interval,
            sync=sync,
        )
        print_result(res, args.steps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
