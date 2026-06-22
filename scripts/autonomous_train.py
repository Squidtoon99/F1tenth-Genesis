#!/usr/bin/env python3
"""Overnight autonomous training orchestrator for standalone_trainer.py."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collapse_detector import (  # noqa: E402
    CollapseVerdict,
    count_warnings,
    evaluate_at_checkpoint,
    parse_log_metrics,
    reward_plateau,
)
from run_layout import (  # noqa: E402
    ORCHESTRATOR_META_DIR,
    default_run_dir,
    run_log_path,
)

CHECKPOINT_STEP = 100_000
POLL_INTERVAL_S = 60
TRAINER_SCRIPT = ROOT / "standalone_trainer.py"
HYPOTHESES_PATH = ROOT / "scripts" / "train_hypotheses.yaml"


@dataclass
class RunRecord:
    hypothesis_key: str
    hypothesis_id: str
    run_id: str
    status: str
    collapsed_at_100k: bool = False
    completed_500k: bool = False
    had_nan_warnings: bool = False
    had_oom: bool = False
    collapse_reasons: list[str] = field(default_factory=list)
    metrics_at_100k: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    duration_s: float = 0.0


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("autonomous_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_hypotheses(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("defaults", {}), data.get("hypotheses", {})


def trainer_processes() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "standalone_trainer.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.strip().splitlines():
        if line.strip().isdigit():
            pids.append(int(line.strip()))
    return pids


def wait_for_trainer(log: logging.Logger, poll_s: int = 30) -> None:
    log.info("Waiting for any running standalone_trainer.py to exit...")
    while True:
        pids = trainer_processes()
        if not pids:
            log.info("No trainer processes running.")
            return
        log.info("Trainer still running (pids=%s); sleeping %ds", pids, poll_s)
        time.sleep(poll_s)


def merge_run_config(
    defaults: dict[str, Any],
    hypothesis: dict[str, Any],
    *,
    opponent: str = "scripted",
    self_play: bool = True,
) -> dict[str, Any]:
    """Merge YAML defaults + hypothesis args.

    Opponent precedence: hypothesis args > orchestrator CLI > YAML defaults > scripted.
    Self-play precedence: hypothesis args > orchestrator CLI > default True.
    """
    hyp_args = hypothesis.get("args", {}) or {}
    args = {**defaults, **hyp_args}
    if "opponent" in hyp_args:
        args["opponent"] = hyp_args["opponent"]
    else:
        args["opponent"] = opponent
    if "self_play" in hyp_args:
        args["self_play"] = bool(hyp_args["self_play"])
    else:
        args["self_play"] = self_play
    return args


def build_trainer_cmd(
    hypothesis: dict[str, Any],
    defaults: dict[str, Any],
    run_id: str,
    wandb_group: str,
    *,
    opponent: str = "scripted",
    self_play: bool = True,
) -> list[str]:
    args = merge_run_config(defaults, hypothesis, opponent=opponent, self_play=self_play)
    cmd = [
        sys.executable,
        str(TRAINER_SCRIPT),
        "--run-id",
        run_id,
        "--wandb-group",
        wandb_group,
        "--hypothesis",
        hypothesis.get("hypothesis", hypothesis.get("id", run_id)),
    ]

    flag_map = {
        "num_envs": "--num-envs",
        "total_steps": "--total-steps",
        "batch_size": "--batch-size",
        "updates_per_step": "--updates-per-step",
        "alpha": "--alpha",
        "min_train_samples": "--min-train-samples",
        "n_step": "--n-step",
        "track": "--track",
        "precision": "--precision",
        "ckpt_interval": "--ckpt-interval",
        "seed": "--seed",
        "buffer_capacity": "--buffer-capacity",
        "log_interval": "--log-interval",
        "wandb_mode": "--wandb-mode",
        "run_dir": "--run-dir",
    }

    for key, flag in flag_map.items():
        if key in args and args[key] is not None:
            cmd.extend([flag, str(args[key])])

    use_self_play = bool(args.get("self_play", self_play))
    if use_self_play:
        cmd.append("--self-play")
        selfplay_flags = {
            "selfplay_snapshot_interval": "--selfplay-snapshot-interval",
            "selfplay_refresh_interval": "--selfplay-refresh-interval",
            "selfplay_pool_size": "--selfplay-pool-size",
            "selfplay_sample": "--selfplay-sample",
            "init_ckpt": "--init-ckpt",
        }
        for key, flag in selfplay_flags.items():
            if key in args and args[key] is not None:
                cmd.extend([flag, str(args[key])])
        if "passing_scale" in args and args["passing_scale"] is not None:
            cmd.extend(["--passing-scale", str(args["passing_scale"])])
    else:
        opponent_mode = str(args.get("opponent", opponent))
        cmd.extend(["--opponent", opponent_mode])
        if opponent_mode != "none":
            optional_opponent_flags = {
                "opponent_target_speed": "--opponent-target-speed",
                "opponent_spawn_gap": "--opponent-spawn-gap",
                "passing_scale": "--passing-scale",
                "opponent_ckpt": "--opponent-ckpt",
            }
            for key, flag in optional_opponent_flags.items():
                if key in args and args[key] is not None:
                    cmd.extend([flag, str(args[key])])

    if args.get("wandb", True):
        cmd.append("--wandb")

    if "run_dir" not in args or args.get("run_dir") is None:
        cmd.extend(["--run-dir", str(default_run_dir(run_id, root=ROOT))])

    return cmd


def latest_logged_step(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"standalone_trainer INFO: step=(\d+) buffer=", text)
    return int(matches[-1]) if matches else None


def run_trainer(
    cmd: list[str],
    log_path: Path,
    log: logging.Logger,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Launching: %s", shlex.join(cmd))
    log.info("Logging to: %s", log_path)

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    proc._log_file = log_file  # type: ignore[attr-defined]
    return proc


def close_proc_log(proc: subprocess.Popen) -> None:
    if hasattr(proc, "_log_file"):
        proc._log_file.close()  # type: ignore[attr-defined]


def monitor_run(
    proc: subprocess.Popen,
    log_path: Path,
    log: logging.Logger,
    checkpoint_step: int = CHECKPOINT_STEP,
) -> tuple[str, CollapseVerdict | None]:
    """Monitor until exit. Returns (status, collapse_verdict_or_none)."""
    gate_evaluated = False
    collapse_verdict: CollapseVerdict | None = None

    while proc.poll() is None:
        time.sleep(POLL_INTERVAL_S)
        step = latest_logged_step(log_path)
        if step is not None:
            log.info("Run progress: step=%d", step)

        if not gate_evaluated and step is not None and step >= checkpoint_step:
            gate_evaluated = True
            log.info("Checkpoint gate at step %d — evaluating health...", checkpoint_step)
            collapse_verdict = evaluate_at_checkpoint(log_path, target_step=checkpoint_step)
            log.info(
                "Collapse verdict: collapsed=%s reasons=%s",
                collapse_verdict.collapsed,
                collapse_verdict.reasons,
            )
            if collapse_verdict.collapsed:
                log.warning("Run collapsed at 100k — terminating trainer.")
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                close_proc_log(proc)
                return "collapsed_at_100k", collapse_verdict

    close_proc_log(proc)
    exit_code = proc.returncode

    if gate_evaluated and collapse_verdict is not None and not collapse_verdict.collapsed:
        final_step = latest_logged_step(log_path) or 0
        if final_step >= 490_000:
            return "completed", collapse_verdict
        if exit_code != 0:
            return "crashed", collapse_verdict
        return "completed", collapse_verdict

    if not gate_evaluated:
        final_step = latest_logged_step(log_path) or 0
        if final_step >= checkpoint_step - 1000:
            collapse_verdict = evaluate_at_checkpoint(log_path, target_step=checkpoint_step)
        if exit_code != 0:
            return "crashed", collapse_verdict
        if final_step >= 490_000:
            return "completed", collapse_verdict
        return "stopped_early", collapse_verdict

    if exit_code != 0:
        return "crashed", collapse_verdict
    return "completed", collapse_verdict


def pick_next_hypothesis(
    history: list[RunRecord],
    hypotheses: dict[str, dict[str, Any]],
    tried: set[str],
    log: logging.Logger,
) -> str | None:
    if not history:
        return "H1"

    last = history[-1]
    cfg = last.config

    if last.status == "collapsed_at_100k":
        alpha = float(cfg.get("alpha", 0.1))
        num_envs = int(cfg.get("num_envs", 256))
        if alpha <= 0.11 and "H2" not in tried:
            log.info("Collapse with alpha=0.1 → trying H2 (explore)")
            return "H2"
        if alpha <= 0.21 and "H3" not in tried:
            log.info("Collapse with alpha=0.2 → trying H3 (stable128)")
            return "H3"
        if num_envs >= 256 and "H3" not in tried:
            log.info("Collapse with 256 envs → trying H3 (stable128)")
            return "H3"
        log.info("All stability levers exhausted.")
        return None

    if last.had_oom:
        num_envs = int(cfg.get("num_envs", 256))
        if num_envs > 64:
            log.info("OOM detected — retry with halved envs (manual fallback to H3-like)")
            if "H3" not in tried:
                return "H3"
        return None

    if last.status == "crashed":
        if "H3" not in tried:
            return "H3"
        return None

    if last.completed_500k:
        hyp_key = last.hypothesis_key
        precision = str(cfg.get("precision", "64"))
        all_steps = last.metrics_at_100k.get("all_steps", [])

        if (
            hyp_key == "H1"
            and precision == "64"
            and not last.had_nan_warnings
            and "H5" not in tried
        ):
            log.info("H1 completed cleanly at fp64 → trying H5 (fp32 speed)")
            return "H5"

        if hyp_key == "H1" and "H4" not in tried:
            log.info("H1 completed → trying H4 (2M buffer)")
            return "H4"

        if hyp_key == "H5" and reward_plateau(all_steps) and "H6" not in tried:
            log.info("H5 reward plateau → trying H6 (2x updates)")
            return "H6"

        if hyp_key == "H5" and "H6" not in tried:
            return "H6"

        log.info("No further hypotheses after successful %s.", hyp_key)
        return None

    return None


def record_from_run(
    hypothesis_key: str,
    hypothesis: dict[str, Any],
    defaults: dict[str, Any],
    run_id: str,
    status: str,
    log_path: Path,
    collapse_verdict: CollapseVerdict | None,
    t0: float,
    *,
    orchestrator_opponent: str = "scripted",
    orchestrator_self_play: bool = True,
) -> RunRecord:
    args = merge_run_config(
        defaults,
        hypothesis,
        opponent=orchestrator_opponent,
        self_play=orchestrator_self_play,
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = (
        collapse_verdict.metrics
        if collapse_verdict is not None
        else parse_log_metrics(log_path, target_step=CHECKPOINT_STEP)
    )

    return RunRecord(
        hypothesis_key=hypothesis_key,
        hypothesis_id=hypothesis.get("id", run_id),
        run_id=run_id,
        status=status,
        collapsed_at_100k=status == "collapsed_at_100k",
        completed_500k=status == "completed",
        had_nan_warnings=count_warnings(log_text) > 0,
        had_oom="CUDA out of memory" in log_text or "OutOfMemoryError" in log_text,
        collapse_reasons=collapse_verdict.reasons if collapse_verdict else [],
        metrics_at_100k=metrics,
        config=args,
        log_path=str(log_path),
        duration_s=time.time() - t0,
    )


def write_summary(history: list[RunRecord], path: Path, wandb_group: str) -> None:
    payload = {
        "wandb_group": wandb_group,
        "runs": [asdict(r) for r in history],
        "completed": sum(1 for r in history if r.completed_500k),
        "collapsed_at_100k": sum(1 for r in history if r.collapsed_at_100k),
        "crashed": sum(1 for r in history if r.status == "crashed"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight autonomous training queue")
    parser.add_argument(
        "--wait-for-trainer",
        action="store_true",
        help="Block until no standalone_trainer.py is running",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=6,
        help="Maximum number of training runs to launch",
    )
    parser.add_argument(
        "--hypotheses",
        type=Path,
        default=HYPOTHESES_PATH,
        help="Path to train_hypotheses.yaml",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group name (default: overnight_YYYY-MM-DD)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=CHECKPOINT_STEP,
        help="Step at which to evaluate collapse",
    )
    parser.add_argument(
        "--start-hypothesis",
        type=str,
        default="H1",
        help="First hypothesis key to run",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="scripted",
        choices=["none", "scripted"],
        help="1v1 opponent mode when --no-self-play (default: scripted). "
        "Override per run via opponent: in train_hypotheses.yaml defaults or "
        "hypothesis args.",
    )
    parser.add_argument(
        "--self-play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable delayed self-play for launched trainers (default: on). "
        "Uses --opponent policy with periodic snapshot refresh. Disable with "
        "--no-self-play to fall back to scripted opponents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print built trainer commands for all hypotheses and exit "
        "(no training launched).",
    )
    return parser.parse_args()


def dry_run_commands(
    hypotheses: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    wandb_group: str,
    *,
    opponent: str = "scripted",
    self_play: bool = True,
) -> None:
    for key, hypothesis in hypotheses.items():
        run_id = hypothesis.get("id", key)
        cmd = build_trainer_cmd(
            hypothesis,
            defaults,
            run_id,
            wandb_group,
            opponent=opponent,
            self_play=self_play,
        )
        print(f"[{key}] {shlex.join(cmd)}")


def main() -> int:
    args = parse_args()
    wandb_group = args.wandb_group or f"overnight_{date.today().isoformat()}"

    OVERNIGHT_DIR = ORCHESTRATOR_META_DIR
    OVERNIGHT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OVERNIGHT_DIR / "orchestrator.log")
    log.info("Starting overnight orchestrator (group=%s, max_runs=%d)", wandb_group, args.max_runs)

    defaults, hypotheses = load_hypotheses(args.hypotheses)
    if not hypotheses:
        log.error("No hypotheses loaded from %s", args.hypotheses)
        return 1

    if args.dry_run:
        dry_run_commands(
            hypotheses,
            defaults,
            wandb_group,
            opponent=args.opponent,
            self_play=args.self_play,
        )
        return 0

    if args.wait_for_trainer:
        wait_for_trainer(log)

    history: list[RunRecord] = []
    tried: set[str] = set()
    next_key: str | None = args.start_hypothesis

    for run_index in range(args.max_runs):
        if next_key is None:
            log.info("No more hypotheses to run.")
            break
        if next_key not in hypotheses:
            log.error("Unknown hypothesis key: %s", next_key)
            break

        tried.add(next_key)
        hypothesis = hypotheses[next_key]
        run_id = hypothesis.get("id", next_key)
        log_path = run_log_path(default_run_dir(run_id, root=ROOT))

        cmd = build_trainer_cmd(
            hypothesis,
            defaults,
            run_id,
            wandb_group,
            opponent=args.opponent,
            self_play=args.self_play,
        )
        t0 = time.time()
        proc = run_trainer(cmd, log_path, log)
        status, collapse_verdict = monitor_run(
            proc, log_path, log, checkpoint_step=args.checkpoint_step
        )
        record = record_from_run(
            next_key,
            hypothesis,
            defaults,
            run_id,
            status,
            log_path,
            collapse_verdict,
            t0,
            orchestrator_opponent=args.opponent,
            orchestrator_self_play=args.self_play,
        )
        history.append(record)
        write_summary(history, OVERNIGHT_DIR / "summary.json", wandb_group)

        log.info(
            "Run %d/%d finished: %s status=%s duration=%.0fs",
            run_index + 1,
            args.max_runs,
            run_id,
            status,
            record.duration_s,
        )

        next_key = pick_next_hypothesis(history, hypotheses, tried, log)
        if next_key:
            log.info("Next hypothesis: %s", next_key)

    log.info(
        "Orchestrator done. completed=%d collapsed=%d crashed=%d",
        sum(1 for r in history if r.completed_500k),
        sum(1 for r in history if r.collapsed_at_100k),
        sum(1 for r in history if r.status == "crashed"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
