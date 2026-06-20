"""Health check for standalone trainer runs at the 100k checkpoint."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STEP_LINE = re.compile(
    r"step=(\d+) .* mean_ep_reward=([-\d.]+|nan|inf)"
)
REWARD_LINE = re.compile(
    r"progress=([-\d.]+|nan|inf) speed="
)
ENV_LINE = re.compile(
    r"env: speed=[-\d.]+ lat_err=[-\d.]+ oob_frac=([-\d.]+|nan|inf)"
)
WARNING_PATTERNS = (
    "Non-finite",
    "Genesis raised",
    "Resetting all envs",
    "CUDA out of memory",
    "OutOfMemoryError",
)


@dataclass
class StepMetrics:
    step: int
    mean_ep_reward: float
    progress: float | None = None
    oob_frac: float | None = None


@dataclass
class CollapseVerdict:
    collapsed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warning_count: int = 0


def _to_float(value: str) -> float:
    lowered = value.lower()
    if lowered in ("nan", "inf", "-inf"):
        return float("nan")
    return float(value)


def parse_log_metrics(
    log_path: Path | str,
    target_step: int = 100_000,
    tolerance: int = 100,
) -> dict[str, Any]:
    """Parse trainer log and return metrics at the checkpoint step."""
    path = Path(log_path)
    if not path.exists():
        return {"found": False, "error": f"log not found: {path}"}

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    best: StepMetrics | None = None
    best_delta = tolerance + 1
    all_steps: list[StepMetrics] = []

    i = 0
    while i < len(lines):
        m = STEP_LINE.search(lines[i])
        if m:
            step = int(m.group(1))
            reward = _to_float(m.group(2))
            progress = None
            oob_frac = None
            if i + 1 < len(lines):
                rm = REWARD_LINE.search(lines[i + 1])
                if rm:
                    progress = _to_float(rm.group(1))
            if i + 2 < len(lines):
                em = ENV_LINE.search(lines[i + 2])
                if em:
                    oob_frac = _to_float(em.group(1))

            entry = StepMetrics(
                step=step,
                mean_ep_reward=reward,
                progress=progress,
                oob_frac=oob_frac,
            )
            all_steps.append(entry)
            delta = abs(step - target_step)
            if delta <= tolerance and delta <= best_delta:
                best = entry
                best_delta = delta
        i += 1

    if best is None:
        return {
            "found": False,
            "error": f"no metrics within {tolerance} of step {target_step}",
            "latest_step": all_steps[-1].step if all_steps else None,
        }

    return {
        "found": True,
        "step": best.step,
        "mean_ep_reward": best.mean_ep_reward,
        "progress": best.progress,
        "oob_frac": best.oob_frac,
        "all_steps": [
            {
                "step": s.step,
                "mean_ep_reward": s.mean_ep_reward,
                "progress": s.progress,
                "oob_frac": s.oob_frac,
            }
            for s in all_steps
        ],
    }


def count_warnings(log_tail: str) -> int:
    return sum(log_tail.count(p) for p in WARNING_PATTERNS)


def count_warnings_in_step_range(
    log_path: Path | str,
    step_min: int = 50_000,
    step_max: int = 100_000,
) -> int:
    """Count warning lines emitted between two logged steps."""
    path = Path(log_path)
    if not path.exists():
        return 0

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_range = False
    count = 0
    for line in lines:
        sm = STEP_LINE.search(line)
        if sm:
            step = int(sm.group(1))
            in_range = step_min <= step <= step_max
        elif in_range and any(p in line for p in WARNING_PATTERNS):
            count += 1
    return count


def reward_plateau(
    all_steps: list[dict[str, Any]],
    checkpoints: tuple[int, ...] = (80_000, 90_000, 100_000),
    tolerance: int = 5_000,
    plateau_threshold: float = 0.10,
) -> bool:
    """True if mean_ep_reward growth between checkpoints is < threshold."""
    values: list[float] = []
    for target in checkpoints:
        best: float | None = None
        best_delta = tolerance + 1
        for entry in all_steps:
            step = entry["step"]
            delta = abs(step - target)
            if delta <= tolerance and delta <= best_delta:
                best = entry["mean_ep_reward"]
                best_delta = delta
        if best is not None and best == best:
            values.append(best)

    if len(values) < 2:
        return False

    base = max(abs(values[0]), 1e-6)
    growth = (values[-1] - values[0]) / base
    return growth < plateau_threshold


def is_collapsed(
    metrics: dict[str, Any],
    log_path: Path | str | None = None,
    log_tail_lines: int = 500,
) -> CollapseVerdict:
    """Return collapse verdict from parsed metrics and recent log tail."""
    reasons: list[str] = []
    warning_count = 0

    if not metrics.get("found"):
        return CollapseVerdict(
            collapsed=True,
            reasons=[metrics.get("error", "metrics not found")],
            metrics=metrics,
        )

    mean_ep_reward = metrics.get("mean_ep_reward")
    oob_frac = metrics.get("oob_frac")
    progress = metrics.get("progress")

    signal_failures = 0

    if mean_ep_reward is None or mean_ep_reward != mean_ep_reward or mean_ep_reward < 30:
        signal_failures += 1
        reasons.append(f"mean_ep_reward={mean_ep_reward} < 30")

    if oob_frac is not None and oob_frac == oob_frac and oob_frac > 0.20:
        signal_failures += 1
        reasons.append(f"oob_frac={oob_frac:.3f} > 0.20")

    if progress is not None and progress == progress and progress < 0.3:
        signal_failures += 1
        reasons.append(f"progress={progress:.3f} < 0.3")

    if log_path is not None:
        path = Path(log_path)
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-log_tail_lines:])
            warning_count = count_warnings(tail)
            range_warnings = count_warnings_in_step_range(path, 50_000, 100_000)

            if warning_count >= 3:
                signal_failures += 1
                reasons.append(
                    f"{warning_count} warning(s) in last {log_tail_lines} log lines"
                )
            if range_warnings > 5:
                signal_failures += 1
                reasons.append(
                    f"{range_warnings} warnings between steps 50k-100k"
                )

    collapsed = signal_failures >= 2 or (
        warning_count >= 3 and any("Non-finite" in r or "Genesis raised" in r for r in reasons)
    )

    return CollapseVerdict(
        collapsed=collapsed,
        reasons=reasons,
        metrics=metrics,
        warning_count=warning_count,
    )


def evaluate_at_checkpoint(
    log_path: Path | str,
    target_step: int = 100_000,
) -> CollapseVerdict:
    metrics = parse_log_metrics(log_path, target_step=target_step)
    return is_collapsed(metrics, log_path=log_path)
