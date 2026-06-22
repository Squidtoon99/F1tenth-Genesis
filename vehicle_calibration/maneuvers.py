"""Load and validate the shared open-loop maneuver schedule.

The same YAML drives the Genesis profiler and the on-car ROS profiler node, so
there is exactly one source of truth for the action sequence. Each maneuver is a
list of constant-action segments held for a fixed duration at the control rate.
Only ``role: measure`` samples whose speed falls inside ``fit_band`` are used by
the fit objective; everything else is diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from . import DEFAULT_PROFILE_YAML, DEFAULT_SETTINGS_YAML

VALID_METRICS = {"accel", "brake", "coast", "corner", "steer_step"}
VALID_ROLES = {"warmup", "measure"}


@dataclass
class Segment:
    throttle: float
    steer: float
    duration_s: float
    role: str = "measure"


@dataclass
class Maneuver:
    id: str
    metric: str
    segments: list[Segment]
    fit_band: tuple[float, float]  # (v_min, v_max); v_max == inf means open-ended
    reset_before: bool = True
    diagnostic_only: bool = False
    description: str = ""

    def num_steps(self, control_hz: float) -> int:
        return sum(round(s.duration_s * control_hz) for s in self.segments)


@dataclass
class Schedule:
    control_hz: float
    maneuvers: list[Maneuver]

    def get(self, maneuver_id: str) -> Maneuver:
        for m in self.maneuvers:
            if m.id == maneuver_id:
                return m
        raise KeyError(f"maneuver {maneuver_id!r} not in schedule")


def _parse_band(raw) -> tuple[float, float]:
    if raw is None:
        return (0.0, float("inf"))
    lo, hi = raw
    lo = 0.0 if lo is None else float(lo)
    hi = float("inf") if hi is None else float(hi)
    return (lo, hi)


def load_schedule(path: str = DEFAULT_PROFILE_YAML) -> Schedule:
    with open(path) as fh:
        doc = yaml.safe_load(fh)

    control_hz = float(doc.get("control_hz", 10.0))
    maneuvers: list[Maneuver] = []
    ids: set[str] = set()
    for entry in doc["maneuvers"]:
        mid = str(entry["id"])
        if mid in ids:
            raise ValueError(f"duplicate maneuver id {mid!r}")
        ids.add(mid)

        metric = str(entry["metric"])
        if metric not in VALID_METRICS:
            raise ValueError(f"{mid}: unknown metric {metric!r}")

        segments = []
        for seg in entry["segments"]:
            role = str(seg.get("role", "measure"))
            if role not in VALID_ROLES:
                raise ValueError(f"{mid}: unknown role {role!r}")
            segments.append(
                Segment(
                    throttle=float(seg["throttle"]),
                    steer=float(seg["steer"]),
                    duration_s=float(seg["duration_s"]),
                    role=role,
                )
            )
        if not segments:
            raise ValueError(f"{mid}: no segments")

        maneuvers.append(
            Maneuver(
                id=mid,
                metric=metric,
                segments=segments,
                fit_band=_parse_band(entry.get("fit_band")),
                reset_before=bool(entry.get("reset_before", True)),
                diagnostic_only=bool(entry.get("diagnostic_only", False)),
                description=str(entry.get("description", "")),
            )
        )

    if not maneuvers:
        raise ValueError("schedule has no maneuvers")
    return Schedule(control_hz=control_hz, maneuvers=maneuvers)


def load_settings(path: str = DEFAULT_SETTINGS_YAML) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)
