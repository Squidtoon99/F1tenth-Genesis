"""Canonical per-step profile schema shared by Genesis and IRL.

Both the in-sim profiler and the rosbag parser emit a CSV with these columns at a
fixed 10 Hz control rate so that ``metrics.py`` and ``compare.py`` can treat the
two sources identically. ``steer_state`` is the internal lagged steering angle and
is only available in Genesis (NaN for IRL, which only knows the commanded angle).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

COLUMNS = [
    "t",            # seconds elapsed since the start of this maneuver
    "maneuver",     # maneuver id (e.g. "L1_accel")
    "role",         # "warmup" | "measure"
    "throttle",     # commanded normalized throttle in [-1, 1]
    "steer",        # commanded normalized steer in [-1, 1]
    "x",            # world / map-frame x (m)
    "y",            # world / map-frame y (m)
    "yaw",          # heading (rad)
    "vx",           # body-frame longitudinal velocity (m/s)
    "vy",           # body-frame lateral velocity (m/s)
    "speed",        # planar speed sqrt(vx^2 + vy^2) (m/s)
    "ax",           # body-frame longitudinal accel (m/s^2)
    "ay",           # body-frame lateral accel (m/s^2)
    "omega_z",      # yaw rate (rad/s)
    "steer_state",  # internal lagged steer angle (rad); NaN for IRL
    "source",       # "genesis" | "irl"
]

_FLOAT_COLS = {
    "t", "throttle", "steer", "x", "y", "yaw", "vx", "vy", "speed",
    "ax", "ay", "omega_z", "steer_state",
}


@dataclass
class ProfileTable:
    """Column-oriented store of profile rows with simple CSV round-tripping."""

    data: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "ProfileTable":
        cols: dict[str, list] = {c: [] for c in COLUMNS}
        for row in rows:
            for c in COLUMNS:
                cols[c].append(row.get(c, math.nan if c in _FLOAT_COLS else ""))
        data = {}
        for c in COLUMNS:
            if c in _FLOAT_COLS:
                data[c] = np.asarray(cols[c], dtype=np.float64)
            else:
                data[c] = np.asarray(cols[c], dtype=object)
        return cls(data=data)

    def __len__(self) -> int:
        return len(self.data["t"]) if self.data else 0

    def to_csv(self, path: str) -> None:
        n = len(self)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for i in range(n):
                writer.writerow(
                    [
                        _fmt(self.data[c][i]) if c in _FLOAT_COLS else self.data[c][i]
                        for c in COLUMNS
                    ]
                )

    @classmethod
    def read_csv(cls, path: str) -> "ProfileTable":
        cols: dict[str, list] = {c: [] for c in COLUMNS}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for c in COLUMNS:
                    raw = row.get(c, "")
                    if c in _FLOAT_COLS:
                        cols[c].append(float(raw) if raw not in ("", None) else math.nan)
                    else:
                        cols[c].append(raw)
        data = {
            c: (
                np.asarray(cols[c], dtype=np.float64)
                if c in _FLOAT_COLS
                else np.asarray(cols[c], dtype=object)
            )
            for c in COLUMNS
        }
        return cls(data=data)

    def maneuver_ids(self) -> list[str]:
        seen: list[str] = []
        for m in self.data["maneuver"]:
            if m not in seen:
                seen.append(m)
        return seen

    def select(self, maneuver: str, role: str | None = None) -> dict[str, np.ndarray]:
        """Return a column dict for one maneuver, optionally filtered by role."""
        mask = self.data["maneuver"] == maneuver
        if role is not None:
            mask = mask & (self.data["role"] == role)
        return {c: self.data[c][mask] for c in COLUMNS}


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return repr(float(value))
