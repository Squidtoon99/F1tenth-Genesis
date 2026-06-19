"""Pure episode-monitoring logic for the evaluation node (no ROS imports).

Tracks lap completion, out-of-bounds, and stuck detection from a stream of Frenet
progress / lateral-error / speed samples so it can be unit-tested deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeEvent:
    progress_ratio: float
    lap_completed: bool
    oob: bool
    stuck: bool
    lap_time: float | None
    lap_count: int
    max_progress: float


@dataclass
class EpisodeMonitor:
    oob_margin_m: float = 0.0
    stuck_speed_mps: float = 0.2
    stuck_timeout_s: float = 3.0
    lap_hi: float = 0.75
    lap_lo: float = 0.25

    lap_count: int = 0
    max_progress: float = 0.0
    last_lap_time: float | None = None
    _prev_progress: float | None = field(default=None, repr=False)
    _lap_start_t: float | None = field(default=None, repr=False)
    _last_move_t: float | None = field(default=None, repr=False)

    def reset(self, t: float):
        self.max_progress = 0.0
        self._prev_progress = None
        self._lap_start_t = t
        self._last_move_t = t

    def update(
        self,
        s: float,
        track_len: float,
        ey: float,
        w_left: float,
        w_right: float,
        speed: float,
        t: float,
    ) -> EpisodeEvent:
        if self._lap_start_t is None:
            self._lap_start_t = t
        if self._last_move_t is None:
            self._last_move_t = t

        progress = (s / track_len) % 1.0 if track_len > 1e-6 else 0.0
        self.max_progress = max(self.max_progress, progress)

        lap_completed = False
        lap_time = None
        if (
            self._prev_progress is not None
            and self._prev_progress > self.lap_hi
            and progress < self.lap_lo
        ):
            lap_completed = True
            lap_time = t - self._lap_start_t
            self.last_lap_time = lap_time
            self.lap_count += 1
            self._lap_start_t = t
            self.max_progress = progress
        self._prev_progress = progress

        oob = (ey > (w_left - self.oob_margin_m)) or (
            ey < -(w_right - self.oob_margin_m)
        )

        if speed >= self.stuck_speed_mps:
            self._last_move_t = t
        stuck = (t - self._last_move_t) > self.stuck_timeout_s

        return EpisodeEvent(
            progress_ratio=progress,
            lap_completed=lap_completed,
            oob=oob,
            stuck=stuck,
            lap_time=lap_time,
            lap_count=self.lap_count,
            max_progress=self.max_progress,
        )
