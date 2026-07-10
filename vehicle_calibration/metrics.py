"""Band-gated metric extraction from a profile table.

Each maneuver declares a ``metric`` type; the matching extractor below reduces its
``measure`` samples (gated to the maneuver's speed ``fit_band``) to a small set of
scalars. Fit-band scalars feed the objective in ``fit.py``; diagnostic scalars
(peak launch accel, coast tau, low-speed buckets) are reported but never fitted.
"""

from __future__ import annotations

import math

import numpy as np

from .maneuvers import Maneuver, Schedule
from .schema import ProfileTable


def _band_mask(speed: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (speed >= band[0]) & (speed <= band[1])


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else math.nan


def _linfit_slope(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2:
        return math.nan
    return float(np.polyfit(t, y, 1)[0])


def _metric_accel(cols: dict, band: tuple[float, float]) -> dict:
    t, speed = cols["t"], cols["speed"]
    out: dict[str, float] = {}

    # Mid-band acceleration: average dv/dt while crossing the fit band, found by
    # the time to go from band low to band high (robust to launch-spike noise).
    lo, hi = band[0], band[1] if math.isfinite(band[1]) else band[0] + 2.0
    t_lo = _first_crossing(t, speed, lo)
    t_hi = _first_crossing(t, speed, hi)
    if t_lo is not None and t_hi is not None and t_hi > t_lo:
        out["accel_band"] = (hi - lo) / (t_hi - t_lo)
    else:
        out["accel_band"] = math.nan

    # Drag-limited plateau: mean of the last ~1.5 s.
    tail = speed[t >= (t[-1] - 1.5)] if t.size else np.array([])
    out["steady_speed"] = _safe_mean(tail)

    # Diagnostic: peak launch accel from rest (finite difference).
    if t.size > 1:
        dv = np.diff(speed) / np.diff(t)
        out["peak_accel_diag"] = float(np.nanmax(dv)) if dv.size else math.nan
    else:
        out["peak_accel_diag"] = math.nan
    return out


def _metric_brake(cols: dict, band: tuple[float, float]) -> dict:
    t, speed = cols["t"], cols["speed"]
    mask = _band_mask(speed, band)
    out = {"brake_decel": math.nan}
    if mask.sum() >= 2:
        # Deceleration is the negative slope of speed over the band window.
        out["brake_decel"] = -_linfit_slope(t[mask], speed[mask])
    return out


def _metric_coast(cols: dict, band: tuple[float, float]) -> dict:
    # Diagnostic only: exponential decay time constant of the in-band coast.
    t, speed = cols["t"], cols["speed"]
    mask = _band_mask(speed, band) & (speed > 0.1)
    out = {"coast_tau_diag": math.nan, "coast_decel_diag": math.nan}
    if mask.sum() >= 3:
        tt = t[mask] - t[mask][0]
        out["coast_decel_diag"] = -_linfit_slope(tt, speed[mask])
        # ln(v) ~ ln(v0) - t/tau
        slope = _linfit_slope(tt, np.log(speed[mask]))
        out["coast_tau_diag"] = (-1.0 / slope) if slope < 0 else math.nan
    return out


def _metric_corner(cols: dict, band: tuple[float, float]) -> dict:
    speed, omega = cols["speed"], cols["omega_z"]
    mask = _band_mask(speed, band)
    out = {
        "mean_speed": math.nan,
        "yaw_rate": math.nan,
        "turn_radius": math.nan,
        "max_lat_acc": math.nan,
    }
    if mask.sum() >= 1:
        v = speed[mask]
        w = np.abs(omega[mask])
        out["mean_speed"] = _safe_mean(v)
        out["yaw_rate"] = _safe_mean(w)
        mw = _safe_mean(w)
        out["turn_radius"] = (_safe_mean(v) / mw) if mw > 1e-3 else math.nan
        out["max_lat_acc"] = float(np.max(v * w)) if v.size else math.nan
    return out


def _metric_steer_step(cols: dict, band: tuple[float, float]) -> dict:
    # First-order rise of yaw rate after the steer step, gated to in-band speed.
    t, speed, omega = cols["t"], cols["speed"], cols["omega_z"]
    mask = _band_mask(speed, band)
    out = {"tau_steer": math.nan, "yaw_ss": math.nan}
    if mask.sum() < 3:
        return out
    tt = t[mask] - t[mask][0]
    w = np.abs(omega[mask])
    yaw_ss = _safe_mean(w[tt >= (tt[-1] - 1.0)])
    out["yaw_ss"] = yaw_ss
    if not math.isfinite(yaw_ss) or yaw_ss <= 1e-3:
        return out
    # tau ~= time to reach 63.2% of steady-state yaw rate.
    target = 0.632 * yaw_ss
    idx = np.argmax(w >= target)
    if w[idx] >= target:
        out["tau_steer"] = float(tt[idx])
    return out


_EXTRACTORS = {
    "accel": _metric_accel,
    "brake": _metric_brake,
    "coast": _metric_coast,
    "corner": _metric_corner,
    "steer_step": _metric_steer_step,
}

# Which scalar keys per metric type are fit targets (the rest are diagnostic).
FIT_KEYS = {
    "accel": ["accel_band", "steady_speed"],
    "brake": ["brake_decel"],
    "coast": [],
    "corner": ["turn_radius", "max_lat_acc"],
    "steer_step": ["tau_steer"],
}


def _first_crossing(t: np.ndarray, y: np.ndarray, level: float) -> float | None:
    for i in range(1, len(y)):
        if y[i - 1] < level <= y[i]:
            # linear interpolation between samples
            dy = y[i] - y[i - 1]
            frac = (level - y[i - 1]) / dy if dy != 0 else 0.0
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return None


def summarize_maneuver(table: ProfileTable, man: Maneuver) -> dict:
    cols = table.select(man.id, role="measure")
    extractor = _EXTRACTORS[man.metric]
    values = extractor(cols, man.fit_band)
    return {
        "metric": man.metric,
        "fit_band": [man.fit_band[0], _band_hi(man.fit_band[1])],
        "diagnostic_only": man.diagnostic_only,
        "n_measure": int(len(cols["t"])),
        "n_in_band": int(_band_mask(cols["speed"], man.fit_band).sum()),
        "values": values,
    }


def summarize(table: ProfileTable, schedule: Schedule, v_fit_min: float) -> dict:
    maneuvers = {}
    for man in schedule.maneuvers:
        if man.id in table.maneuver_ids():
            maneuvers[man.id] = summarize_maneuver(table, man)
    return {
        "v_fit_min": v_fit_min,
        "source": str(table.data["source"][0]) if len(table) else "",
        "maneuvers": maneuvers,
    }


def flatten_fit_targets(summary: dict, v_fit_min: float) -> dict[str, float]:
    """Flat ``{maneuver.key: value}`` map of fit-band scalars only.

    Diagnostic-only maneuvers and out-of-band / NaN values are dropped so the
    objective never sees low-speed crunching.
    """
    flat: dict[str, float] = {}
    for mid, m in summary["maneuvers"].items():
        if m["diagnostic_only"]:
            continue
        for key in FIT_KEYS.get(m["metric"], []):
            val = m["values"].get(key, math.nan)
            if isinstance(val, float) and math.isfinite(val):
                flat[f"{mid}.{key}"] = val
    return flat


def _band_hi(hi: float):
    return None if math.isinf(hi) else hi
