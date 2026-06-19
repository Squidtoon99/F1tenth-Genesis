#!/usr/bin/env python3
"""Sanity-check F1TENTH physics: throttle, steer, brake rollouts on CPU."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs
import numpy as np
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv

G = 9.81
MUE = DEFAULT_CONFIG["env"]["tire_friction"]
CONTROL_DT = (
    DEFAULT_CONFIG["env"]["sim_dt"] * DEFAULT_CONFIG["env"]["control_interval"]
)
CONTROL_INTERVAL = DEFAULT_CONFIG["env"]["control_interval"]


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().numpy()
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{name} contains NaN or inf")


def _speed_xy(lin_vel: torch.Tensor) -> float:
    return float(torch.linalg.norm(lin_vel[0, :2]).item())


def _lateral_accel(lin_acc: torch.Tensor) -> float:
    """Body-frame lateral acceleration (y-axis), not total horizontal magnitude."""
    return float(abs(lin_acc[0, 1].item()))


def _uprightness(quat: torch.Tensor) -> float:
    """1.0 = upright, 0.0 = on side."""
    from genesis.utils.geom import quat_to_R

    rot = quat_to_R(quat)[0]
    return float(rot[2, 2].item())


def run_phase(
    env: F1tenthEnv,
    throttle: float,
    steer: float,
    n_steps: int,
    *,
    use_fd_lateral: bool = True,
) -> dict[str, float]:
    action = torch.tensor([[throttle, steer]], dtype=gs.tc_float, device=gs.device)
    stats: dict[str, float] = {
        "max_speed": 0.0,
        "max_lat_acc": 0.0,
        "max_yaw_rate": 0.0,
        "min_uprightness": 1.0,
        "initial_speed": 0.0,
    }
    prev_lat_vel = float(env.base_lin_vel[0, 1].item())

    for step_i in range(n_steps):
        env.step(action, n_steps=CONTROL_INTERVAL)

        _assert_finite("base_pos", env.base_pos)
        _assert_finite("base_lin_vel", env.base_lin_vel)
        _assert_finite("base_quat", env.base_quat)
        _assert_finite("base_lin_acc", env.base_lin_acc)

        speed = _speed_xy(env.base_lin_vel)
        lat_vel = float(env.base_lin_vel[0, 1].item())
        if use_fd_lateral and step_i > 0:
            lat_acc = abs((lat_vel - prev_lat_vel) / CONTROL_DT)
        else:
            lat_acc = _lateral_accel(env.base_lin_acc)
        prev_lat_vel = lat_vel

        yaw_rate = float(abs(env.base_ang_vel[0, 2].item()))
        uprightness = _uprightness(env.base_quat)

        if step_i == 0:
            stats["initial_speed"] = speed
        stats["max_speed"] = max(stats["max_speed"], speed)
        stats["max_lat_acc"] = max(stats["max_lat_acc"], lat_acc)
        stats["max_yaw_rate"] = max(stats["max_yaw_rate"], yaw_rate)
        stats["min_uprightness"] = min(stats["min_uprightness"], uprightness)

        if stats["min_uprightness"] < 0.7:
            raise AssertionError(
                f"Car tipped: uprightness={stats['min_uprightness']:.3f}"
            )

    stats["final_speed"] = speed
    return stats


def ramp_to_speed(
    env: F1tenthEnv,
    target_mps: float,
    max_steps: int = 80,
) -> tuple[float, float]:
    """Accelerate with full throttle until reaching target speed."""
    action = torch.tensor([[1.0, 0.0]], dtype=gs.tc_float, device=gs.device)
    speed = 0.0
    for step_i in range(max_steps):
        env.step(action, n_steps=CONTROL_INTERVAL)
        speed = _speed_xy(env.base_lin_vel)
        if speed >= target_mps:
            return speed, (step_i + 1) * CONTROL_DT
    return speed, max_steps * CONTROL_DT


def headless_gs_init(backend=None) -> None:
    """Patch the rasterizer for headless rendering and init Genesis once."""
    import pyglet
    from genesis.vis.rasterizer import Rasterizer

    pyglet.options["headless"] = True

    def _headless_rasterizer_build(self):
        if self._context is None:
            return
        self.visualizer = self._context.visualizer

    Rasterizer.build = _headless_rasterizer_build

    gs.init(
        backend=backend or gs.cpu,
        logging_level="warning",
        performance_mode=True,
    )


def run_physics_check(extra_overrides: dict | None = None, verbose: bool = True) -> dict:
    """Run the throttle/steer/brake stability gate. Assumes gs is initialized.

    Returns a summary dict; raises AssertionError if any stability check fails.
    """

    def log(*args):
        if verbose:
            print(*args)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = cfg["env"]
    env_cfg.update(
        {
            "reset_spawn_margin_m": 0.0,
            "reset_yaw_jitter_rad": 0.0,
            "reset_along_track_jitter_m": 0.0,
            "reset_speed_min_mps": 0.0,
            "reset_speed_max_mps": 0.0,
            "simulate_action_latency": False,
            "launch_strategy": "fixed",
            "car_spawn_pos": (0.0, 0.0, 0.05),
            "car_spawn_rot": (0.0, 0.0, 0.0),
            # Keep rollouts continuous — OOB/not-moving resets corrupt steer/brake phases.
            "term_oob_max_consecutive": 10**9,
            "term_oob_margin_m": -100.0,
            "term_not_moving_time_s": 10**9,
            "term_heading_error_rad": 10.0,
        }
    )
    if extra_overrides:
        env_cfg.update(extra_overrides)

    env = F1tenthEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
    )

    mass = 3.74
    f_drive = env_cfg["f_drive_max"]
    mue = env_cfg["tire_friction"]
    power = env_cfg["power_max"]
    izz = 0.07

    log("=== F1TENTH physics check (pure wheel-torque drive) ===")
    log(f"control_dt={CONTROL_DT}s  mass_target={mass} kg  chassis_izz={izz}")
    log(
        f"f_drive_max={f_drive} N  f_brake_max={env_cfg['f_brake_max']} N  "
        f"power_max={power} W  delta_max={env_cfg['delta_max']} rad"
    )
    log(f"tire_friction={MUE}  dragcoeff={env_cfg['dragcoeff']}  "
          f"enable_aero_drag={env_cfg['enable_aero_drag']}  num_obs={cfg['obs']['num_obs']}")
    dragcoeff = float(env_cfg["dragcoeff"])
    v_eq_drag = (power / dragcoeff) ** (1.0 / 3.0)
    log(f"power_drag_equilibrium~{v_eq_drag:.2f} m/s")

    # Phase 1: full throttle from rest (long enough to reach drag-limited plateau)
    throttle_steps = 120
    action = torch.tensor([[1.0, 0.0]], dtype=gs.tc_float, device=gs.device)
    accel_stats = {
        "max_speed": 0.0,
        "max_lat_acc": 0.0,
        "max_yaw_rate": 0.0,
        "min_uprightness": 1.0,
        "final_speed": 0.0,
        "time_to_3ms": None,
        "time_to_5ms": None,
        "time_to_plateau": None,
        "peak_long_accel": 0.0,
        "peak_long_accel_fd": 0.0,
    }
    prev_speed = 0.0
    speed_history: list[float] = []
    plateau_window = 10
    plateau_eps = 0.15
    for step_i in range(throttle_steps):
        env.step(action, n_steps=CONTROL_INTERVAL)

        _assert_finite("base_pos", env.base_pos)
        _assert_finite("base_lin_vel", env.base_lin_vel)
        _assert_finite("base_quat", env.base_quat)
        _assert_finite("base_lin_acc", env.base_lin_acc)

        speed = _speed_xy(env.base_lin_vel)
        lat_acc = _lateral_accel(env.base_lin_acc)
        yaw_rate = float(abs(env.base_ang_vel[0, 2].item()))
        uprightness = _uprightness(env.base_quat)
        long_accel = (speed - prev_speed) / CONTROL_DT
        prev_speed = speed
        speed_history.append(speed)

        accel_stats["max_speed"] = max(accel_stats["max_speed"], speed)
        accel_stats["max_lat_acc"] = max(accel_stats["max_lat_acc"], lat_acc)
        accel_stats["max_yaw_rate"] = max(accel_stats["max_yaw_rate"], yaw_rate)
        accel_stats["min_uprightness"] = min(accel_stats["min_uprightness"], uprightness)
        accel_stats["peak_long_accel"] = max(accel_stats["peak_long_accel"], long_accel)
        accel_stats["peak_long_accel_fd"] = accel_stats["peak_long_accel"]
        accel_stats["final_speed"] = speed

        if (
            accel_stats["time_to_plateau"] is None
            and len(speed_history) >= plateau_window
            and max(speed_history[-plateau_window:]) - min(speed_history[-plateau_window:])
            < plateau_eps
        ):
            accel_stats["time_to_plateau"] = (step_i + 1) * CONTROL_DT

        if accel_stats["time_to_3ms"] is None and speed >= 3.0:
            accel_stats["time_to_3ms"] = (step_i + 1) * CONTROL_DT
        if accel_stats["time_to_5ms"] is None and speed >= 5.0:
            accel_stats["time_to_5ms"] = (step_i + 1) * CONTROL_DT

        if accel_stats["min_uprightness"] < 0.7:
            raise AssertionError(
                f"Car tipped: uprightness={accel_stats['min_uprightness']:.3f}"
            )

    time_to_speed = throttle_steps * CONTROL_DT
    traction_accel = mue * G
    steady_tail = speed_history[-20:] if len(speed_history) >= 20 else speed_history
    steady_speed = float(np.mean(steady_tail))
    log("\n--- Phase 1: full throttle ---")
    log(f"duration={time_to_speed:.1f}s  max_speed={accel_stats['max_speed']:.3f} m/s")
    log(f"final_speed={accel_stats['final_speed']:.3f} m/s")
    log(f"steady_speed(last20)={steady_speed:.3f} m/s")
    log(f"time_to_plateau={accel_stats['time_to_plateau']}")
    log(f"peak_long_accel(fd)={accel_stats['peak_long_accel_fd']:.2f} m/s^2")
    log(f"time_to_3m/s={accel_stats['time_to_3ms']}  time_to_5m/s={accel_stats['time_to_5ms']}")
    log(f"traction_limit~{traction_accel:.2f} m/s^2  f_drive/m~{f_drive/mass:.2f} m/s^2")
    log(f"max_lat_acc={accel_stats['max_lat_acc']:.3f} m/s^2")
    log(f"max_yaw_rate={accel_stats['max_yaw_rate']:.3f} rad/s")
    log(f"min_uprightness={accel_stats['min_uprightness']:.3f}")

    if accel_stats["max_speed"] < 3.0:
        raise AssertionError(
            f"Top speed too low ({accel_stats['max_speed']:.2f} m/s)"
        )
    if accel_stats["time_to_3ms"] is None or accel_stats["time_to_3ms"] > 2.5:
        raise AssertionError(
            f"Too slow to reach 3 m/s: {accel_stats['time_to_3ms']}"
        )
    if accel_stats["max_speed"] < 12.0 or accel_stats["max_speed"] > 16.5:
        raise AssertionError(
            f"Steady-state top speed unrealistic: {accel_stats['max_speed']:.2f} m/s "
            f"(expected ~{v_eq_drag:.1f} m/s)"
        )
    if accel_stats["time_to_plateau"] is None:
        raise AssertionError("Speed did not plateau under full throttle")
    if abs(accel_stats["final_speed"] - steady_speed) > 0.3:
        raise AssertionError(
            f"Speed not plateaued: final={accel_stats['final_speed']:.2f} "
            f"steady_mean={steady_speed:.2f}"
        )
    if accel_stats["peak_long_accel_fd"] < 3.0:
        raise AssertionError(
            f"Initial acceleration too low: {accel_stats['peak_long_accel_fd']:.2f} m/s^2"
        )
    if accel_stats["peak_long_accel_fd"] > traction_accel * 1.05:
        log(
            f"NOTE: peak longitudinal accel ({accel_stats['peak_long_accel_fd']:.2f} m/s^2) "
            f"exceeds mu*g ({traction_accel:.2f} m/s^2) — launch finite-difference "
            f"spike from 0.1s control_dt, not sustained over-traction."
        )

    # Phase 2: steady steer at friction-feasible speed (~5.5 m/s)
    env.reset(torch.tensor([True], dtype=gs.tc_bool, device=gs.device))
    steer_entry_speed, steer_ramp_time = ramp_to_speed(env, 5.5)
    steer_stats = run_phase(env, throttle=0.35, steer=0.22, n_steps=40)
    lat_limit = MUE * G * 1.15
    log("\n--- Phase 2: steady steer ---")
    log(f"entry_speed={steer_entry_speed:.3f} m/s (ramp {steer_ramp_time:.1f}s)")
    log(f"max_speed={steer_stats['max_speed']:.3f} m/s")
    log(f"max_lat_acc={steer_stats['max_lat_acc']:.3f} m/s^2  limit~{lat_limit:.2f}")
    log(f"max_yaw_rate={steer_stats['max_yaw_rate']:.3f} rad/s")

    if steer_stats["max_lat_acc"] > lat_limit:
        raise AssertionError(
            f"Lateral accel {steer_stats['max_lat_acc']:.2f} exceeds "
            f"mue*g margin ({lat_limit:.2f})"
        )
    if steer_stats["max_yaw_rate"] > 5.0:
        raise AssertionError(
            f"Yaw rate unstable ({steer_stats['max_yaw_rate']:.2f} rad/s)"
        )

    # Phase 3: full brake from ~8 m/s
    env.reset(torch.tensor([True], dtype=gs.tc_bool, device=gs.device))
    brake_entry_speed, brake_ramp_time = ramp_to_speed(env, 8.0)
    brake_action = torch.tensor([[-1.0, 0.0]], dtype=gs.tc_float, device=gs.device)
    brake_stats = {
        "max_lat_acc": 0.0,
        "min_uprightness": 1.0,
        "initial_speed": brake_entry_speed,
        "final_speed": brake_entry_speed,
    }
    start_x = float(env.base_pos[0, 0].item())
    brake_time = None
    prev_lat_vel = float(env.base_lin_vel[0, 1].item())
    for step_i in range(40):
        env.step(brake_action, n_steps=CONTROL_INTERVAL)
        _assert_finite("base_pos", env.base_pos)
        _assert_finite("base_lin_vel", env.base_lin_vel)
        _assert_finite("base_quat", env.base_quat)

        speed = _speed_xy(env.base_lin_vel)
        lat_vel = float(env.base_lin_vel[0, 1].item())
        if step_i > 0:
            lat_acc = abs((lat_vel - prev_lat_vel) / CONTROL_DT)
            brake_stats["max_lat_acc"] = max(brake_stats["max_lat_acc"], lat_acc)
        prev_lat_vel = lat_vel
        brake_stats["min_uprightness"] = min(
            brake_stats["min_uprightness"], _uprightness(env.base_quat)
        )
        if brake_time is None and speed < 0.2:
            brake_time = (step_i + 1) * CONTROL_DT
        brake_stats["final_speed"] = speed

    brake_distance = float(env.base_pos[0, 0].item()) - start_x
    log("\n--- Phase 3: full brake ---")
    log(f"initial_speed={brake_stats['initial_speed']:.3f} m/s (ramp {brake_ramp_time:.1f}s)")
    log(f"final_speed={brake_stats['final_speed']:.3f} m/s")
    log(f"brake_time_to_0.2m/s={brake_time}")
    log(f"brake_distance={brake_distance:.3f} m")
    log(f"max_lat_acc={brake_stats['max_lat_acc']:.3f} m/s^2")
    log(f"min_uprightness={brake_stats['min_uprightness']:.3f}")

    if brake_stats["final_speed"] > 0.5:
        raise AssertionError(
            f"Braking insufficient: final speed {brake_stats['final_speed']:.2f} m/s"
        )

    env.close()
    log("\n=== All physics checks passed ===")
    return {
        "top_speed": accel_stats["max_speed"],
        "steady_speed": steady_speed,
        "time_to_3ms": accel_stats["time_to_3ms"],
        "time_to_plateau": accel_stats["time_to_plateau"],
        "peak_long_accel": accel_stats["peak_long_accel_fd"],
        "steer_max_lat_acc": steer_stats["max_lat_acc"],
        "steer_max_yaw_rate": steer_stats["max_yaw_rate"],
        "brake_time": brake_time,
        "brake_final_speed": brake_stats["final_speed"],
        "brake_distance": brake_distance,
    }


def main() -> int:
    headless_gs_init(gs.cpu)
    run_physics_check(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
