from typing import Any

import torch
import genesis as gs
import genesis.utils.geom as gu
from .utils import compute_oob_from_boundary_state


def init_reward_state(
    reward_scales: dict[str, float],
    num_envs: int,
    device: torch.device,
) -> dict[str, Any]:
    episode_sums = {
        name: torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
        for name in reward_scales.keys()
    }
    return {
        "reward_scales": reward_scales,
        "episode_sums": episode_sums,
        "last_reward_terms": {},
        "prev_s": None,
        "prev_step_counter": None,
        "last_progress_ds": torch.zeros((num_envs,), dtype=gs.tc_float, device=device),
    }


def ensure_progress_delta(
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    lap_count_buf: torch.Tensor,
) -> dict[str, Any]:
    if "progress_ds" in step_state:
        return step_state

    frenet_state = step_state["frenet"]
    s = frenet_state["s"].reshape(-1)
    length = frenet_state["L"]
    step_now = episode_steps_buf.to(dtype=gs.tc_float)
    batch = s.shape[0]

    prev_step_counter = reward_state["prev_step_counter"]
    prev_s = reward_state["prev_s"]
    if prev_step_counter is None or prev_step_counter.numel() != batch:
        prev_step_counter = step_now.detach().clone()
    if prev_s is None or prev_s.numel() != batch:
        prev_s = s.detach().clone()

    prev_step = prev_step_counter.reshape(-1)
    prev_s_flat = prev_s.reshape(-1)
    reset_mask = step_now < prev_step

    ds = s - prev_s_flat
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)

    max_step_frac = float(reward_cfg.get("progress_max_step_frac", 0.05))
    max_ds = max_step_frac * length
    ds = ds.clamp(min=-max_ds, max=max_ds)
    ds = torch.where(reset_mask, torch.zeros_like(ds), ds)

    lap_cross = (
        (~reset_mask) & (prev_s_flat > 0.9 * length) & (s < 0.1 * length) & (ds > 0.0)
    )
    lap_count_buf += lap_cross.to(dtype=lap_count_buf.dtype)

    reward_state["prev_s"] = s.detach().clone()
    reward_state["prev_step_counter"] = step_now.detach().clone()
    reward_state["last_progress_ds"] = ds.detach().clone()

    step_state["progress_ds"] = ds
    step_state["track_length"] = length
    return step_state


def sync_progress_state_for_resets(
    reward_state: dict[str, Any],
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reset_mask: torch.Tensor,
) -> None:
    s = step_state["frenet"]["s"].reshape(-1)

    prev_s = reward_state["prev_s"]
    if prev_s is None or prev_s.numel() != s.numel():
        reward_state["prev_s"] = s.detach().clone()
    else:
        prev_s[reset_mask] = s[reset_mask].detach()

    step_now = episode_steps_buf.to(dtype=gs.tc_float)
    prev_step_counter = reward_state["prev_step_counter"]
    if prev_step_counter is None or prev_step_counter.numel() != s.numel():
        reward_state["prev_step_counter"] = step_now.detach().clone()
    else:
        prev_step_counter[reset_mask] = step_now[reset_mask].detach()

    reward_state["last_progress_ds"][reset_mask] = 0.0


def reward_progress(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    frenet_state = step_state["frenet"]
    ds = step_state["progress_ds"]
    pos = frenet_state["pos"].reshape(-1, 2)
    proj = frenet_state["proj"].reshape(-1, 2)

    k_fwd = float(reward_cfg.get("progress_k_fwd", 5.0))
    k_back = float(reward_cfg.get("progress_k_back", 5.0))
    fwd = torch.clamp(ds, min=0.0)
    back = torch.clamp(ds, max=0.0)
    reward = k_fwd * fwd + k_back * back

    max_lateral_m = reward_cfg.get("progress_max_lateral_m", 1.0)
    if max_lateral_m is not None:
        e_lat = torch.linalg.norm(pos - proj, dim=-1).reshape(-1)
        reward = torch.where(
            e_lat <= float(max_lateral_m), reward, torch.zeros_like(reward)
        )

    return reward


def reward_oob_penalty(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    margin_m = float(reward_cfg.get("oob_margin_m", 0.5))
    k_oob = float(reward_cfg.get("oob_k", 10.0))
    _, oob_dist = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=margin_m
    )
    return -k_oob * oob_dist


def reward_tyre_slip_penalty(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
) -> torch.Tensor:
    """
    Minimal MVP tyre-slip penalty (no step_state, no extra structure).

    Conventions:
    - wheel order is [left_rear, right_rear, left_front, right_front]
    - motion velocity is sampled at wheel links
    - local frame is non-spinning: base for rear, steering hinges for front
    """

    wheel_state = step_state["wheel_state"]
    # --- pull state ---
    wheel_lin_vel_world = wheel_state["motion_link_vel"]  # (N, 4, 3)
    wheel_frame_quat_world = wheel_state["frame_quat"]  # (N, 4, 4)
    spin_rate = wheel_state["dof_vel"]  # (N, n_dofs)

    # --- world -> non-spinning local slip frame ---
    lin_vel_local = gu.inv_transform_by_quat(
        wheel_lin_vel_world, wheel_frame_quat_world
    )

    v_fwd = lin_vel_local[:, :, 0]
    v_lat = lin_vel_local[:, :, 1]

    # Optional sign overrides in case URDF axis conventions are inverted.
    front_sign = float(reward_cfg.get("slip_forward_sign_front", 1.0))
    rear_sign = float(reward_cfg.get("slip_forward_sign_rear", 1.0))
    v_fwd = v_fwd.clone()
    v_fwd[:, :2] = rear_sign * v_fwd[:, :2]
    v_fwd[:, 2:] = front_sign * v_fwd[:, 2:]

    # --- slip calculations ---
    eps = float(reward_cfg.get("slip_eps", 0.1))
    wheel_radius = float(reward_cfg.get("wheel_radius_m", 0.05))

    slip_angle = torch.atan2(v_lat, torch.abs(v_fwd).clamp_min(eps))

    wheel_speed = wheel_radius * spin_rate
    denom = torch.maximum(torch.abs(wheel_speed), torch.abs(v_fwd)).clamp_min(eps)
    slip_ratio = (wheel_speed - v_fwd) / denom

    # --- Sophy-style penalty ---
    slip_angle_mag = torch.abs(slip_angle)
    slip_ratio_mag = torch.clamp(torch.abs(slip_ratio), max=1.0)

    per_wheel = slip_ratio_mag * slip_angle_mag
    penalty = -torch.sum(per_wheel, dim=1)

    penalty = torch.where(penalty > -1.0, torch.zeros_like(penalty), penalty)

    return penalty


def compute_rewards(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    lap_count_buf: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    step_state = ensure_progress_delta(
        step_state=step_state,
        episode_steps_buf=episode_steps_buf,
        reward_cfg=reward_cfg,
        reward_state=reward_state,
        lap_count_buf=lap_count_buf,
    )

    num_envs = episode_steps_buf.shape[0]
    device = episode_steps_buf.device
    reward_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=device)

    progress = reward_progress(step_state, reward_cfg)
    oob_penalty = reward_oob_penalty(step_state, reward_cfg)
    tyre_slip_penalty = reward_tyre_slip_penalty(step_state, reward_cfg)
    progress = torch.where(oob_penalty < 0.0, torch.zeros_like(progress), progress)

    progress *= reward_cfg["reward_scales"]["progress"]
    oob_penalty *= reward_cfg["reward_scales"]["oob_penalty"]
    tyre_slip_penalty *= reward_cfg["reward_scales"]["tyre_slip_penalty"]
    last_terms: dict[str, torch.Tensor] = {
        "progress": progress.clone(),
        "oob_penalty": oob_penalty.clone(),
        "tyre_slip_penalty": tyre_slip_penalty.clone(),
    }

    reward_buf += progress + oob_penalty + tyre_slip_penalty

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state
