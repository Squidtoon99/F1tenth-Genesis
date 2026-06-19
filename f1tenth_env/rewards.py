from typing import Any

import torch
import genesis as gs
from .car import compute_tyre_slip
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
    oob_dist_cap = float(reward_cfg.get("oob_dist_cap_m", 1.0))
    v_ref = float(reward_cfg.get("oob_speed_ref_mps", 3.0))
    _, oob_dist = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=margin_m
    )
    oob_dist = torch.clamp(oob_dist, max=oob_dist_cap)

    # GT Sophy-style: scale the off-course penalty by (squared) speed so high-speed
    # excursions are punished far harder than low-speed ones. This makes the speed
    # limit bind through the penalty and suppresses the fast off-track excursions
    # that drive the simulator into NaN spin-outs.
    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    speed_factor = 1.0 + (v / v_ref) ** 2
    return -k_oob * oob_dist * speed_factor


def reward_speed(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """
    Forward (track-aligned) speed reward, capped at a target speed.

    Rewards velocity projected onto the track tangent (so spinning/sliding does
    not pay), normalized to [0, 1] and saturated at speed_target_mps. The cap
    keeps the agent below the unstable high-speed spin-out regime instead of
    letting it farm raw speed.
    """
    v_xy = step_state["base_lin_vel"][:, :2]
    seg_dir = step_state["frenet"]["seg_dir"]
    seg_dir = seg_dir / torch.linalg.norm(seg_dir, dim=-1, keepdim=True).clamp_min(1e-6)
    v_long = (v_xy * seg_dir).sum(dim=-1)

    target = float(reward_cfg.get("speed_target_mps", 3.0))
    return torch.clamp(v_long, min=0.0, max=target) / target


def reward_smoothness_penalty(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Penalize large action changes (jerk) to discourage bang-bang control."""
    actions = step_state.get("actions")
    last_actions = step_state.get("last_actions")
    if actions is None or last_actions is None:
        ref = step_state["boundary"]["ey"].reshape(-1)
        return torch.zeros_like(ref)
    delta = actions - last_actions
    return -torch.sum(delta * delta, dim=-1)


def reward_tyre_slip_penalty(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
) -> torch.Tensor:
    """
    Tyre-slip penalty using per-wheel slip ratio and angle.
    """
    wheel_state = step_state["wheel_state"]
    eps = float(reward_cfg.get("slip_eps", 0.1))
    wheel_radius = float(reward_cfg.get("wheel_radius_m", 0.05))

    slip = compute_tyre_slip(wheel_state, wheel_radius=wheel_radius, slip_eps=eps)
    slip_ratio_mag = torch.clamp(torch.abs(slip[:, :4]), max=1.0)
    slip_angle_mag = torch.abs(slip[:, 4:])

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
    speed = reward_speed(step_state, reward_cfg)
    smoothness_penalty = reward_smoothness_penalty(step_state, reward_cfg)

    off_track = oob_penalty < 0.0
    progress = torch.where(off_track, torch.zeros_like(progress), progress)
    speed = torch.where(off_track, torch.zeros_like(speed), speed)

    scales = reward_cfg["reward_scales"]
    progress *= scales["progress"]
    oob_penalty *= scales["oob_penalty"]
    tyre_slip_penalty *= scales["tyre_slip_penalty"]
    speed *= scales.get("speed", 0.0)
    smoothness_penalty *= scales.get("smoothness", 0.0)

    # Single global knob to shrink overall reward magnitude (keeps the relative
    # balance between terms intact) so returns / critic targets stay O(1).
    global_scale = float(reward_cfg.get("global_reward_scale", 1.0))
    progress *= global_scale
    oob_penalty *= global_scale
    tyre_slip_penalty *= global_scale
    speed *= global_scale
    smoothness_penalty *= global_scale

    last_terms: dict[str, torch.Tensor] = {
        "progress": progress.clone(),
        "oob_penalty": oob_penalty.clone(),
        "tyre_slip_penalty": tyre_slip_penalty.clone(),
        "speed": speed.clone(),
        "smoothness": smoothness_penalty.clone(),
    }

    reward_buf += (
        progress + oob_penalty + tyre_slip_penalty + speed + smoothness_penalty
    )

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state
