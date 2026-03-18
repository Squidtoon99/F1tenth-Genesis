from typing import Any

import torch

import genesis as gs

from .utils import compute_oob_from_boundary_state


def init_reward_state(
    reward_scales: dict[str, float],
    num_envs: int,
    dt: float,
    device: torch.device,
) -> dict[str, Any]:
    scaled = {name: float(scale) * dt for name, scale in reward_scales.items()}
    episode_sums = {
        name: torch.zeros((num_envs,), dtype=gs.tc_float, device=device)
        for name in scaled.keys()
    }
    return {
        "reward_scales": scaled,
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
    last_terms: dict[str, torch.Tensor] = {}

    for name, scale in reward_state["reward_scales"].items():
        if name == "progress":
            term = reward_progress(step_state, reward_cfg)
        elif name == "oob_penalty":
            term = reward_oob_penalty(step_state, reward_cfg)
        else:
            raise ValueError(f"Unknown reward term requested in reward_scales: {name}")

        rew = term * scale
        reward_buf += rew
        reward_state["episode_sums"][name] += rew
        last_terms[name] = rew

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state
