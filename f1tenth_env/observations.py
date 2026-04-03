from typing import Any

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz


def obs_track_progress(
    centerline: np.ndarray,
    base_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    centerline_t = torch.as_tensor(centerline, device=device, dtype=gs.tc_float)
    points = centerline_t.unsqueeze(0) - base_pos[:, :2].unsqueeze(1)
    distances = torch.linalg.norm(points, dim=-1)
    closest_idx = torch.argmin(distances, dim=-1)

    progress_ratio = closest_idx.to(dtype=gs.tc_float) / max(
        (centerline_t.shape[0] - 1), 1
    )
    angle = 2.0 * torch.tensor(np.pi, dtype=gs.tc_float, device=device) * progress_ratio
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def obs_centerline_angle(
    step_state: dict[str, Any],
    base_quat: torch.Tensor,
) -> torch.Tensor:
    cached = step_state.get("centerline_angle")
    if cached is not None:
        return cached

    frenet_state = step_state["frenet"]
    track_angle = torch.atan2(
        frenet_state["seg_dir"][:, 1], frenet_state["seg_dir"][:, 0]
    )

    euler_xyz = quat_to_xyz(base_quat, rpy=True, degrees=False)
    yaw = euler_xyz[:, 2]

    theta_err = yaw - track_angle
    theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))
    out = theta_err.unsqueeze(-1)
    step_state["centerline_angle"] = out
    return out


def obs_centerline_distance(
    step_state: dict[str, Any],
) -> torch.Tensor:
    return step_state["boundary"]["ey"].unsqueeze(-1)


def obs_contact_flag(
    step_state: dict[str, Any], obs_cfg: dict[str, Any]
) -> torch.Tensor:
    boundary_dist = step_state["boundary"]["boundary_dist"]
    contact_margin = float(obs_cfg.get("contact_margin_m", 0.08))
    return (boundary_dist < contact_margin).float().unsqueeze(-1)


def obs_future_track_points(
    centerline: np.ndarray,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    centerline_t = torch.as_tensor(centerline, device=device, dtype=gs.tc_float)
    robot_pos = base_pos[:, :2]
    lin_vel = base_lin_vel[:, :2]
    yaw = quat_to_xyz(base_quat, rpy=True, degrees=False)[:, 2]

    n = centerline_t.shape[0]
    seg = centerline_t[1:] - centerline_t[:-1]
    seg_len = torch.linalg.vector_norm(seg, dim=-1)
    cumlen = torch.cat(
        [
            torch.zeros(1, device=device, dtype=gs.tc_float),
            torch.cumsum(seg_len, dim=0),
        ],
        dim=0,
    )
    total_len = cumlen[-1].clamp(min=1e-6)

    batch = robot_pos.shape[0]
    samples = int(obs_cfg.get("future_track_num_points", 60))
    horizon_s = float(obs_cfg.get("future_track_horizon_s", 6.0))
    track_width = float(obs_cfg.get("future_track_width", 2.0))

    dists = torch.linalg.vector_norm(
        centerline_t.unsqueeze(0) - robot_pos.unsqueeze(1), dim=-1
    )
    closest_idx = torch.argmin(dists, dim=-1)
    s0 = cumlen[closest_idx]

    speed = torch.linalg.vector_norm(lin_vel, dim=-1)
    lookahead = speed * horizon_s

    steps = torch.arange(1, samples + 1, device=device, dtype=gs.tc_float) / samples
    s_targets = s0.unsqueeze(1) + lookahead.unsqueeze(1) * steps.unsqueeze(0)
    s_targets = torch.remainder(s_targets, total_len)

    seg_idx = torch.searchsorted(cumlen, s_targets, right=True) - 1
    seg_idx = seg_idx.clamp(min=0, max=n - 2)

    seg_idx_flat = seg_idx.reshape(-1)
    p0 = centerline_t[seg_idx_flat]
    p1 = centerline_t[seg_idx_flat + 1]

    seg_len_sel = seg_len[seg_idx_flat].clamp(min=1e-8)
    s_base = cumlen[seg_idx_flat]
    alpha = ((s_targets.reshape(-1) - s_base) / seg_len_sel).unsqueeze(-1)

    center_pts = p0 + alpha * (p1 - p0)
    tangents = (p1 - p0) / seg_len_sel.unsqueeze(-1)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)

    half_w = 0.5 * track_width
    left_pts = center_pts + half_w * normals
    right_pts = center_pts - half_w * normals

    center_pts = center_pts.view(batch, samples, 2)
    left_pts = left_pts.view(batch, samples, 2)
    right_pts = right_pts.view(batch, samples, 2)

    cos_y = torch.cos(yaw).view(batch, 1, 1)
    sin_y = torch.sin(yaw).view(batch, 1, 1)

    def world_to_ego(points: torch.Tensor) -> torch.Tensor:
        d = points - robot_pos.unsqueeze(1)
        x = d[..., 0:1]
        y = d[..., 1:2]
        x_p = cos_y * x + sin_y * y
        y_p = -sin_y * x + cos_y * y
        return torch.cat([x_p, y_p], dim=-1)

    center_ego = world_to_ego(center_pts)
    left_ego = world_to_ego(left_pts)
    right_ego = world_to_ego(right_pts)

    all_ego = torch.stack([center_ego, left_ego, right_ego], dim=1)
    return all_ego.reshape(batch, -1)


def build_observation(
    num_obs: int,
    num_envs: int,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    last_actions: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    centerline: np.ndarray,
    obs_cfg: dict[str, Any],
    step_state: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    obs = torch.concatenate(
        (
            base_lin_vel[:, :2],
            base_ang_vel[:, 2:3],
            last_actions,
            obs_track_progress(centerline, base_pos, device),
            obs_centerline_angle(step_state, base_quat),
            obs_centerline_distance(step_state),
            obs_contact_flag(step_state, obs_cfg),
            obs_future_track_points(
                centerline,
                base_pos,
                base_quat,
                base_lin_vel,
                obs_cfg,
                device,
            ),
        ),
        dim=-1,
    )

    actual_obs_dim = int(obs.shape[1])
    if actual_obs_dim != num_obs:
        raise ValueError(
            "Observation shape mismatch: "
            f"expected num_obs={num_obs}, got {actual_obs_dim}. "
            "Check obs_cfg['num_obs'] and observation component sizes."
        )

    return obs
