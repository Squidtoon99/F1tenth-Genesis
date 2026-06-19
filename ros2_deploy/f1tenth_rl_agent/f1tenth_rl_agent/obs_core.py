"""Genesis-free port of the F1TENTH observation pipeline.

This module reproduces ``f1tenth_env/observations.py`` and the Frenet / boundary /
future-point math from ``f1tenth_env/utils.py`` using only ``torch`` and ``numpy``
(no ``genesis`` dependency), so it can run inside a ROS 2 Humble container for
inference.

The output of :func:`build_observation` is numerically identical (within float
tolerance) to ``f1tenth_env.observations.build_observation`` for the same inputs.
A parity test in ``test/test_obs_parity.py`` enforces this.

All tensors are CPU float32. Functions accept batched inputs (shape ``[B, ...]``)
but the ROS deployment uses ``B = 1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

TC_FLOAT = torch.float32


# --- quaternion helpers -------------------------------------------------------
def quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about z) from a (w, x, y, z) quaternion. Shape [B, 4] -> [B]."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert a ROS (x, y, z, w) quaternion to genesis (w, x, y, z) order."""
    return torch.stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]], dim=-1
    )


# --- track cache + Frenet projection (port of utils.py) -----------------------
def build_track_cache(
    centerline: np.ndarray,
    device: torch.device,
    coarse_stride: int = 10,
) -> dict[str, Any]:
    cl = torch.as_tensor(centerline, device=device, dtype=TC_FLOAT)
    if torch.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        cl = torch.cat([cl, cl[0:1]], dim=0)

    c = cl[:-1]
    d = cl[1:]
    seg = d - c
    seg_len = torch.linalg.norm(seg, dim=-1).clamp_min(1e-8)
    cumlen = torch.zeros_like(seg_len)
    cumlen[1:] = torch.cumsum(seg_len[:-1], dim=0)
    length = seg_len.sum()
    m = int(c.shape[0])

    coarse_idx = torch.arange(0, m, coarse_stride, device=device)
    coarse_pts = c[coarse_idx]

    return {
        "C": c,
        "seg": seg,
        "seg_len": seg_len,
        "cumlen": cumlen,
        "L": length,
        "M": m,
        "coarse_stride": coarse_stride,
        "coarse_idx": coarse_idx,
        "coarse_pts": coarse_pts,
    }


def frenet_projection(
    base_pos: torch.Tensor,
    geom: dict[str, Any],
    device: torch.device,
    window: int = 40,
) -> dict[str, Any]:
    pos = base_pos[:, :2].to(device=device, dtype=TC_FLOAT)
    batch = pos.shape[0]

    c_all = geom["C"]
    seg_all = geom["seg"]
    seg_len_all = geom["seg_len"]
    cumlen_all = geom["cumlen"]
    length = geom["L"]
    m = geom["M"]
    coarse_pts = geom["coarse_pts"]
    coarse_idx = geom["coarse_idx"]

    diffc = coarse_pts.unsqueeze(0) - pos.unsqueeze(1)
    dist2c = (diffc * diffc).sum(dim=-1)
    j = dist2c.argmin(dim=-1)
    i0 = coarse_idx[j]

    offsets = torch.arange(-window, window + 1, device=device)
    cand = (i0.unsqueeze(1) + offsets.unsqueeze(0)) % m

    c = c_all[cand]
    seg = seg_all[cand]
    seg_len2 = (seg * seg).sum(dim=-1).clamp_min(1e-10)

    p = pos.unsqueeze(1)
    t = ((p - c) * seg).sum(dim=-1) / seg_len2
    t = t.clamp(0.0, 1.0)

    proj = c + t.unsqueeze(-1) * seg
    dist2 = ((proj - p) ** 2).sum(dim=-1)

    k = dist2.argmin(dim=-1)
    ar = torch.arange(batch, device=device)
    best_idx = cand[ar, k]
    best_t = t[ar, k]
    best_proj = proj[ar, k]

    best_seg = seg_all[best_idx]
    seg_dir = best_seg / torch.linalg.norm(best_seg, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    s = cumlen_all[best_idx] + best_t * seg_len_all[best_idx]

    return {
        "pos": pos,
        "best_idx": best_idx,
        "best_t": best_t,
        "proj": best_proj,
        "seg_dir": seg_dir,
        "s": s,
        "L": length,
    }


def interp_width_at_s(
    best_idx: torch.Tensor,
    best_t: torch.Tensor,
    widths: torch.Tensor,
) -> torch.Tensor:
    w0 = widths[best_idx]
    w1 = widths[(best_idx + 1) % widths.shape[0]]
    return w0 + best_t * (w1 - w0)


def build_boundary_state(
    frenet_state: dict[str, Any],
    w_tr_left: torch.Tensor,
    w_tr_right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t_hat = frenet_state["seg_dir"]
    n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)
    ey = ((frenet_state["pos"] - frenet_state["proj"]) * n_hat).sum(-1)

    w_l_s = interp_width_at_s(frenet_state["best_idx"], frenet_state["best_t"], w_tr_left)
    w_r_s = interp_width_at_s(frenet_state["best_idx"], frenet_state["best_t"], w_tr_right)

    d_left = w_l_s - ey
    d_right = w_r_s + ey
    boundary_dist = torch.minimum(d_left, d_right)

    return {
        "ey": ey,
        "w_l_s": w_l_s,
        "w_r_s": w_r_s,
        "boundary_dist": boundary_dist,
    }


# --- observation components (port of observations.py) -------------------------
def obs_track_progress(
    centerline: np.ndarray,
    base_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    centerline_t = torch.as_tensor(centerline, device=device, dtype=TC_FLOAT)
    points = centerline_t.unsqueeze(0) - base_pos[:, :2].unsqueeze(1)
    distances = torch.linalg.norm(points, dim=-1)
    closest_idx = torch.argmin(distances, dim=-1)

    progress_ratio = closest_idx.to(dtype=TC_FLOAT) / max((centerline_t.shape[0] - 1), 1)
    angle = 2.0 * torch.tensor(np.pi, dtype=TC_FLOAT, device=device) * progress_ratio
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def obs_centerline_angle(
    frenet_state: dict[str, Any],
    base_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    track_angle = torch.atan2(
        frenet_state["seg_dir"][:, 1], frenet_state["seg_dir"][:, 0]
    )
    yaw = quat_wxyz_to_yaw(base_quat_wxyz)
    theta_err = yaw - track_angle
    theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))
    return theta_err.unsqueeze(-1)


def obs_centerline_distance(boundary_state: dict[str, Any]) -> torch.Tensor:
    return boundary_state["ey"].unsqueeze(-1)


def obs_contact_flag(
    boundary_state: dict[str, Any], obs_cfg: dict[str, Any]
) -> torch.Tensor:
    boundary_dist = boundary_state["boundary_dist"]
    contact_margin = float(obs_cfg.get("contact_margin_m", 0.08))
    return (boundary_dist < contact_margin).float().unsqueeze(-1)


def obs_future_track_points(
    centerline: np.ndarray,
    base_pos: torch.Tensor,
    base_quat_wxyz: torch.Tensor,
    base_lin_vel: torch.Tensor,
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    centerline_t = torch.as_tensor(centerline, device=device, dtype=TC_FLOAT)
    robot_pos = base_pos[:, :2]
    lin_vel = base_lin_vel[:, :2]
    yaw = quat_wxyz_to_yaw(base_quat_wxyz)

    n = centerline_t.shape[0]
    seg = centerline_t[1:] - centerline_t[:-1]
    seg_len = torch.linalg.vector_norm(seg, dim=-1)
    cumlen = torch.cat(
        [
            torch.zeros(1, device=device, dtype=TC_FLOAT),
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

    steps = torch.arange(1, samples + 1, device=device, dtype=TC_FLOAT) / samples
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


class ObservationBuilder:
    """Stateful helper that owns the track geometry and builds observations.

    Construct once with the track centerline + widths, then call
    :meth:`build` each control step with the current pose, velocities and last action.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        w_tr_left: np.ndarray,
        w_tr_right: np.ndarray,
        obs_cfg: dict[str, Any] | None = None,
        device: torch.device | None = None,
        coarse_stride: int = 10,
    ):
        self.device = device or torch.device("cpu")
        self.centerline = np.asarray(centerline, dtype=np.float32)
        self.obs_cfg = obs_cfg or {}
        self.num_obs = int(self.obs_cfg.get("num_obs", 372))
        self.w_tr_left = torch.as_tensor(w_tr_left, device=self.device, dtype=TC_FLOAT)
        self.w_tr_right = torch.as_tensor(w_tr_right, device=self.device, dtype=TC_FLOAT)
        self.geom = build_track_cache(
            self.centerline, device=self.device, coarse_stride=coarse_stride
        )

    def build(
        self,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        base_lin_acc: torch.Tensor,
        last_actions: torch.Tensor,
        base_pos: torch.Tensor,
        base_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        frenet_state = frenet_projection(base_pos, self.geom, self.device)
        boundary_state = build_boundary_state(
            frenet_state, self.w_tr_left, self.w_tr_right
        )

        obs = torch.cat(
            (
                base_lin_vel[:, :2],
                base_ang_vel[:, 2:3],
                base_lin_acc[:, :2],
                last_actions,
                obs_track_progress(self.centerline, base_pos, self.device),
                obs_centerline_angle(frenet_state, base_quat_wxyz),
                obs_centerline_distance(boundary_state),
                obs_contact_flag(boundary_state, self.obs_cfg),
                obs_future_track_points(
                    self.centerline,
                    base_pos,
                    base_quat_wxyz,
                    base_lin_vel,
                    self.obs_cfg,
                    self.device,
                ),
            ),
            dim=-1,
        )

        actual_obs_dim = int(obs.shape[1])
        if actual_obs_dim != self.num_obs:
            raise ValueError(
                f"Observation shape mismatch: expected num_obs={self.num_obs}, "
                f"got {actual_obs_dim}."
            )
        return obs

    def frenet(self, base_pos: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        """Expose Frenet + boundary state (used by evaluation_node)."""
        frenet_state = frenet_projection(base_pos, self.geom, self.device)
        boundary_state = build_boundary_state(
            frenet_state, self.w_tr_left, self.w_tr_right
        )
        return frenet_state, boundary_state
