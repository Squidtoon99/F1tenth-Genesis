import os
from typing import Any

import numpy as np
import torch

import genesis as gs

# load racetracks from f1tenth_racetracks
import requests
from io import StringIO
import logging
from tqdm import tqdm

TRACKS = {}


def load_tracks(force=False) -> None:
    # https://github.com/f1tenth/f1tenth_racetracks/tree/main

    # check for local copy first (for development convenience)
    local_path = os.path.join(os.path.dirname(__file__), "tracks.pickle")
    if os.path.exists(local_path) and not force:
        try:
            import pickle

            with open(local_path, "rb") as f:
                global TRACKS
                TRACKS = pickle.load(f)
                if TRACKS:
                    logging.info(
                        f"Loaded {len(TRACKS)} tracks from local file: {local_path}"
                    )
                    return
        except Exception as e:
            logging.warning(f"Failed to load local tracks file: {e}")

    repo = "f1tenth/f1tenth_racetracks"
    api_url = f"https://api.github.com/repos/{repo}/contents/"

    response = requests.get(api_url)
    if response.status_code != 200 or not isinstance(contents := response.json(), list):
        logging.warning(f"Failed to fetch repository contents: {response.status_code}")
        return

    for item in tqdm(contents, desc="Loading tracks", colour="blue"):
        if item.get("type") == "dir" and (track_name := item.get("name")):
            try:
                centerline = requests.get(
                    "https://raw.githubusercontent.com/"
                    f"{repo}/main/{track_name}/{track_name.replace(' ', '')}_centerline.csv"
                )
                if centerline.status_code == 200:
                    TRACKS[track_name] = np.genfromtxt(
                        StringIO(centerline.text),
                        delimiter=",",
                        names=True,
                        dtype=np.float32,
                    )
                    logging.info(f"Loaded track: {track_name}")
                else:
                    logging.warning(
                        f"Failed to load centerline for {track_name}: {centerline.status_code}"
                    )
            except Exception as e:
                logging.error(f"Error loading track {track_name}: {e}")

    # Save a local copy for future runs
    if TRACKS:
        try:
            import pickle

            with open(local_path, "wb") as f:
                pickle.dump(TRACKS, f)
                logging.info(f"Saved tracks to local file: {local_path}")
        except Exception as e:
            logging.warning(f"Failed to save local tracks file: {e}")


load_tracks()


def resolve_track_data(reward_cfg: dict[str, Any], workspace_dir: str) -> np.ndarray:
    configured = reward_cfg.get("centerline_path")
    if configured is not None:
        if not configured.endswith(".csv"):
            # Check if its one of the tracks from f1tenth racetracks
            if (track := TRACKS.get(configured)) is not None:
                return track
        if not os.path.exists(configured):
            raise FileNotFoundError(f"centerline_path does not exist: {configured}")
        return np.genfromtxt(configured, delimiter=",", names=True, dtype=np.float32)

    candidates = [
        os.path.join(
            workspace_dir,
            "custom_assets",
            "SaoPaulo_centerline_with_boundaries.csv",
        ),
        (
            "x:\\F1Tenth-Managed\\F1Tenth\\source\\F1Tenth\\F1Tenth"
            "\\tasks\\manager_based\\f1tenth\\custom_assets"
            "\\SaoPaulo_centerline_with_boundaries.csv"
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)

    raise FileNotFoundError(
        "Could not locate SaoPaulo_centerline_with_boundaries.csv. "
        "Set reward_cfg['centerline_path'] explicitly."
    )


def load_track_state(
    reward_cfg: dict[str, Any],
    workspace_dir: str,
    device: torch.device,
) -> dict[str, Any]:
    data = resolve_track_data(reward_cfg, workspace_dir)
    if data is None or data.dtype.names is None:
        raise ValueError(
            f"Could not parse track csv data from {reward_cfg.get('centerline_path')}"
        )
    fields: Any = data

    required = {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"}
    missing = required.difference(set(data.dtype.names))
    if missing:
        raise ValueError(f"Track csv missing required columns: {sorted(missing)}")

    centerline = np.stack([fields["x_m"], fields["y_m"]], axis=-1).astype(np.float32)
    w_tr_right = np.asarray(fields["w_tr_right_m"], dtype=np.float32)
    w_tr_left = np.asarray(fields["w_tr_left_m"], dtype=np.float32)

    return {
        "centerline": centerline,
        "w_tr_right": w_tr_right,
        "w_tr_left": w_tr_left,
        "w_tr_left_torch": torch.as_tensor(w_tr_left, device=device, dtype=gs.tc_float),
        "w_tr_right_torch": torch.as_tensor(
            w_tr_right, device=device, dtype=gs.tc_float
        ),
        "track_geom_cache": {},
        "frenet_step_cache": {},
    }


def compute_track_boundaries(
    centerline: np.ndarray,
    w_tr_left: np.ndarray,
    w_tr_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cl = centerline.astype(np.float32)
    nxt = np.roll(cl, -1, axis=0)
    tangent = nxt - cl
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.clip(tangent_norm, 1e-8, None)
    tangent = tangent / tangent_norm

    normal = np.zeros_like(tangent)
    normal[:, 0] = -tangent[:, 1]
    normal[:, 1] = tangent[:, 0]

    left = cl + normal * w_tr_left[:, None]
    right = cl - normal * w_tr_right[:, None]
    return left, right


def draw_track_boundaries_debug(
    scene: Any,
    centerline: np.ndarray,
    w_tr_left: np.ndarray,
    w_tr_right: np.ndarray,
    reward_cfg: dict[str, Any],
) -> None:
    left, right = compute_track_boundaries(centerline, w_tr_left, w_tr_right)
    z = float(reward_cfg.get("debug_boundary_z", 0.03))
    radius = float(reward_cfg.get("debug_boundary_radius", 0.1))

    draw_line = getattr(scene, "draw_debug_line")
    n = left.shape[0]
    for i in range(n):
        j = (i + 1) % n

        l0 = (float(left[i, 0]), float(left[i, 1]), z)
        l1 = (float(left[j, 0]), float(left[j, 1]), z)
        draw_line(l0, l1, radius=radius, color=(1.0, 0.1, 0.1, 0.85))

        r0 = (float(right[i, 0]), float(right[i, 1]), z)
        r1 = (float(right[j, 0]), float(right[j, 1]), z)
        draw_line(r0, r1, radius=radius, color=(0.1, 0.3, 1.0, 0.85))


def build_track_cache(
    centerline: np.ndarray,
    device: torch.device,
    coarse_stride: int = 10,
) -> dict[str, Any]:
    cl = torch.as_tensor(centerline, device=device, dtype=gs.tc_float)
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


def frenet_projection_cached(
    base_pos: torch.Tensor,
    episode_steps_buf: torch.Tensor,
    track_state: dict[str, Any],
    device: torch.device,
    cache_id: str,
    window: int = 40,
    coarse_stride: int = 10,
) -> dict[str, Any]:
    geom = track_state["track_geom_cache"].get(cache_id)
    if geom is None or geom["coarse_stride"] != coarse_stride:
        geom = build_track_cache(
            centerline=track_state["centerline"],
            device=device,
            coarse_stride=coarse_stride,
        )
        track_state["track_geom_cache"][cache_id] = geom

    step_key = episode_steps_buf.detach().clone()
    step_entry = track_state["frenet_step_cache"].get(cache_id)
    if step_entry is not None:
        last_step = step_entry["step"]
        if last_step.shape == step_key.shape and torch.equal(last_step, step_key):
            return step_entry["data"]

    pos = base_pos[:, :2].to(device=device, dtype=gs.tc_float)
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

    data = {
        "pos": pos,
        "best_idx": best_idx,
        "best_t": best_t,
        "proj": best_proj,
        "seg_dir": seg_dir,
        "s": s,
        "L": length,
    }

    track_state["frenet_step_cache"][cache_id] = {"step": step_key, "data": data}
    return data


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
    w_tr_left_torch: torch.Tensor,
    w_tr_right_torch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t_hat = frenet_state["seg_dir"]
    n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)
    ey = ((frenet_state["pos"] - frenet_state["proj"]) * n_hat).sum(-1)

    w_l_s = interp_width_at_s(
        frenet_state["best_idx"], frenet_state["best_t"], w_tr_left_torch
    )
    w_r_s = interp_width_at_s(
        frenet_state["best_idx"],
        frenet_state["best_t"],
        w_tr_right_torch,
    )

    d_left = w_l_s - ey
    d_right = w_r_s + ey
    boundary_dist = torch.minimum(d_left, d_right)

    return {
        "ey": ey,
        "w_l_s": w_l_s,
        "w_r_s": w_r_s,
        "boundary_dist": boundary_dist,
    }


def compute_oob_from_boundary_state(
    boundary_state: dict[str, torch.Tensor],
    margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ey = boundary_state["ey"]
    w_l_s = boundary_state["w_l_s"]
    w_r_s = boundary_state["w_r_s"]

    left_oob = ey > (w_l_s - margin_m)
    right_oob = ey < -(w_r_s - margin_m)
    oob = left_oob | right_oob

    outside_left = (ey - (w_l_s - margin_m)).clamp_min(0.0)
    outside_right = (-(w_r_s - margin_m) - ey).clamp_min(0.0)
    oob_dist = outside_left + outside_right
    return oob, oob_dist


def build_step_state(
    base_pos: torch.Tensor,
    episode_steps_buf: torch.Tensor,
    track_state: dict[str, Any],
    device: torch.device,
    cache_id: str,
) -> dict[str, Any]:
    frenet_state = frenet_projection_cached(
        base_pos=base_pos,
        episode_steps_buf=episode_steps_buf,
        track_state=track_state,
        device=device,
        cache_id=cache_id,
    )
    boundary_state = build_boundary_state(
        frenet_state,
        track_state["w_tr_left_torch"],
        track_state["w_tr_right_torch"],
    )
    return {
        "frenet": frenet_state,
        "boundary": boundary_state,
    }


def invalidate_step_caches(track_state: dict[str, Any]) -> None:
    track_state["frenet_step_cache"].clear()
