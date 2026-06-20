"""In-sim observation accuracy harness (read-only).

Drives the real Genesis `F1tenthEnv` through scripted maneuvers and, at each
control step, compares every observation slice against an INDEPENDENT
re-derivation from raw Genesis state:

  - body-frame linear/angular velocity and acceleration  (hand-written quaternion
    rotation of the world-frame getters)
  - track progress / centerline heading error / signed lateral error / contact
    (brute-force all-segment Frenet projection on the real centerline)
  - tyre slip (env value vs the frame-corrected value that applies `frame_quat`)

It also reports two structural diagnostics:
  - acceleration gravity leakage (raw world az and body ax/ay at near-rest)
  - tyre-slip velocity frame (raw wheel `motion_link_vel` vs body velocity)
  - future-point closed-loop vs open-polyline arc-length seam gap

Outputs a per-observation error table to stdout and a CSV under
outputs/standalone/obs_audit/. Nothing is modified; this only reads state.

Usage:
    python scripts/audit_observations.py --num-envs 8 --precision 32
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import genesis as gs  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env.car import compute_tyre_slip  # noqa: E402
from scripts.physics_check import headless_gs_init  # noqa: E402


# --- independent math (do NOT call the env's helpers) -------------------------
def quat_inv_rotate(quat_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame (inverse of the body quat).

    quat_wxyz: (N, 4), v: (N, 3) -> (N, 3). Pure torch, independent of genesis.
    """
    w = quat_wxyz[:, 0]
    ux = -quat_wxyz[:, 1]
    uy = -quat_wxyz[:, 2]
    uz = -quat_wxyz[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    cx = uy * tz - uz * ty
    cy = uz * tx - ux * tz
    cz = ux * ty - uy * tx
    return torch.stack([vx + w * tx + cx, vy + w * ty + cy, vz + w * tz + cz], dim=-1)


def quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def brute_force_frenet_np(pos_xy: np.ndarray, centerline: np.ndarray):
    """Exhaustive nearest-segment projection over the closed centerline loop."""
    cl = centerline.astype(np.float64)
    if np.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        cl = np.concatenate([cl, cl[:1]], axis=0)
    c = cl[:-1]
    seg = cl[1:] - c
    seg_len = np.maximum(np.linalg.norm(seg, axis=-1), 1e-8)
    cumlen = np.zeros_like(seg_len)
    cumlen[1:] = np.cumsum(seg_len[:-1])
    length = float(seg_len.sum())

    pos = pos_xy.astype(np.float64)[:, :2]
    b = pos.shape[0]
    seg_len2 = np.maximum((seg * seg).sum(-1), 1e-10)
    diff = pos[:, None, :] - c[None, :, :]
    t = np.clip((diff * seg[None]).sum(-1) / seg_len2[None], 0.0, 1.0)
    proj = c[None] + t[..., None] * seg[None]
    dist2 = ((proj - pos[:, None, :]) ** 2).sum(-1)
    bi = dist2.argmin(1)
    ar = np.arange(b)
    bt = t[ar, bi]
    bproj = proj[ar, bi]
    sdir = seg[bi] / np.maximum(np.linalg.norm(seg[bi], axis=-1, keepdims=True), 1e-8)
    s = cumlen[bi] + bt * seg_len[bi]
    n_hat = np.stack([-sdir[:, 1], sdir[:, 0]], axis=-1)
    ey = ((pos - bproj) * n_hat).sum(-1)
    return {"s": s, "L": length, "seg_dir": sdir, "ey": ey, "best_idx": bi, "best_t": bt}


# --- error accumulator --------------------------------------------------------
class ErrAcc:
    def __init__(self):
        self.maxe: dict[str, float] = {}
        self.sume: dict[str, float] = {}
        self.cnt: dict[str, int] = {}

    def add(self, name: str, err):
        e = np.atleast_1d(np.asarray(err, dtype=np.float64))
        m = float(np.nanmax(np.abs(e)))
        self.maxe[name] = max(self.maxe.get(name, 0.0), m)
        self.sume[name] = self.sume.get(name, 0.0) + float(np.nansum(np.abs(e)))
        self.cnt[name] = self.cnt.get(name, 0) + e.size

    def rows(self):
        out = []
        for k in self.maxe:
            mean = self.sume[k] / max(self.cnt[k], 1)
            out.append((k, self.maxe[k], mean))
        return out


def make_action(num_envs, phase, t, device):
    a = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=device)
    if phase == "accel":
        a[:, 0] = 1.0
    elif phase == "circle":
        a[:, 0] = 0.6
        a[:, 1] = 0.5
    elif phase == "slalom":
        a[:, 0] = 0.6
        a[:, 1] = math.sin(0.3 * t)
    elif phase == "brake":
        a[:, 0] = -1.0
    elif phase == "coast":
        a[:, 0] = 0.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    ap.add_argument("--steps-per-phase", type=int, default=40)
    ap.add_argument("--track", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/standalone/obs_audit")
    ap.add_argument(
        "--stress",
        action="store_true",
        help="disable terminations/resets so the car can run far off-track "
        "(stresses the windowed Frenet projection; may hit unstable sim states)",
    )
    args = ap.parse_args()

    headless_gs_init(gs.cpu)
    # headless_gs_init calls gs.init; precision is fixed by that call's default.
    # Re-init is not allowed, so honor whatever tc_float is set.

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = cfg["env"]
    if args.track:
        env_cfg["track"] = args.track
    env_cfg.update(
        {
            "simulate_action_latency": False,
            "launch_strategy": "uniform_jittered",
            "launch_strategy_data": {"num_cars": args.num_envs},
        }
    )
    if args.stress:
        # Let the car run far off-track to stress the windowed Frenet projection.
        env_cfg.update(
            {
                "term_oob_max_consecutive": 10**9,
                "term_oob_margin_m": -100.0,
                "term_not_moving_time_s": 10**9,
                "term_heading_error_rad": 100.0,
                "episode_length": 10**6,
            }
        )

    obs_cfg = cfg["obs"]
    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=cfg["reward"],
        show_viewer=False,
    )
    control_interval = int(env_cfg["control_interval"])
    n = int(obs_cfg["future_track_num_points"])
    margin = float(obs_cfg.get("contact_margin_m", 0.08))

    print(f"num_obs={obs_cfg['num_obs']}  track={env_cfg['track']}  "
          f"tc_float={gs.tc_float}  num_envs={args.num_envs}")

    # structural diagnostic: closed-loop L vs open-polyline arc-length
    cl = env.centerline.astype(np.float64)
    closed = cl if np.linalg.norm(cl[0] - cl[-1]) < 1e-6 else np.concatenate([cl, cl[:1]])
    seg_closed = np.linalg.norm(np.diff(closed, axis=0), axis=-1).sum()
    seg_open = np.linalg.norm(np.diff(cl, axis=0), axis=-1).sum()
    seam_gap = float(seg_closed - seg_open)
    print(f"[seam] closed_loop_len={seg_closed:.3f}  open_polyline_len={seg_open:.3f}  "
          f"gap={seam_gap:.3f} m (future-point indexing uses the open length)")

    acc = ErrAcc()
    slip_frame_report = []
    accel_report = []
    n_skipped = 0
    low_speed_spreads = []  # future-point center-line spread when nearly stopped

    # on-track threshold for conditioning the centerline error (max track width)
    on_track_thr = float(max(env.w_tr_left.max(), env.w_tr_right.max()))

    env.reset()
    prev_a = torch.zeros((args.num_envs, 2))
    prev_body_v = None
    prev_done = torch.ones(args.num_envs, dtype=torch.bool)
    phases = ["accel", "circle", "slalom", "brake", "coast"]
    for phase in phases:
        for t in range(args.steps_per_phase):
            a = make_action(args.num_envs, phase, t, env.device)
            _, _, done, _ = env.step(a, n_steps=control_interval)
            done = done.detach().cpu().bool()

            obs = env.obs_buf.detach().cpu()
            quat = env.base_quat.detach()
            pos = env.base_pos.detach().cpu().numpy()

            # drop any env that went non-finite (unstable sim) so a NaN blow-up
            # does not masquerade as an observation bug
            finite = (
                torch.isfinite(obs).all(dim=1)
                & torch.isfinite(quat).all(dim=1)
                & torch.isfinite(env.base_pos.detach().cpu()).all(dim=1)
            ).numpy()
            n_skipped += int((~finite).sum())
            if not finite.any():
                continue

            fm = torch.tensor(finite)
            fm_np = finite

            # 1-2: body-frame linear/angular velocity via independent rotation
            world_v = env.car.get_vel().detach()
            world_w = env.car.get_ang().detach()
            body_v = quat_inv_rotate(quat, world_v).cpu()
            body_w = quat_inv_rotate(quat, world_w).cpu()

            acc.add("lin_vel[0:2]", (obs[fm, 0:2] - body_v[fm, :2]).numpy())
            acc.add("ang_vel[2]", (obs[fm, 2] - body_w[fm, 2]).numpy())

            # 3: acceleration. NOTE get_links_acc is stateful (re-calling it after
            # the env already read it yields inconsistent values), so we (a) check
            # obs matches the env's own stored body accel (transcription), and (b)
            # cross-check against a finite-difference of body velocity (approximate:
            # instantaneous link accel vs control-step average differ, but sign and
            # magnitude should agree). az is read from the env's own buffer.
            body_a_state = env.base_lin_acc.detach().cpu()
            acc.add("lin_acc[3:5]_obs_vs_state", (obs[fm, 3:5] - body_a_state[fm, :2]).numpy())
            if prev_body_v is not None:
                fd_mask = fm_np & (~prev_done.numpy())
                if fd_mask.any():
                    fd = (body_v[:, :2] - prev_body_v[:, :2]) / env.control_dt
                    acc.add("lin_acc[3:5]_fd_xcheck",
                            (body_a_state.numpy()[fd_mask, :2] - fd.numpy()[fd_mask]))
            accel_report.append(
                (float(body_v[fm, :2].norm(dim=-1).mean()),
                 float(body_a_state[fm, 2].mean()),
                 float(body_a_state[fm, :2].abs().mean()))
            )

            # 5: progress / 6: heading err / 7: lateral err / contact via brute force
            bf = brute_force_frenet_np(pos, env.centerline)
            ang = (2.0 * math.pi) * (bf["s"] / max(bf["L"], 1e-6))
            prog_ref = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
            acc.add("progress[7:9]", obs[:, 7:9].numpy()[fm_np] - prog_ref[fm_np])

            yaw = quat_to_yaw(quat).cpu().numpy()
            track_ang = np.arctan2(bf["seg_dir"][:, 1], bf["seg_dir"][:, 0])
            head_ref = np.arctan2(np.sin(yaw - track_ang), np.cos(yaw - track_ang))
            head_err = np.arctan2(
                np.sin(obs[:, 9].numpy() - head_ref),
                np.cos(obs[:, 9].numpy() - head_ref),
            )
            dist_err = obs[:, 10].numpy() - bf["ey"]
            on_track = (np.abs(bf["ey"]) <= on_track_thr) & fm_np
            off_track = (np.abs(bf["ey"]) > on_track_thr) & fm_np
            # seg_dir (hence heading) is only unambiguous in a segment interior; at
            # a shared vertex the env and brute-force projection may pick different
            # adjacent segments (a tie), so split heading by interior vs vertex.
            interior = (bf["best_t"] > 0.05) & (bf["best_t"] < 0.95)
            if on_track.any():
                acc.add("centerline_dist[10]_on_track", dist_err[on_track])
                m_int = on_track & interior
                m_vtx = on_track & (~interior)
                if m_int.any():
                    acc.add("centerline_angle[9]_on_track_interior", head_err[m_int])
                if m_vtx.any():
                    acc.add("centerline_angle[9]_on_track_vertex", head_err[m_vtx])
            if off_track.any():
                acc.add("centerline_angle[9]_off_track", head_err[off_track])
                acc.add("centerline_dist[10]_off_track", dist_err[off_track])

            # contact flag vs brute-force boundary distance
            bidx, bt = bf["best_idx"], bf["best_t"]
            wl = env.w_tr_left.astype(np.float64)
            wr = env.w_tr_right.astype(np.float64)
            nxt = (bidx + 1) % wl.shape[0]
            w_l_s = wl[bidx] + bt * (wl[nxt] - wl[bidx])
            w_r_s = wr[bidx] + bt * (wr[nxt] - wr[bidx])
            bdist = np.minimum(w_l_s - bf["ey"], w_r_s + bf["ey"])
            contact_ref = (bdist < margin).astype(np.float32)
            acc.add("contact[11]", obs[:, 11].numpy()[fm_np] - contact_ref[fm_np])

            # 8: tyre slip - env value vs frame-corrected value
            ss = env._get_step_state()
            ws = ss["wheel_state"]
            mlv = ws["motion_link_vel"].detach()          # (N,4,3)
            fquat = ws["frame_quat"].detach()             # (N,4,4)
            dof = ws["dof_vel"].detach()                  # (N,4)
            env_slip = ss["tyre_slip"].detach().cpu()     # (N,8) == obs[:,372:380]
            # corrected: rotate each wheel velocity world->wheel frame first
            corrected_mlv = torch.zeros_like(mlv)
            for k in range(4):
                corrected_mlv[:, k, :] = quat_inv_rotate(fquat[:, k, :], mlv[:, k, :])
            wheel_radius = float(env_cfg.get("wheel_radius", 0.05))
            slip_eps = float(cfg["reward"].get("slip_eps", 0.1))
            corrected = compute_tyre_slip(
                {"motion_link_vel": corrected_mlv, "dof_vel": dof},
                wheel_radius=wheel_radius,
                slip_eps=slip_eps,
            ).cpu()
            acc.add("tyre_slip[372:380]_env_vs_corrected",
                    (env_slip[fm] - corrected[fm]).numpy())
            # also verify obs slice == step_state slip (transcription)
            acc.add("tyre_slip[372:380]_obs_vs_state",
                    (obs[fm, 372:380] - env_slip[fm]).numpy())

            # frame diagnostic: raw rear-wheel vel x vs body vel x
            slip_frame_report.append(
                (float(mlv[fm, 0, 0].mean()),       # raw motion_link_vel x, LR wheel
                 float(corrected_mlv[fm, 0, 0].mean()),
                 float(body_v[fm, 0].mean()),
                 float(world_v[fm, 0].mean().cpu()))
            )

            # last actions echo: obs holds the PREVIOUS executed action (correct by
            # design). Compare to the action we sent last step, excluding envs that
            # reset THIS step (their last_actions were reset before obs was built)
            # and non-finite envs.
            la_mask = fm_np & (~done.numpy())
            if la_mask.any():
                acc.add("last_actions[5:7]",
                        (obs.numpy()[la_mask, 5:7] - prev_a.numpy()[la_mask]))
            # future-point preview at low speed: center samples must NOT collapse
            # (post-fix the lookahead is floored at future_track_min_lookahead_m)
            speed_xy = body_v[:, :2].norm(dim=-1)
            low = (speed_xy < 0.5) & fm
            if low.any():
                center = obs[:, 12:12 + 2 * n].view(-1, n, 2)[low]
                spread = (center - center[:, :1, :]).abs().amax(dim=(1, 2))
                low_speed_spreads.extend(spread.tolist())

            prev_a = a.detach().cpu().clone()
            prev_body_v = body_v
            prev_done = done

    # --- report ---------------------------------------------------------------
    print(f"\n(skipped {n_skipped} non-finite env-steps; stress={args.stress})")
    print("\n=== per-observation error vs independent ground truth ===")
    print(f"{'observation':<40} {'max_abs_err':>14} {'mean_abs_err':>14}")
    rows = sorted(acc.rows(), key=lambda r: -r[1])
    for name, mx, mean in rows:
        print(f"{name:<40} {mx:>14.6e} {mean:>14.6e}")

    sf = np.array(slip_frame_report)
    print("\n=== tyre-slip velocity frame diagnostic (LR wheel, mean over run) ===")
    print(f"raw motion_link_vel.x = {sf[:,0].mean():.4f}   "
          f"frame-corrected.x = {sf[:,1].mean():.4f}   "
          f"body vel.x = {sf[:,2].mean():.4f}   world vel.x = {sf[:,3].mean():.4f}")
    print("(if raw ~ world and != body, motion_link_vel is world-frame -> "
          "frame_quat rotation is missing)")

    ar = np.array(accel_report)
    print("\n=== acceleration diagnostic (mean over run) ===")
    print(f"speed={ar[:,0].mean():.3f} m/s   body az (from env buffer)={ar[:,1].mean():.3f} "
          f"m/s^2   body |a_xy|={ar[:,2].mean():.4f} m/s^2")
    print("(body az far from 0 would indicate gravity leaking into the accel obs)")

    # --- pass/fail gates for the two applied fixes ----------------------------
    slip_err = acc.maxe.get("tyre_slip[372:380]_env_vs_corrected", float("nan"))
    slip_ok = slip_err < 1e-4
    print("\n=== fix verification ===")
    print(f"[tyre slip] env vs frame-corrected max abs err = {slip_err:.3e}  "
          f"-> {'PASS' if slip_ok else 'FAIL'} (expect ~0 after frame_quat fix)")
    if low_speed_spreads:
        min_spread = float(np.min(low_speed_spreads))
        n_low = len(low_speed_spreads)
        preview_ok = min_spread > 0.5
        print(f"[future points] low-speed (<0.5 m/s) center spread: "
              f"min={min_spread:.3f} m over {n_low} samples -> "
              f"{'PASS' if preview_ok else 'FAIL'} (expect > 0.5 m, no collapse)")
    else:
        print("[future points] no low-speed steps captured this run")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "obs_audit.csv")
    with open(csv_path, "w", newline="") as f:
        wr_csv = csv.writer(f)
        wr_csv.writerow(["observation", "max_abs_err", "mean_abs_err"])
        for name, mx, mean in rows:
            wr_csv.writerow([name, mx, mean])
    print(f"\nwrote {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
