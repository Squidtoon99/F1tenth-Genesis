import math
import os
from typing import Any

import genesis as gs
import numpy as np
import torch

from .car import (
    URDF_PATH,
    ackermann_left_right,
    setup_entity_controls,
    wheel_omegas_from_v,
)
from .observations import build_observation
from .rewards import (
    compute_rewards,
    init_reward_state,
    sync_progress_state_for_resets,
)
from .terminations import (
    compute_terminations,
    init_termination_params,
    init_termination_state,
    reset_termination_state,
)
from .utils import (
    build_step_state,
    compute_oob_from_boundary_state,
    draw_track_boundaries_debug,
    invalidate_step_caches,
    load_track_state,
)
from .validators import validate_observation_ranges, validate_reward_ranges


class F1tenthEnv:

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        show_viewer=False,
    ):
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]

        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.show_viewer = show_viewer

        self.device = gs.device if gs.device is not None else torch.device("cpu")
        self.simulate_action_latency = self.env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_steps = math.ceil(self.env_cfg["episode_length"] / self.dt)

        self.obs_scales = obs_cfg["obs_scales"]
        self.track_cache_id = self.reward_cfg.get("track_cache_id", "track")

        self.track_state = load_track_state(
            track=self.env_cfg["track"],
            workspace_dir=os.path.dirname(os.path.dirname(__file__)),
            device=self.device,
        )

        self.centerline = self.track_state["centerline"]
        self.w_tr_left = self.track_state["w_tr_left"]
        self.w_tr_right = self.track_state["w_tr_right"]

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -2.5, 1.5),
                camera_lookat=(0.4, 0.0, 0.2),
                camera_fov=35,
                res=(960, 640),
                max_FPS=int(1.0 / self.dt),
            ),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False),
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.car = self.scene.add_entity(
            gs.morphs.URDF(
                file=URDF_PATH,
                pos=self.env_cfg["car_spawn_pos"],
                euler=self.env_cfg["car_spawn_rot"],
            )
        )

        self.scene.build(n_envs=num_envs)
        if self.show_viewer:
            draw_track_boundaries_debug(
                scene=self.scene,
                centerline=self.centerline,
                w_tr_left=self.w_tr_left,
                w_tr_right=self.w_tr_right,
                reward_cfg=self.reward_cfg,
            )

        self.wheel_dofs, self.steer_dofs = setup_entity_controls(self.car)

        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_pos = torch.empty(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_quat = torch.empty(
            (self.num_envs, 4), dtype=gs.tc_float, device=gs.device
        )

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device
        )
        self.reward_buf = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        self.reset_buf = torch.zeros(
            (self.num_envs,), dtype=gs.tc_bool, device=gs.device
        )

        self.episode_steps_buf = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=gs.device
        )
        self.lap_count_buf = torch.zeros(
            (self.num_envs,), dtype=torch.int32, device=gs.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.reward_state = init_reward_state(
            reward_scales=self.reward_cfg["reward_scales"],
            num_envs=self.num_envs,
            dt=self.dt,
            device=self.device,
        )
        self.term_params = init_termination_params(self.env_cfg, self.dt)
        self.term_state = init_termination_state(self.num_envs, self.device)

        self.oob_consecutive_buf = self.term_state["oob_consecutive_buf"]
        self.not_moving_steps_buf = self.term_state["not_moving_steps_buf"]
        self.episode_sums = self.reward_state["episode_sums"]

        self.extras: dict[str, Any] = {
            "observations": {},
            "termination": {},
            "rewards": {},
            "metrics": {},
        }

        self._step_state: dict[str, Any] = {}
        self._step_state_valid = False

        self.reset()

    @staticmethod
    def _mask_to_env_ids(mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def _yaw_to_quat(self, yaw: np.ndarray) -> torch.Tensor:
        yaw = yaw + float(self.env_cfg.get("reset_yaw_offset_rad", 0.0))
        half = 0.5 * yaw
        c = np.cos(half).astype(np.float32)
        s = np.sin(half).astype(np.float32)

        quat_order = str(self.env_cfg.get("reset_quat_order", "wxyz")).lower()
        quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
        if quat_order == "wxyz":
            quat[:, 0] = c
            quat[:, 3] = s
        else:
            quat[:, 2] = s
            quat[:, 3] = c

        return torch.from_numpy(quat).to(device=self.device, dtype=gs.tc_float)

    def _sample_track_spawn(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(env_ids.numel())
        if n == 0:
            return (
                torch.empty((0, 3), dtype=gs.tc_float, device=self.device),
                torch.empty((0, 4), dtype=gs.tc_float, device=self.device),
            )

        centerline = self.centerline
        num_pts = centerline.shape[0]
        if num_pts < 3:
            raise ValueError(
                "centerline must contain at least 3 points for reset sampling"
            )

        idx = torch.randint(0, num_pts, (n,), device=self.device)
        idx_np = idx.detach().cpu().numpy().astype(np.int64)

        prev_idx = (idx_np - 1) % num_pts
        next_idx = (idx_np + 1) % num_pts

        p_prev = centerline[prev_idx]
        p_curr = centerline[idx_np]
        p_next = centerline[next_idx]

        tangent = p_next - p_prev
        tan_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tan_norm = np.clip(tan_norm, 1e-8, None)
        tangent = tangent / tan_norm

        normal = np.zeros_like(tangent)
        normal[:, 0] = -tangent[:, 1]
        normal[:, 1] = tangent[:, 0]

        w_left = self.w_tr_left[idx_np]
        w_right = self.w_tr_right[idx_np]

        spawn_margin = float(self.env_cfg.get("reset_spawn_margin_m", 0.2))
        max_left = np.maximum(w_left - spawn_margin, 0.05)
        max_right = np.maximum(w_right - spawn_margin, 0.05)
        lateral = np.random.uniform(-max_right, max_left).astype(np.float32)

        along_jitter = float(self.env_cfg.get("reset_along_track_jitter_m", 0.1))
        along = np.random.uniform(-along_jitter, along_jitter, size=(n,)).astype(
            np.float32
        )

        spawn_xy = p_curr + normal * lateral[:, None] + tangent * along[:, None]

        yaw = np.arctan2(tangent[:, 1], tangent[:, 0]).astype(np.float32)
        yaw_jitter = float(self.env_cfg.get("reset_yaw_jitter_rad", 0.2))
        yaw += np.random.uniform(-yaw_jitter, yaw_jitter, size=(n,)).astype(np.float32)

        spawn_z = float(self.env_cfg.get("car_spawn_pos", (0.0, 0.0, 0.01))[2])
        pos = np.concatenate(
            [spawn_xy.astype(np.float32), np.full((n, 1), spawn_z, dtype=np.float32)],
            axis=1,
        )
        quat = self._yaw_to_quat(yaw)

        return (
            torch.from_numpy(pos).to(device=self.device, dtype=gs.tc_float),
            quat,
        )

    def _apply_root_reset(self, env_ids: torch.Tensor):
        pos, quat = self._sample_track_spawn(env_ids)
        if pos.shape[0] == 0:
            return

        car = self.car  # type: Any
        if not hasattr(car, "set_pos") or not hasattr(car, "set_quat"):
            raise AttributeError(
                "Genesis car entity must expose set_pos and set_quat for randomized reset"
            )

        car.set_pos(pos, envs_idx=env_ids)
        car.set_quat(quat, envs_idx=env_ids, zero_velocity=True, relative=False)

    def _get_step_state(self) -> dict[str, Any]:
        if not self._step_state_valid:
            self._step_state = build_step_state(
                base_pos=self.base_pos,
                episode_steps_buf=self.episode_steps_buf,
                track_state=self.track_state,
                device=self.device,
                cache_id=self.track_cache_id,
            )
            self._step_state_valid = True
        return self._step_state

    def _update_state_buffers(self):
        car = self.car  # type: Any
        self.base_pos = car.get_pos()
        self.base_quat = car.get_quat()
        self.base_lin_vel = car.get_vel()
        self.base_ang_vel = car.get_ang()

        invalidate_step_caches(self.track_state)
        self._step_state_valid = False

    def _update_observation(self):
        step_state = self._get_step_state()
        self.obs_buf = build_observation(
            num_obs=self.num_obs,
            num_envs=self.num_envs,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            centerline=self.centerline,
            obs_cfg=self.obs_cfg,
            step_state=step_state,
            device=self.device,
        )

    def _compute_rewards(self):
        step_state = self._get_step_state()
        self.reward_buf, self._step_state = compute_rewards(
            step_state=step_state,
            reward_cfg=self.reward_cfg,
            reward_state=self.reward_state,
            episode_steps_buf=self.episode_steps_buf,
            lap_count_buf=self.lap_count_buf,
        )
        self.extras["rewards"]["total"] = self.reward_buf
        self.extras["rewards"]["terms"] = self.reward_state.get("last_reward_terms", {})

    def _compute_terminations(self):
        step_state = self._get_step_state()
        self.reset_buf, self.extras["termination"], self.extras["time_outs"] = (
            compute_terminations(
                step_state=step_state,
                episode_steps_buf=self.episode_steps_buf,
                max_episode_steps=self.max_episode_steps,
                base_pos=self.base_pos,
                base_quat=self.base_quat,
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                lap_count_buf=self.lap_count_buf,
                term_state=self.term_state,
                term_params=self.term_params,
            )
        )

        boundary = step_state["boundary"]
        oob_mask, oob_dist = compute_oob_from_boundary_state(
            boundary,
            margin_m=float(self.term_params["term_oob_margin_m"]),
        )
        progress_ds = step_state.get(
            "progress_ds",
            torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device),
        )
        speed_xy = torch.linalg.norm(self.base_lin_vel[:, :2], dim=-1)

        self.extras["metrics"] = {
            "progress_ds": progress_ds,
            "oob_dist": oob_dist,
            "oob_mask": oob_mask.to(dtype=gs.tc_float),
            "boundary_dist": boundary["boundary_dist"],
            "lateral_error": boundary["ey"],
            "speed_xy": speed_xy,
            "episode_steps": self.episode_steps_buf.to(dtype=gs.tc_float),
            "lap_count": self.lap_count_buf.to(dtype=gs.tc_float),
        }

    def reset(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        elif isinstance(envs_idx, (list, tuple, np.ndarray)):
            mask = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
            if len(envs_idx) > 0:
                mask[list(envs_idx)] = True
            envs_idx = mask

        if envs_idx.sum().item() == 0:
            return self.obs_buf, self.extras

        env_ids = self._mask_to_env_ids(envs_idx)
        self._apply_root_reset(env_ids)
        self.scene.step()

        self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
        self.actions.masked_fill_(envs_idx[:, None], 0.0)
        self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
        self.episode_steps_buf.masked_fill_(envs_idx, 0)
        self.lap_count_buf.masked_fill_(envs_idx, 0)
        self.reset_buf.masked_fill_(envs_idx, False)

        reset_termination_state(self.term_state, envs_idx)

        for value in self.reward_state["episode_sums"].values():
            value.masked_fill_(envs_idx, 0.0)

        self._update_state_buffers()
        step_state = self._get_step_state()
        sync_progress_state_for_resets(
            reward_state=self.reward_state,
            step_state=step_state,
            episode_steps_buf=self.episode_steps_buf,
            reset_mask=envs_idx,
        )

        self._update_observation()
        validate_observation_ranges(self.obs_buf)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def _apply_actions(self, exec_actions: torch.Tensor):
        throttle = exec_actions[:, 0].detach().cpu().numpy()
        steer = exec_actions[:, 1].detach().cpu().numpy()

        steer_targets = np.zeros(
            (self.num_envs, len(self.steer_dofs)), dtype=np.float32
        )
        wheel_omegas = np.zeros((self.num_envs, len(self.wheel_dofs)), dtype=np.float32)

        for i in range(self.num_envs):
            left, right = ackermann_left_right(
                delta_center=float(steer[i]) * self.env_cfg.get("max_steer", 0.35),
                L=self.env_cfg.get("wheelbase", 0.325),
                W=self.env_cfg.get("track_width", 0.20),
            )
            steer_targets[i] = np.array([left, right], dtype=np.float32)
            wheel_omegas[i] = wheel_omegas_from_v(
                delta_center=float(steer[i]) * self.env_cfg.get("max_steer", 0.35),
                v=float(throttle[i]) * self.env_cfg.get("max_speed", 5.0),
                r=self.env_cfg.get("wheel_radius", 0.05),
                L=self.env_cfg.get("wheelbase", 0.325),
                W=self.env_cfg.get("track_width", 0.20),
            )

        car = self.car  # type: Any
        car.control_dofs_velocity(
            torch.from_numpy(wheel_omegas).to(device=gs.device), self.wheel_dofs
        )
        car.control_dofs_position(
            torch.from_numpy(steer_targets).to(device=gs.device), self.steer_dofs
        )

    def step(self, actions):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )

        self._apply_actions(exec_actions)
        self.scene.step()
        self.episode_steps_buf += 1
        self._update_state_buffers()

        self._compute_rewards()
        validate_reward_ranges(self.reward_buf, self.reward_state["last_reward_terms"])
        self._compute_terminations()

        done = self.reset_buf.clone()
        if done.any().item():
            self.reset(done)
        else:
            self._update_observation()
            validate_observation_ranges(self.obs_buf)
            self.extras["observations"]["critic"] = self.obs_buf

        self.last_actions.copy_(self.actions)
        return self.obs_buf, self.reward_buf, done, self.extras

    def close(self):
        self.scene.destroy()
