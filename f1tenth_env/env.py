import math
import os
from typing import Any

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch

from .car import (
    URDF_PATH,
    ackermann_left_right,
    compute_dissipative_force_world,
    compute_tyre_slip,
    compute_wheel_torques,
    setup_entity_controls,
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
import rerun as rr

class F1tenthEnv:

    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        show_viewer=False,
        enable_recording=False,
    ):
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]

        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.show_viewer = show_viewer

        self.device = gs.device if gs.device is not None else torch.device("cpu")
        self.simulate_action_latency = self.env_cfg.get(
            "simulate_action_latency", False
        )
        self.dt = self.env_cfg.get("sim_dt", 0.01)
        self.control_interval = int(self.env_cfg.get("control_interval", 10))
        self.control_dt = self.dt * self.control_interval
        self.max_episode_steps = math.ceil(
            self.env_cfg["episode_length"] / self.control_dt
        )

        self.spawn_strategy = self.env_cfg.get("launch_strategy", "uniform_jittered")
        self.spawn_data = self.env_cfg.get("launch_strategy_data", {"num_cars": 20})

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

        # On-device copies for sync-free reset sampling.
        self.centerline_t = torch.as_tensor(
            self.centerline, device=self.device, dtype=gs.tc_float
        )
        self.num_pts = int(self.centerline_t.shape[0])
        self.w_tr_left_torch = self.track_state["w_tr_left_torch"]
        self.w_tr_right_torch = self.track_state["w_tr_right_torch"]
        self.spawn_z = float(self.env_cfg.get("car_spawn_pos", (0.0, 0.0, 0.01))[2])

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -5.0, 3.5),
                camera_lookat=(0.4, 0.0, 0.2),
                camera_fov=35,
                res=(960, 640),
                max_FPS=int(1.0 / self.dt),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                constraint_solver=gs.constraint_solver.Newton,
                # Genesis requires constraint_timeconst >= 2 * dt for a stable
                # Newton solve; anything smaller can yield NaN constraint forces.
                constraint_timeconst=float(
                    self.env_cfg.get("constraint_timeconst", max(0.02, 2.0 * self.dt))
                ),
                iterations=int(self.env_cfg.get("solver_iterations", 50)),
                ls_iterations=int(self.env_cfg.get("solver_ls_iterations", 50)),
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt, substeps=int(self.env_cfg.get("sim_substeps", 10))
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=bool(self.env_cfg.get("show_fps", False)),
            ),
            show_viewer=show_viewer,
        )

        self.ground = self.scene.add_entity(gs.morphs.Plane())
        tire_friction = float(self.env_cfg.get("tire_friction", 0.7))
        self.ground.set_friction(tire_friction)

        self.car = self.scene.add_entity(
            gs.morphs.URDF(
                file=URDF_PATH,
                pos=self.env_cfg["car_spawn_pos"],
                euler=self.env_cfg["car_spawn_rot"],
                recompute_inertia=False,
                default_armature=0.0,
            )
        )

        if self.env_cfg.get("opponent_strategy") is not None:
            self.opponent = self.scene.add_entity(
                gs.morphs.URDF(
                    file=URDF_PATH,
                    pos=self.env_cfg["car_spawn_pos"],
                    euler=self.env_cfg["car_spawn_rot"],
                    recompute_inertia=False,
                )
            )
        else:
            self.opponent = None

        if self.scene.viewer and show_viewer:
            self.scene.viewer.follow_entity(self.car)

        if enable_recording:
            self.cam1 = self.scene.add_camera(
                res=(1024, 1024), pos=(2, 0, 1), lookat=(0, 0, 0.5), debug=True
            )
        else:
            self.cam1 = None
        self.scene.build(n_envs=num_envs)
        if self.show_viewer or enable_recording:
            draw_track_boundaries_debug(
                scene=self.scene,
                centerline=self.centerline,
                w_tr_left=self.w_tr_left,
                w_tr_right=self.w_tr_right,
                reward_cfg=self.reward_cfg,
            )
        if self.cam1 is not None:
            self.cam1.start_recording()

        self.wheel_dofs, self.steer_dofs = setup_entity_controls(
            self.car, self.env_cfg
        )
        if self.opponent is not None:
            setup_entity_controls(self.opponent, self.env_cfg)

        self.vehicle_mass = 3.74
        self.base_link_idx = self.car.get_link("base_link").idx
        self.base_link_idx_local = self.car.get_link("base_link").idx_local
        self.root_dof_vel_idx = list(
            self.car.get_joint("root_joint").dofs_idx_local[3:6]
        )
        self.slip_motion_link_idx = [
            self.car.get_link("left_rear_wheel").idx_local,
            self.car.get_link("right_rear_wheel").idx_local,
            self.car.get_link("left_front_wheel").idx_local,
            self.car.get_link("right_front_wheel").idx_local,
        ]
        self.slip_frame_link_idx = [
            self.car.get_link("base_link").idx_local,
            self.car.get_link("base_link").idx_local,
            self.car.get_link("left_steering_hinge").idx_local,
            self.car.get_link("right_steering_hinge").idx_local,
        ]
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )

        self.base_lin_acc = torch.zeros(
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

        self.wheel_omega_cmd = torch.zeros(
            (self.num_envs, len(self.wheel_dofs)),
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.steer_state = torch.zeros(
            (self.num_envs,), dtype=gs.tc_float, device=gs.device
        )
        t_delta = float(self.env_cfg.get("t_delta", 0.1))
        self.steer_lag_alpha = self.control_dt / (t_delta + self.control_dt)

        self.reward_state = init_reward_state(
            reward_scales=self.reward_cfg["reward_scales"],
            num_envs=self.num_envs,
            device=self.device,
        )
        self.term_params = init_termination_params(self.env_cfg, self.control_dt)
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
        self._eval_launch_initialized = False

        # Optionally compile the pure-tensor observation builder. Default off; the
        # eager function is used unless explicitly enabled for the GPU target.
        if bool(self.env_cfg.get("compile_obs", False)):
            self._build_observation = torch.compile(
                build_observation, dynamic=False
            )
        else:
            self._build_observation = build_observation

        self.reset()

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """World-frame quaternion (B, 4) from yaw (B,), fully on-device."""
        yaw = yaw + float(self.env_cfg.get("reset_yaw_offset_rad", 0.0))
        half = 0.5 * yaw
        c = torch.cos(half)
        s = torch.sin(half)

        quat = torch.zeros((yaw.shape[0], 4), dtype=gs.tc_float, device=self.device)
        if str(self.env_cfg.get("reset_quat_order", "wxyz")).lower() == "wxyz":
            quat[:, 0] = c
            quat[:, 3] = s
        else:
            quat[:, 2] = s
            quat[:, 3] = c

        return quat

    def _reset_speed_range(self) -> tuple[float, float]:
        speed_range = self.spawn_data.get("mps_range")
        if speed_range is not None and len(speed_range) == 2:
            v_min, v_max = float(speed_range[0]), float(speed_range[1])
        else:
            v_min = float(self.env_cfg.get("reset_speed_min_mps", 0.0))
            v_max = float(self.env_cfg.get("reset_speed_max_mps", 0.0))
        if v_max < v_min:
            v_min, v_max = v_max, v_min
        return v_min, v_max

    def _sample_reset_speed(self) -> torch.Tensor:
        v_min, v_max = self._reset_speed_range()
        if abs(v_min) < 1e-8 and abs(v_max) < 1e-8:
            return torch.zeros((self.num_envs,), dtype=gs.tc_float, device=self.device)
        return (
            torch.rand((self.num_envs,), device=self.device, dtype=gs.tc_float)
            * (v_max - v_min)
            + v_min
        )

    def _sample_track_spawn_batch(
        self, centerline_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-batch (B,) on-device track spawn -> (pos (B, 3), quat (B, 4))."""
        if self.num_pts < 3:
            raise ValueError(
                "centerline must contain at least 3 points for reset sampling"
            )
        B = self.num_envs
        if centerline_idx is None:
            idx = torch.randint(0, self.num_pts, (B,), device=self.device)
        else:
            idx = centerline_idx.to(device=self.device, dtype=torch.long)

        prev_idx = (idx - 1) % self.num_pts
        next_idx = (idx + 1) % self.num_pts
        p_prev = self.centerline_t[prev_idx]
        p_curr = self.centerline_t[idx]
        p_next = self.centerline_t[next_idx]

        tangent = p_next - p_prev
        tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(
            1e-8
        )
        normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)

        w_left = self.w_tr_left_torch[idx]
        w_right = self.w_tr_right_torch[idx]
        spawn_margin = float(self.env_cfg.get("reset_spawn_margin_m", 0.2))
        max_left = (w_left - spawn_margin).clamp_min(0.05)
        max_right = (w_right - spawn_margin).clamp_min(0.05)
        lateral = (
            torch.rand((B,), device=self.device, dtype=gs.tc_float)
            * (max_left + max_right)
            - max_right
        )

        along_jitter = float(self.env_cfg.get("reset_along_track_jitter_m", 0.1))
        along = (
            torch.rand((B,), device=self.device, dtype=gs.tc_float) * 2.0 - 1.0
        ) * along_jitter

        spawn_xy = p_curr + normal * lateral.unsqueeze(1) + tangent * along.unsqueeze(1)

        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        yaw_jitter = float(self.env_cfg.get("reset_yaw_jitter_rad", 0.2))
        yaw = (
            yaw
            + (torch.rand((B,), device=self.device, dtype=gs.tc_float) * 2.0 - 1.0)
            * yaw_jitter
        )

        z = torch.full((B, 1), self.spawn_z, dtype=gs.tc_float, device=self.device)
        pos = torch.cat([spawn_xy, z], dim=1)
        quat = self._yaw_to_quat(yaw)
        return pos, quat

    def _sample_spawn_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return full-batch (pos, quat, speed, preserve_buffers)."""
        B = self.num_envs
        preserve_buffers = False
        match self.spawn_strategy:
            case "fixed":
                pos = (
                    torch.tensor(
                        self.env_cfg["car_spawn_pos"],
                        dtype=gs.tc_float,
                        device=self.device,
                    )
                    .reshape(1, 3)
                    .expand(B, 3)
                    .contiguous()
                )
                euler = torch.tensor(
                    self.env_cfg["car_spawn_rot"],
                    dtype=gs.tc_float,
                    device=self.device,
                )
                q = gu.xyz_to_quat(euler, rpy=True, degrees=False)
                quat = q.reshape(1, 4).expand(B, 4).contiguous()
            case "eval_launch":
                if not self._eval_launch_initialized:
                    centerline_idx = torch.zeros(
                        (B,), dtype=torch.long, device=self.device
                    )
                    self._eval_launch_initialized = True
                else:
                    centerline_idx = self._closest_centerline_indices(
                        self.base_pos[:, :2]
                    )
                    preserve_buffers = True
                pos, quat = self._sample_track_spawn_batch(centerline_idx=centerline_idx)
            case _:
                pos, quat = self._sample_track_spawn_batch()
        speed = self._sample_reset_speed()
        return pos, quat, speed, preserve_buffers

    def _closest_centerline_indices(self, pos_xy: torch.Tensor) -> torch.Tensor:
        deltas = pos_xy[:, None, :] - self.centerline_t[None, :, :2]
        dist_sq = (deltas * deltas).sum(dim=-1)
        return torch.argmin(dist_sq, dim=1).to(dtype=torch.long)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Sync-free, mask-based reset of car state and per-env buffers.

        ``mask`` is a boolean (num_envs,) tensor. All Genesis setters and buffer
        writes operate on the full batch and select rows via ``mask``/torch.where,
        so this runs with no host-device synchronization regardless of how many
        envs are resetting. State setters use immediate forward kinematics so the
        post-reset pose is readable by ``_update_state_buffers`` without a
        physics ``scene.step``.
        """
        car = self.car  # type: Any
        if not hasattr(car, "set_pos") or not hasattr(car, "set_quat"):
            raise AttributeError(
                "Genesis car entity must expose set_pos and set_quat for randomized reset"
            )

        pos, quat, speed, preserve_buffers = self._sample_spawn_batch()
        m = mask.unsqueeze(1)

        # Pose: full-batch source + boolean mask is sync-free for set_pos/set_quat,
        # and zero_velocity=True clears every dof velocity for the masked envs (so a
        # rest spawn needs no explicit dof setters). Immediate forward kinematics
        # (default skip_forward=False) makes the new pose readable without scene.step.
        car.set_pos(pos, envs_idx=mask, zero_velocity=True, relative=False)
        car.set_quat(quat, envs_idx=mask, zero_velocity=True, relative=False)

        if self.opponent is not None:
            opp_pos, opp_quat = self._sample_track_spawn_batch()
            self.opponent.set_pos(
                opp_pos, envs_idx=mask, zero_velocity=True, relative=False
            )
            self.opponent.set_quat(
                opp_quat, envs_idx=mask, zero_velocity=True, relative=False
            )

        # Non-zero launch speed needs explicit dof velocity/steering setters, which
        # in Genesis 0.4.3 require index-based envs (one nonzero sync). This path is
        # only taken when a launch-speed range is configured, so the common rest
        # spawn stays fully sync-free.
        v_min, v_max = self._reset_speed_range()
        launch = not (abs(v_min) < 1e-8 and abs(v_max) < 1e-8)
        if launch:
            self._apply_launch_velocity(mask, quat, speed)

        # Per-env buffers (masked writes, no sync).
        self.reset_buf = self.reset_buf & ~mask
        reset_termination_state(self.term_state, mask)
        self.steer_state = torch.where(
            mask, torch.zeros_like(self.steer_state), self.steer_state
        )
        self.wheel_omega_cmd = torch.where(
            m, torch.zeros_like(self.wheel_omega_cmd), self.wheel_omega_cmd
        )

        if not preserve_buffers:
            self.episode_steps_buf = torch.where(
                mask, torch.zeros_like(self.episode_steps_buf), self.episode_steps_buf
            )
            self.lap_count_buf = torch.where(
                mask, torch.zeros_like(self.lap_count_buf), self.lap_count_buf
            )
            zero_act = torch.zeros_like(self.actions)
            self.actions = torch.where(m, zero_act, self.actions)
            self.last_actions = torch.where(m, zero_act, self.last_actions)

            if launch:
                max_speed = max(float(self.env_cfg.get("max_speed", 5.0)), 1e-6)
                clip_actions = float(self.env_cfg.get("clip_actions", 1.0))
                reset_throttle = torch.clamp(
                    speed / max_speed, min=-clip_actions, max=clip_actions
                )
                self.actions[:, 0] = torch.where(
                    mask, reset_throttle, self.actions[:, 0]
                )
                self.last_actions[:, 0] = torch.where(
                    mask, reset_throttle, self.last_actions[:, 0]
                )

            for value in self.reward_state["episode_sums"].values():
                value.masked_fill_(mask, 0.0)

    def _apply_launch_velocity(
        self, mask: torch.Tensor, quat: torch.Tensor, speed: torch.Tensor
    ) -> None:
        """Set wheel spin, neutral steering, and chassis launch velocity for the
        reset envs. Uses index-based dof setters (the only form Genesis 0.4.3
        accepts for explicit dof values), so it incurs a single nonzero sync; only
        called when a non-zero reset-speed range is configured."""
        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        n = int(env_ids.numel())
        if n == 0:
            return

        car = self.car  # type: Any
        wheel_radius = max(float(self.env_cfg.get("wheel_radius", 0.05)), 1e-6)
        sp = speed[env_ids]
        wheel_omega = (
            (sp / wheel_radius)
            .unsqueeze(1)
            .expand(n, len(self.wheel_dofs))
            .contiguous()
        )
        steer_zeros = torch.zeros(
            (n, len(self.steer_dofs)), dtype=gs.tc_float, device=self.device
        )
        yaw = gu.quat_to_xyz(quat[env_ids], rpy=True, degrees=False)[:, 2]
        root_vel = torch.stack(
            [sp * torch.cos(yaw), sp * torch.sin(yaw), torch.zeros_like(sp)], dim=-1
        )

        car.set_dofs_position(
            steer_zeros, self.steer_dofs, envs_idx=env_ids, zero_velocity=False
        )
        car.set_dofs_velocity(wheel_omega, self.wheel_dofs, envs_idx=env_ids)
        car.set_dofs_velocity(root_vel, self.root_dof_vel_idx, envs_idx=env_ids)

        if self.opponent is not None:
            self.opponent.set_dofs_position(
                steer_zeros, self.steer_dofs, envs_idx=env_ids, zero_velocity=False
            )
            self.opponent.set_dofs_velocity(
                wheel_omega, self.wheel_dofs, envs_idx=env_ids
            )
            self.opponent.set_dofs_velocity(
                root_vel, self.root_dof_vel_idx, envs_idx=env_ids
            )

    def _get_step_state(self) -> dict[str, Any]:
        if not self._step_state_valid:
            self._step_state = build_step_state(
                base_pos=self.base_pos,
                episode_steps_buf=self.episode_steps_buf,
                track_state=self.track_state,
                device=self.device,
                cache_id=self.track_cache_id,
            )

            self._step_state["wheel_state"] = dict(
                motion_link_vel=self.car.get_links_vel(
                    links_idx_local=self.slip_motion_link_idx, ref="link_com"
                ),
                frame_quat=self.car.get_links_quat(
                    links_idx_local=self.slip_frame_link_idx
                ),
                dof_vel=self.car.get_dofs_velocity(dofs_idx_local=self.wheel_dofs),
            )
            wheel_radius = float(self.env_cfg.get("wheel_radius", 0.05))
            slip_eps = float(self.reward_cfg.get("slip_eps", 0.1))
            self._step_state["tyre_slip"] = compute_tyre_slip(
                self._step_state["wheel_state"],
                wheel_radius=wheel_radius,
                slip_eps=slip_eps,
            )
            self._step_state_valid = True
        return self._step_state

    def _update_state_buffers(self):
        car = self.car  # type: Any
        quat = car.get_quat()
        self.base_pos = car.get_pos()
        self.base_quat = quat

        self.base_lin_vel = gu.inv_transform_by_quat(car.get_vel(), quat)
        self.base_ang_vel = gu.inv_transform_by_quat(car.get_ang(), quat)

        self.base_lin_acc = gu.inv_transform_by_quat(
            car.get_links_acc(links_idx_local=[self.base_link_idx_local])[:, 0, :],
            quat,
        )
        invalidate_step_caches(self.track_state)
        self._step_state_valid = False

    def _update_observation(self):
        step_state = self._get_step_state()
        self.obs_buf = self._build_observation(
            num_obs=self.num_obs,
            num_envs=self.num_envs,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            base_lin_acc=self.base_lin_acc,
            last_actions=self.last_actions,
            base_pos=self.base_pos,
            base_quat=self.base_quat,
            obs_cfg=self.obs_cfg,
            step_state=step_state,
            device=self.device,
        )

    def _compute_rewards(self):
        step_state = self._get_step_state()
        step_state["base_lin_vel"] = self.base_lin_vel
        step_state["actions"] = self.actions
        step_state["last_actions"] = self.last_actions
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

    def _normalize_reset_mask(self, envs_idx) -> torch.Tensor:
        """Coerce a None / index-list / index-tensor / bool-mask into a bool mask."""
        if envs_idx is None:
            return torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        if isinstance(envs_idx, (list, tuple, np.ndarray)):
            mask = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
            if len(envs_idx) > 0:
                mask[list(envs_idx)] = True
            return mask
        if envs_idx.dtype == torch.bool:
            return envs_idx
        mask = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        mask[envs_idx] = True
        return mask

    def reset(self, envs_idx=None):
        """Sync-free reset. Safe to call unconditionally every step with a (possibly
        all-False) done mask: masked envs are teleported/zeroed and the rest are
        untouched, with no host-device synchronization."""
        mask = self._normalize_reset_mask(envs_idx)

        self._reset_envs(mask)

        # Re-read state after the masked teleport (immediate FK, no scene.step) and
        # force fresh kinematics for the reset envs.
        self._update_state_buffers()
        self.base_lin_acc = torch.where(
            mask.unsqueeze(1), torch.zeros_like(self.base_lin_acc), self.base_lin_acc
        )

        step_state = self._get_step_state()
        sync_progress_state_for_resets(
            reward_state=self.reward_state,
            step_state=step_state,
            episode_steps_buf=self.episode_steps_buf,
            reset_mask=mask,
        )

        self._update_observation()
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def _apply_actions(
        self, exec_actions: torch.Tensor, env_ids: torch.Tensor | None = None
    ):
        """
        Apply force-based throttle/brake and lagged steering controls.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        throttle_cmd = exec_actions[:, 0]
        steer = exec_actions[:, 1]

        delta_max = float(
            self.env_cfg.get(
                "delta_max", self.env_cfg.get("max_steer", 0.44)
            )
        )
        delta_cmd = torch.clamp(steer * delta_max, min=-delta_max, max=delta_max)

        self.steer_state[env_ids] = self.steer_state[env_ids] + self.steer_lag_alpha * (
            delta_cmd[env_ids] - self.steer_state[env_ids]
        )
        delta_center = self.steer_state[env_ids]

        steer_targets = ackermann_left_right(
            delta_center=delta_center,
            L=self.env_cfg.get("wheelbase", 0.325),
            W=self.env_cfg.get("track_width", 0.20),
        )

        base_vel_body = self.base_lin_vel[env_ids]
        wheel_dof_vel = self.car.get_dofs_velocity(
            dofs_idx_local=self.wheel_dofs, envs_idx=env_ids
        )
        wheel_torques = compute_wheel_torques(
            throttle_cmd=throttle_cmd[env_ids],
            base_lin_vel_body=base_vel_body,
            wheel_dof_vel=wheel_dof_vel,
            env_cfg=self.env_cfg,
            vehicle_mass=self.vehicle_mass,
        )

        car = self.car  # type: Any
        car.control_dofs_force(
            wheel_torques.to(device=gs.device),
            self.wheel_dofs,
            envs_idx=env_ids,
        )
        car.control_dofs_position(
            steer_targets.to(device=gs.device), self.steer_dofs, envs_idx=env_ids
        )

    def _dissipative_enabled(self) -> bool:
        return bool(self.env_cfg.get("enable_aero_drag", False)) or (
            float(self.env_cfg.get("c_roll", 0.0)) > 0.0
        )

    def _compute_dissipative_force(self) -> torch.Tensor | None:
        """
        Sample chassis velocity once per control step and return the drag force.

        Drag varies slowly within one 0.1s control step, so we sample velocity
        once (instead of once per substep) and re-apply the same force each
        substep. This avoids ~9 extra GPU reads per control step.
        """
        if not self._dissipative_enabled():
            return None
        lin_vel_world = self.car.get_vel()
        return compute_dissipative_force_world(lin_vel_world, self.env_cfg)

    def _apply_dissipative_force(self, force: torch.Tensor) -> None:
        # External forces are cleared at the end of every scene.step(), so the
        # cached force must be re-applied immediately before each substep.
        self.car._solver.apply_links_external_force(
            force,
            links_idx=(self.base_link_idx,),
            ref="link_com",
            local=False,
        )

    def step(self, actions, n_steps=1):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )

        self._apply_actions(exec_actions)
        drag_force = self._compute_dissipative_force()
        for _ in range(n_steps):
            if drag_force is not None:
                self._apply_dissipative_force(drag_force)
            self.scene.step()
            if self.cam1 is not None:
                position = self.car.get_pos(envs_idx=[0])[0]

                self.cam1.set_pose(
                    lookat=position.cpu() + np.array([0.0, 0.0, 0.5]),
                    pos=position.cpu() + np.array([3.0, 0.0, 7.0]),
                )
                rgb, *_ = self.cam1.render()
                rr.log("image", rr.Image(rgb))
        self.episode_steps_buf += 1
        self._update_state_buffers()

        self._compute_rewards()
        self._compute_terminations()

        done = self.reset_buf.clone()
        # Unconditional, mask-based reset: no per-step host-device sync. reset()
        # recomputes the observation for the full batch (reset envs reflect their
        # new spawn pose), so no separate _update_observation is needed.
        self.reset(done)

        self.last_actions.copy_(self.actions)
        return self.obs_buf, self.reward_buf, done, self.extras

    def close(self):
        self.scene.destroy()
