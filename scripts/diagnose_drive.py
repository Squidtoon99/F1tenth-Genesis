#!/usr/bin/env python3
"""Temporary drive diagnostic — pure wheel torque only."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import genesis as gs
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env.car import compute_tyre_slip

import pyglet
from genesis.vis.rasterizer import Rasterizer

pyglet.options["headless"] = True
Rasterizer.build = lambda self: (
    setattr(self, "visualizer", self._context.visualizer) if self._context else None
)

gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)

cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg["env"].update(
    {
        "simulate_action_latency": False,
        "launch_strategy": "fixed",
        "car_spawn_pos": (0.0, 0.0, 0.05),
        "f_drive_max": 23.0,
        "dragcoeff": 0.075,
        "tire_friction": 0.65,
        "enable_aero_drag": False,
        "c_roll": 0.0,
    }
)

env = F1tenthEnv(
    num_envs=1,
    env_cfg=cfg["env"],
    obs_cfg=cfg["obs"],
    reward_cfg=cfg["reward"],
    show_viewer=False,
)

print("ground friction set:", cfg["env"]["tire_friction"])
action = torch.tensor([[1.0, 0.0]], device=gs.device)
for i in range(30):
    env.step(action, n_steps=10)
    speed = float(torch.linalg.norm(env.base_lin_vel[0, :2]))
    wv = env.car.get_dofs_velocity(dofs_idx_local=env.wheel_dofs)[0].tolist()
    cf = env.car.get_dofs_control_force(dofs_idx_local=env.wheel_dofs)[0].tolist()
    step_state = env._get_step_state()
    slip = compute_tyre_slip(step_state["wheel_state"], 0.05)[0].tolist()
    print(
        f"step {i+1:2d}: v={speed:.3f} body_x={env.base_lin_vel[0,0]:.3f} "
        f"omega={[round(x,2) for x in wv]} tau={[round(x,3) for x in cf]} "
        f"slip_r={[round(x,2) for x in slip[:4]]}"
    )

env.close()
