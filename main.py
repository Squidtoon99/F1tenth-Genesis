import os
import genesis as gs
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# flake8: noqa: F401
import numpy as np
import torch
from config import Config
from policy import Policy, UniformRandomPolicy
from replay import ReplayServer
from task import get_task
from f1tenth_env import F1tenthEnv

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def rollout_loop(cfg: "Config"):
    env_cfg = {
        "num_actions": 2,
        "episode_length": 25.0,
        "clip_actions": 1.0,
        "simulate_action_latency": True,
        "term_oob_margin_m": 0.15,
        "term_oob_max_consecutive": 15,
        "term_speed_threshold": 0.2,
        "term_not_moving_time_s": 2.0,
        "term_not_moving_min_ds": 1e-3,
        "term_heading_error_rad": 3.0,
        "target_laps": 0,
        "car_spawn_pos": (0.0, 0.0, 0.01),
        "car_spawn_rot": (0.0, 0.0, 0.0),
        "joint_names": [
            "left_rear_wheel_joint",
            "right_rear_wheel_joint",
        ],
        "default_joint_angles": {
            "left_rear_wheel_joint": 0.0,
            "right_rear_wheel_joint": 0.0,
        },
        "max_speed": 7.0,  # m/s
        "max_steer": 0.4189,  # radianss
        "wheelbase": 0.325,
        "track_width": 0.20,
        "wheel_radius": 0.05,
        "track": "Montreal",
    }
    obs_cfg = {
        "num_obs": 371,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        },
        "contact_margin_m": 0.08,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        "future_track_width": 2.0,
    }
    reward_cfg = {
        "progress_k_fwd": 5.0,
        "progress_k_back": 5.0,
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.5,
        "oob_k": 10.0,
        "reward_scales": {
            "progress": 1.4,
            "oob_penalty": 1.0,
        },
    }

    max_steps = int(os.getenv("COLLECTOR_STEPS", "10000"))
    n_steps = int(os.getenv("COLLECTOR_CHUNK_STEPS", "5"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("ROLLOUT_DIR", "outputs/random_rollouts")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_server = ReplayServer()
    while True:

        # Handling task selection
        task = get_task(cfg)

        if task.random_policy:
            agent_policy = UniformRandomPolicy(
                action_dim=env_cfg["num_actions"], action_clip=env_cfg["clip_actions"]
            )
        else:
            agent_policy = Policy(
                cfg=cfg,
                action_dim=env_cfg["num_actions"],
                action_clip=env_cfg["clip_actions"],
            )
            agent_policy.maybe_refresh(force=True)

        num_envs = int(task.launch_strategy.data.get("num_cars", 20))
        env = F1tenthEnv(
            num_envs=num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            show_viewer=__name__
            == "__main__",  # Only show viewer if running main.py directly
        )

        obs, _ = env.reset()
        trajs = [[] for _ in range(num_envs)]
        for step in range(max_steps):
            actions = agent_policy.get_actions(obs)
            next_obs, reward, done, _extras = env.step(actions)
            # if step % 10 == 0:
            #     logging.warning(
            #         f"Step {step}: reward={reward.mean().item():.3f}, done={done.float().mean().item():.3f} extras={_extras}"
            #     )
            for agent_id in range(num_envs):
                agent_done = bool(done[agent_id].item())
                trajs[agent_id].append(
                    {
                        "obs": obs[agent_id].detach().cpu(),
                        "action": actions[agent_id].detach().cpu(),
                        "reward": reward[agent_id].item(),
                        "done": agent_done,
                    }
                )
                # TODO: Check the mistakes and introduce mistake learning tasks here

            if step + 1 >= n_steps and task.table_name:
                # Construct N-Step subtrajectory if data collection task
                for traj in trajs:
                    replay_server.write(task.table_name, traj[-n_steps:])
            obs = next_obs

            if task.time_out_fn(step, obs):
                break
        logging.info(f"Task finished after {step+1} steps. Resetting environment...")
        env.close()

        if __name__ == "__main__":
            break  # Only run one episode if running main.py directly


def main():
    logging.info("Initializing Genesis and setting up the scene...")
    gs.init()

    session_id = os.getenv("SESSION_ID", "0")
    cfg = Config(session_id=session_id, redis_uri=os.getenv("REDIS_URI"))
    rollout_loop(cfg)


if __name__ == "__main__":
    main()
