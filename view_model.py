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
from eval_worker import process_eval_data
from policy import Policy, UniformRandomPolicy
from replay import ReplayServer
from task import LaunchStrategy, Task, get_task
from f1tenth_env import F1tenthEnv
import rerun as rr

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import wandb

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _slice_for_agent(value: Any, agent_id: int, num_envs: int) -> Any:
    """Return per-agent data when the first dimension aligns with number of envs."""
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == num_envs:
            return value[agent_id]
        return value

    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == num_envs:
            return value[agent_id]
        return value

    if isinstance(value, (list, tuple)) and len(value) == num_envs:
        return value[agent_id]

    return value


def _log_rerun_value(path: str, value: Any) -> None:
    """Log a python value to Rerun with reasonable defaults by type."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            rr.log(path, rr.Scalars(float(value.item())))
        else:
            rr.log(path, rr.Tensor(value))
        return

    if isinstance(value, bool):
        rr.log(path, rr.Scalars(1.0 if value else 0.0))
        return

    if isinstance(value, (int, float, np.number)):
        rr.log(path, rr.Scalars(float(value)))
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _log_rerun_value(f"{path}/{key}", item)
        return

    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.dtype != object:
            if arr.ndim == 0:
                rr.log(path, rr.Scalars(float(arr.item())))
            else:
                rr.log(path, rr.Tensor(arr))
            return

    rr.log(path, rr.TextLog(str(value)))


def _log_action_fields(agent_path: str, action: torch.Tensor) -> None:
    action_cpu = action.detach().cpu().to(dtype=torch.float32)
    _log_rerun_value(f"{agent_path}/action/raw", action_cpu)

    if action_cpu.numel() >= 1:
        rr.log(f"{agent_path}/action/throttle", rr.Scalars(float(action_cpu[0].item())))
    if action_cpu.numel() >= 2:
        rr.log(f"{agent_path}/action/steering", rr.Scalars(float(action_cpu[1].item())))


def _log_observation_fields(agent_path: str, obs: torch.Tensor) -> None:
    obs_cpu = obs.detach().cpu().to(dtype=torch.float32).reshape(-1)
    obs_dim = int(obs_cpu.numel())

    _log_rerun_value(f"{agent_path}/obs/raw", obs_cpu)
    if obs_dim < 10:
        return

    lin_vel = obs_cpu[0:2]
    ang_vel = obs_cpu[2:3]
    _last_action = obs_cpu[3:5]
    track_progress = obs_cpu[5:7]
    centerline_angle = obs_cpu[7]
    centerline_distance = obs_cpu[8]
    wall_contact_flag = obs_cpu[9]
    future_pts_flat = obs_cpu[10:]

    rr.log(
        f"{agent_path}/obs/linear_velocity",
        rr.Arrows2D(vectors=[(-lin_vel[:2]).numpy()]),
    )
    rr.log(
        f"{agent_path}/obs/yaw_rate",
        rr.Scalars(float(-ang_vel[0].item())),
    )
    rr.log(
        f"{agent_path}/obs/track_progress_cos",
        rr.Scalars(float(track_progress[0].item())),
    )
    rr.log(
        f"{agent_path}/obs/track_progress_sin",
        rr.Scalars(float(track_progress[1].item())),
    )
    rr.log(
        f"{agent_path}/obs/centerline_angle",
        rr.Scalars(float(centerline_angle.item())),
    )
    rr.log(
        f"{agent_path}/obs/centerline_distance",
        rr.Scalars(float(centerline_distance.item())),
    )
    rr.log(
        f"{agent_path}/obs/wall_contact_flag",
        rr.Scalars(float(wall_contact_flag.item())),
    )

    if future_pts_flat.numel() == 0:
        return

    n_curves = 3
    dims = 2
    per_env = int(future_pts_flat.numel())
    if per_env % (n_curves * dims) != 0:
        logging.warning(
            f"Future points shape is not divisible by (n_curves * dims): {future_pts_flat.shape} {n_curves * dims}"
        )
        return

    n_points = per_env // (n_curves * dims)
    future_pts = future_pts_flat.view(n_curves, n_points, dims).numpy()
    center_ego = -future_pts[0]
    left_ego = -future_pts[1]
    right_ego = -future_pts[2]

    rr.log(f"{agent_path}/obs/future_center_ego", rr.Points2D(center_ego))
    rr.log(f"{agent_path}/obs/future_left_ego", rr.Points2D(left_ego))
    rr.log(f"{agent_path}/obs/future_right_ego", rr.Points2D(right_ego))


def _log_agent_step(
    agent_id: int,
    step: int,
    episode: int,
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    extra: Any,
    num_envs: int,
) -> None:
    rr.set_time("episode", sequence=episode)
    rr.set_time("sim_step", sequence=step)

    agent_path = f"agents/{agent_id}"
    _log_observation_fields(agent_path, obs[agent_id])
    _log_action_fields(agent_path, action[agent_id])
    _log_rerun_value(f"{agent_path}/rewards/total", reward[agent_id])

    for reward_name, reward_value in extra["rewards"]["terms"].items():
        _log_rerun_value(f"{agent_path}/rewards/{reward_name}", reward_value[agent_id])
    _log_rerun_value(f"{agent_path}/done", done[agent_id])

    # Log point to view agent
    _log_rerun_value(f"{agent_path}/position", rr.Points2D([0.0, 0.0]))
    if isinstance(extra, dict):
        for key, value in extra.items():
            _log_rerun_value(
                f"{agent_path}/extras/{key}",
                _slice_for_agent(value, agent_id=agent_id, num_envs=num_envs),
            )
    elif extra is not None:
        _log_rerun_value(
            f"{agent_path}/extras/value",
            _slice_for_agent(extra, agent_id=agent_id, num_envs=num_envs),
        )


def rollout_loop(cfg: "Config"):
    max_steps = int(os.getenv("COLLECTOR_STEPS", "500"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("ROLLOUT_DIR", "outputs/testing")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handling task selection
    task = Task(
        launch_strategy=LaunchStrategy.uniform_jittered(
            num_cars=2,
            mps_range=(0.0, 1.0),
        ),
        random_policy=False,
        table_name=None,
        time_out_fn=lambda step, obs: step >= max_steps - 1,
        is_eval=True,
    )

    if task.random_policy:
        agent_policy = UniformRandomPolicy(
            action_dim=cfg.env["num_actions"],
            action_clip=cfg.env["clip_actions"],
        )
    else:
        agent_policy = Policy(
            cfg=cfg,
            action_dim=cfg.env["num_actions"],
            action_clip=cfg.env["clip_actions"],
        )
        agent_policy.maybe_refresh(force=True)

    num_envs = int(task.launch_strategy.data.get("num_cars", 20))
    env = F1tenthEnv(
        env_cfg={
            "launch_strategy": task.launch_strategy.type,
            "launch_strategy_data": task.launch_strategy.data,
            **cfg.env,
        },
        num_envs=num_envs,
        obs_cfg=cfg.obs,
        reward_cfg=cfg.reward,
        show_viewer=False,
        enable_recording=True,
    )

    obs, _ = env.reset()
    extras = []
    trajs = [[] for _ in range(num_envs)]  # list of trajectories for each agent
    step = 0
    for step in range(max_steps):
        actions = agent_policy.get_actions(obs, exploit=task.is_eval)

        next_obs, reward, done, extra = env.step(
            actions, n_steps=cfg.env.get("control_interval", 10)
        )
        for agent_id in range(num_envs):
            agent_done = bool(done[agent_id].item())
            _log_agent_step(
                agent_id=agent_id,
                step=step,
                episode=1,
                obs=obs,
                action=actions,
                reward=reward,
                done=done,
                extra=extra,
                num_envs=num_envs,
            )
            
            trajs[agent_id].append(
                {
                    "obs": obs[agent_id].detach().cpu(),
                    "action": actions[agent_id].detach().cpu(),
                    "reward": reward[agent_id].item(),
                    "next_obs": next_obs[agent_id].detach().cpu(),
                    "done": agent_done,
                }
            )
        obs = next_obs

    logging.info(f"Task finished after {step+1} steps. Resetting environment...")
    if env.cam1 is not None:
        env.cam1.stop_recording(
            save_to_filename=output_dir / f"episode_{1}.mp4", fps=60
        )

    if task.is_eval:
        process_eval_data(cfg, task, extras, trajs[0])
    env.close()

def main():
    logging.info("Initializing Genesis and setting up the scene...")
    # rr.init("f1tenth-genesis", spawn=False)
    # rr.init("f1tenth-genesis")
    # server_uri = rr.serve_grpc()
    # rr.serve_web_viewer(connect_to=server_uri)
    gs.init()

    session_id = os.getenv("SESSION_ID", "0")
    cfg = Config(session_id=session_id, bootstrap_db=False)
    rollout_loop(cfg)


if __name__ == "__main__":
    main()
