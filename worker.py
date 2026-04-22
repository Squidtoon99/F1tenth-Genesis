import os
import time
import genesis as gs
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# flake8: noqa: F401
import numpy as np
import torch
import tensorflow as tf

# Keep gpu for pytorch
tf.config.set_visible_devices([], "GPU")
from config import Config
from policy import Policy, UniformRandomPolicy
from reverb import Client
from task import TaskServer, get_task
from f1tenth_env import F1tenthEnv
from eval_worker import process_eval_data
from collections import deque

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import wandb

# logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

max_steps = 3000


def rollout_loop(cfg: "Config"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("ROLLOUT_DIR", "outputs/random_rollouts")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    task_server = TaskServer(cfg)
    logging.info("Waiting for trainer to start...")
    while task_server.wandb_id is None:
        time.sleep(1)

    wandb.init(
        project="f1tenth-genesis",
        config=cfg.dict(),
        id=task_server.wandb_id,
    )

    replay_client = Client(
        server_address=cfg.redis.get("replay_server_address") or "localhost:8000",
    )

    # Waiting for trainer to start
    logging.info("Waiting for trainer to start...")
    while task_server.wandb_id is None:
        time.sleep(1)

    wandb.init(
        project="f1tenth-genesis",
        config=cfg.dict(),
        id=task_server.wandb_id,
    )

    while cfg.redis.get("done") != "1":
        # Handling task selection
        task = get_task(task_server)

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
            enable_recording=task.is_eval,
        )

        obs, _ = env.reset()
        trajs = [
            deque(maxlen=cfg.model["n_step"]) for _ in range(num_envs)
        ]  # list of trajectories for each agent
        extras = []
        step = 0
        with replay_client.trajectory_writer(
            num_keep_alive_refs=cfg.model["n_step"]
        ) as writer:
            for step in range(max_steps):
                actions = agent_policy.get_actions(obs, exploit=task.is_eval)
                next_obs, reward, done, extra = env.step(actions)
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
                    if task.is_eval:
                        extras.append(extra)

                if step + 1 >= cfg.model["n_step"] and task.table_name:
                    # Construct N-Step subtrajectory if data collection task
                    for agent_id in range(num_envs):
                        replay_client.insert(
                            priorities={task.table_name: 1.0},
                            data={
                                "obs": np.stack(
                                    [t["obs"].cpu().numpy() for t in trajs[agent_id]],
                                    axis=0,
                                ).astype(np.float32),
                                "action": np.stack(
                                    [t["action"] for t in trajs[agent_id]], axis=0
                                ).astype(np.float32),
                                "reward": np.asarray(
                                    [t["reward"] for t in trajs[agent_id]],
                                    dtype=np.float32,
                                ).reshape(cfg.model["n_step"], 1),
                                "done": np.asarray(
                                    [float(t["done"]) for t in trajs[agent_id]],
                                    dtype=np.float32,
                                ).reshape(cfg.model["n_step"], 1),
                            },
                        )
                        if trajs[agent_id][-1]["done"]:
                            trajs[agent_id].clear()  # Clear trajectory on episode end

                obs = next_obs

                if task.time_out_fn(step, obs):
                    break

        if env.cam1:
            env.cam1.stop_recording(
                save_to_filename=output_dir
                / f"policy_{agent_policy.policy['version']}.mp4",
                fps=60,
            )
        if task.is_eval:
            data = process_eval_data(cfg, task, extras, trajs[0])

            # log lap times into redis
            if "lap_times" in data:
                for _, lap_time in enumerate(data["lap_times"]):
                    task_server.redis.zadd(
                        "lap_times",
                        {f"policy_{agent_policy.policy['version']}": lap_time},
                    )

            task_server.eval_workers_available = (
                True  # ensure only one eval worker is running at a time
            )
        else:
            writer.flush()

        logging.info(
            f"Task finished after %d steps. Resetting environment...", step + 1
        )
        env.close()


def main():
    logging.info("Initializing Genesis and setting up the scene...")
    gs.init()

    cfg = Config()
    rollout_loop(cfg)


if __name__ == "__main__":
    main()
