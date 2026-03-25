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
import rerun as rr

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import wandb

rr.init("f1tenth-genesis", spawn=True)
# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def rollout_loop(cfg: "Config"):
    max_steps = int(os.getenv("COLLECTOR_STEPS", "5000"))
    n_steps = cfg.model.get("n_step", 5)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("ROLLOUT_DIR", "outputs/random_rollouts")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_server = ReplayServer(cfg)

    run = wandb.init(
        project="f1tenth-genesis",
        name=f"rollout_{cfg.session_id}",
        config=cfg.dict(),
    )

    while True:

        # Handling task selection
        task = get_task(cfg)

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
            show_viewer=os.getenv("SHOW_VIEWER", "1" if __name__ == "__main__" else "0")
            == "1",
        )

        obs, _ = env.reset()
        trajs = [[] for _ in range(num_envs)]  # list of trajectories for each agent

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
                        "next_obs": next_obs[agent_id].detach().cpu(),
                        "done": agent_done,
                    }
                )

            obs = next_obs

            if task.time_out_fn(step, obs):
                break
        logging.info(f"Task finished after {step+1} steps. Resetting environment...")
        env.close()

def main():
    logging.info("Initializing Genesis and setting up the scene...")
    gs.init()

    session_id = os.getenv("SESSION_ID", "0")
    cfg = Config(session_id=session_id, redis_uri=os.getenv("REDIS_URI"))
    rollout_loop(cfg)


if __name__ == "__main__":
    main()
