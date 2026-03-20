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
            num_envs=num_envs,
            env_cfg=cfg.env,
            obs_cfg=cfg.obs,
            reward_cfg=cfg.reward,
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
