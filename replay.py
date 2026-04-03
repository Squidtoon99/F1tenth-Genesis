import json
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from config import Config


class ReplayServer:

    def __init__(self, cfg: "Config"):
        # Initialize connection to Redis or other storage backend here
        self.redis_client = cfg.redis
        self.cfg = cfg
        self.gamma = cfg.model.get("rew_gamma", 0.99)
        self.n_step = cfg.model.get("n_step", 5)

    def compute_n_step_returns(self, trajectory):
        R = 0.0
        done = False
        for i in range(self.n_step):
            r = trajectory[i]["reward"]
            d = trajectory[i]["done"]
            R += (self.gamma**i) * r
            if d:
                done = True
                break

        obs = trajectory[0]["obs"]
        action = trajectory[0]["action"]
        next_obs = trajectory[i]["next_obs"] # type: ignore
        return obs, action, R, next_obs, done

    @staticmethod
    def _encode_for_redis(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        return json.dumps(value)

    def write_trajectories(self, table_name: str, trajectories: list):
        pipe = self.redis_client.pipeline(transaction=False)
        for traj in trajectories:
            obs, action, rew, next_obs, done = self.compute_n_step_returns(traj)
            pipe.xadd(  # type: ignore
                self.cfg.rkey("replay_buffer"),
                {
                    "table_name": table_name,
                    "obs": self._encode_for_redis(obs),
                    "action": self._encode_for_redis(action),
                    "reward": str(float(rew)),
                    "next_obs": self._encode_for_redis(next_obs),
                    "done": str(int(done)),
                },
                maxlen=1_000_000,
            )
        pipe.execute()  # type: ignore
