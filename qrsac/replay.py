import torch

from config import Config
from spinningup.sac import ReplayBuffer


class TabledReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, size, tables, rew_gamma=0.99):
        self.tables = {
            table_name: ReplayBuffer(obs_dim, act_dim, size) for table_name in tables
        }
        self.rew_gamma = rew_gamma

    def write(self, table_name: str, traj):
        obs = traj[0]["obs"]
        action = traj[0]["action"]

        # reward   = r0 + gamma*r1 + gamma^2*r2 + ... + gamma^6*r6
        rew = sum([step["reward"] * (self.rew_gamma**i) for i, step in enumerate(traj)])
        next_obs = traj[-1]["obs"]
        done = traj[-1]["done"]
        self.store(table_name, obs, action, rew, next_obs, done)

    def store(self, table_name: str, obs, act, rew, next_obs, done):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in replay buffer.")
        return self.tables[table_name].store(obs, act, rew, next_obs, done)

    def sample_batch(self, table_name: str, batch_size=32):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in replay buffer.")
        return self.tables[table_name].sample_batch(batch_size)

    def even_sample_batch(self, table_name: str, batch_size=32):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in replay buffer.")

        total = []

        for table in self.tables.values():
            total.append(table.sample_batch(batch_size // len(self.tables)))
        return {
            key: torch.cat([batch[key] for batch in total], dim=0)
            for key in total[0].keys()
        }


class ReplayServer:
    """
    Redis based ReplayServer that reads a stream of <s_t, a_t, r_t, s_t+1> pairs.

    Streams in data from the distributed training workers and stores it into the in-memory ReplayBuffers
    """

    def __init__(
        self, cfg: "Config", tables, obs_dim, act_dim, size=1000000, rew_gamma=0.99
    ):
        self.cfg = cfg
        self.replay_buffer = TabledReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=size,
            tables=tables,
            rew_gamma=rew_gamma,
        )

        self.redis_client = self.cfg.redis
