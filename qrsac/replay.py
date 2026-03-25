import json
import time
import torch

from threading import Thread
from tqdm import tqdm
from config import Config
from .spinningup.sac import ReplayBuffer


class TabledReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, size, tables, rew_gamma=0.99):
        self.tables = {
            table_name: ReplayBuffer(obs_dim, act_dim, size) for table_name in tables
        }
        self.rew_gamma = rew_gamma

    def write(self, table_name: str, obs, action, rew, next_obs, done):
        self.store(table_name, obs, action, rew, next_obs, done)

    def store(self, table_name: str, obs, act, rew, next_obs, done):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in replay buffer.")
        return self.tables[table_name].store(obs, act, rew, next_obs, done)

    def sample_batch(self, table_name: str, batch_size=32):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found in replay buffer.")
        return self.tables[table_name].sample_batch(batch_size)

    def uniform_sample_batch(self, batch_size=32):
        total = []

        # filter to only sample tables that have enough samples
        valid_tables = [
            table
            for table in self.tables.values()
            if table.size >= batch_size // len(self.tables)
        ]

        if not valid_tables:
            raise ValueError("Not enough samples in any table to sample a batch.")
        if len(valid_tables) == 1:
            return valid_tables[0].sample_batch(batch_size)

        for table in valid_tables:
            total.append(table.sample_batch(batch_size // len(valid_tables)))
        return {
            key: torch.cat([batch[key] for batch in total], dim=0)
            for key in total[0].keys()
        }


class ReplayServer(TabledReplayBuffer):
    """
    Redis based ReplayServer that reads a stream of <s_t, a_t, r_t, s_t+1> pairs.

    Streams in data from the distributed training workers and stores it into the in-memory ReplayBuffers
    """

    def __init__(
        self, cfg: "Config", tables, obs_dim, act_dim, size=1000000, rew_gamma=0.99
    ):
        self.cfg = cfg
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=size,
            tables=tables,
            rew_gamma=rew_gamma,
        )

        self.redis_client = self.cfg.redis

        self.last_id = "0"
        self._running = False

    def start(self):
        self._running = True
        self.thread = Thread(target=self._subscribe, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        self.thread.join()

    @staticmethod
    def _decode_message_data(message_data):
        parsed = {}
        for key, value in message_data.items():
            decoded_key = key.decode("utf-8") if isinstance(key, bytes) else key
            decoded_value = value.decode("utf-8") if isinstance(value, bytes) else value
            parsed[decoded_key] = decoded_value

        return {
            "table_name": parsed["table_name"],
            "obs": torch.tensor(json.loads(parsed["obs"]), dtype=torch.float32),
            "action": torch.tensor(json.loads(parsed["action"]), dtype=torch.float32),
            "rew": float(parsed.get("rew", parsed["reward"])),
            "next_obs": torch.tensor(
                json.loads(parsed["next_obs"]), dtype=torch.float32
            ),
            "done": bool(int(parsed["done"])),
        }

    def _subscribe(self):
        while self._running:
            message: list | None = self.redis_client.xread(
                {self.cfg.rkey("replay_buffer"): self.last_id}, count=None, block=1000
            )  # type: ignore
            # message format: [(stream_name, [(message_id, message_data)])]
            if not message:
                continue

            message = message[0]
            for cmd in message[1]:  # type: ignore[index]
                self.last_id = (
                    cmd[0].decode("utf-8") if isinstance(cmd[0], bytes) else cmd[0]
                )
                parsed_cmd = self._decode_message_data(cmd[1])
                self.write(**parsed_cmd)

    def block_and_wait(self, minimum_samples=40000):
        total_samples = 0
        pbar = tqdm(total=minimum_samples, desc="Waiting for replay buffer to fill")
        while total_samples < minimum_samples:
            total_samples = sum(table.size for table in self.tables.values())

            pbar.n = total_samples
            pbar.refresh()
            time.sleep(1)
