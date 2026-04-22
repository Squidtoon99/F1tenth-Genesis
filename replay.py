from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import tensorflow as tf
import torch
from torch.utils import dlpack
import reverb

if TYPE_CHECKING:
    from config import Config


def _tf_to_torch(x, device: Optional[torch.device] = None) -> torch.Tensor:
    try:
        t = dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(x))
    except Exception:
        t = torch.from_numpy(x.numpy())
    if device is not None:
        t = t.to(device, non_blocking=True)
    return t


def sample_data(
    table_datasets: List[reverb.TrajectoryDataset],
    cfg: "Config",
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    discount = cfg.model["rew_gamma"]
    n_step = cfg.model["n_step"]

    total = []

    for dataset in table_datasets:
        sample = next(iter(dataset))
        data = sample.data

        batch = {
            "obs": _tf_to_torch(data["obs"], device).float(),
            "action": _tf_to_torch(data["action"], device).float(),
            "reward": _tf_to_torch(data["reward"], device).float(),
            "done": _tf_to_torch(data["done"], device).float(),
        }
        total.append(batch)

    sub_traj_batches = {
        key: torch.cat([batch[key] for batch in total], dim=0)
        for key in total[0].keys()
    }

    reward = sub_traj_batches["reward"]
    done = sub_traj_batches["done"]

    discounting = torch.pow(
        torch.tensor(discount, dtype=reward.dtype, device=reward.device),
        torch.arange(n_step, dtype=reward.dtype, device=reward.device),
    ).view(1, n_step, 1)

    discounted_reward = torch.sum(reward * discounting, dim=1)  # [B, 1]
    return {
        "obs": sub_traj_batches["obs"][:, 0, :],
        "action": sub_traj_batches["action"][:, 0, :],
        "reward": discounted_reward,
        "next_obs": sub_traj_batches["obs"][:, -1, :],
        "done": done[:, -1, :],
    }
