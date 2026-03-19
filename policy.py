import io
from typing import TYPE_CHECKING

import genesis as gs
import torch
from torch import nn

if TYPE_CHECKING:
    from config import Config


class Model(nn.Module):

    def __init__(self, obs_dim: int = 6, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# https://github.com/sony/CaR/blob/main/car/algorithms/qrsac.py


class PolicyBase:
    def maybe_refresh(self, force: bool = False) -> bool:
        raise NotImplementedError

    def get_actions(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformRandomPolicy(PolicyBase):
    def __init__(self, action_dim: int = 2, action_clip: float = 1.0):
        self.action_dim = action_dim
        self.action_clip = float(action_clip)

    def maybe_refresh(self, force: bool = False) -> bool:
        return False

    def get_actions(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("Observation batch must be rank-2 [batch, obs_dim]")
        return torch.tensor(
            [0.5, 0.0], dtype=observation.dtype, device=observation.device
        ) + 0.5 * (
            2.0
            * torch.rand(
                (observation.shape[0], self.action_dim),
                dtype=observation.dtype,
                device=observation.device,
            )
            - 1.0
        )


class Policy(PolicyBase):
    def __init__(
        self,
        cfg: "Config",
        action_dim: int = 2,
        action_clip: float = 1.0,
        policy_blob_key: str = "current_policy",
        policy_version_key: str = "current_policy_version",
        fallback_mode: str = "random",
    ):
        self.cfg = cfg
        self.log = self.cfg.logger.getChild("Policy")
        self.action_dim = action_dim
        self.action_clip = float(action_clip)
        self.policy_blob_key = policy_blob_key
        self.policy_version_key = policy_version_key
        self.fallback_mode = fallback_mode

        self._model: nn.Module | None = None
        self._policy_version: str | None = None

    @property
    def policy(self):
        return {
            "loaded": self._model is not None,
            "version": self._policy_version,
        }

    def maybe_refresh(self, force: bool = False) -> bool:
        version = self.cfg.redis.get(self.policy_version_key)
        if not force and version == self._policy_version:
            return False

        blob = self.cfg.redis_b.get(self.policy_blob_key)
        if blob is None:
            if force:
                self.log.info(
                    "No policy blob at key '%s'; using fallback actions.",
                    self.policy_blob_key,
                )
            return False

        try:
            if isinstance(blob, memoryview):
                blob_bytes = blob.tobytes()
            elif isinstance(blob, bytearray):
                blob_bytes = bytes(blob)
            elif isinstance(blob, bytes):
                blob_bytes = blob
            else:
                raise TypeError(f"Unsupported policy payload type: {type(blob)}")

            payload = torch.load(io.BytesIO(blob_bytes), map_location="cpu")
            state_dict = (
                payload.get("state_dict", payload)
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(state_dict, dict):
                raise TypeError("Expected a state_dict dictionary payload")

            if self._model is None:
                self._model = Model(action_dim=self.action_dim)

            self._model.load_state_dict(state_dict, strict=False)
            self._model.eval()
            self._policy_version = str(version) if version is not None else "unknown"
            self.log.info("Loaded policy version %s", self._policy_version)
            return True
        except Exception as exc:
            self.log.exception("Failed to refresh policy from Redis: %s", exc)
            return False

    def get_actions(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("Observation batch must be rank-2 [batch, obs_dim]")

        if self._model is None:
            raise RuntimeError("Policy model is not loaded")
        else:
            with torch.no_grad():
                model_obs = observation.to(dtype=torch.float32, device=gs.device)
                model_action = self._model(model_obs)
                action = model_action.to(
                    device=observation.device, dtype=observation.dtype
                )

        return torch.clamp(action, -self.action_clip, self.action_clip)
