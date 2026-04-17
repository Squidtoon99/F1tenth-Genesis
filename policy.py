from typing import TYPE_CHECKING

import genesis as gs
import torch
from torch import nn
from qrsac import SquashedGaussianMLPActor
from param import PolicyFetcher

if TYPE_CHECKING:
    from config import Config


class PolicyBase:
    def maybe_refresh(self, force: bool = False) -> bool:
        raise NotImplementedError

    def get_actions(
        self, observation: torch.Tensor, exploit: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError


class UniformRandomPolicy(PolicyBase):
    def __init__(self, action_dim: int = 2, action_clip: float = 1.0):
        self.action_dim = action_dim
        self.action_clip = float(action_clip)

    def maybe_refresh(self, force: bool = False) -> bool:
        return False

    def get_actions(
        self, observation: torch.Tensor, exploit: bool = False
    ) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("Observation batch must be rank-2 [batch, obs_dim]")
        return torch.tensor(
            [0.0, 0.0], dtype=observation.dtype, device=observation.device
        ) + 1.0 * (
            2.0
            * torch.rand(
                (observation.shape[0], self.action_dim),
                dtype=observation.dtype,
                device=observation.device,
            )
            - 1.0
        )

    @property
    def policy(self):
        return {
            "loaded": True,
            "version": "uniform_random",
        }


class Policy(PolicyBase):

    def __init__(
        self,
        cfg: "Config",
        action_dim: int = 2,
        action_clip: float = 1.0,
        fallback_mode: str = "random",
    ):
        self.cfg = cfg
        self.log = self.cfg.logger.getChild("Policy")
        self.action_dim = action_dim
        self.action_clip = float(action_clip)
        self.fallback_mode = fallback_mode

        self._model: SquashedGaussianMLPActor = SquashedGaussianMLPActor(
            obs_dim=self.cfg.obs["num_obs"],
            act_dim=self.action_dim,
            hidden_sizes=self.cfg.model["hidden_layers"],
            activation=nn.ReLU,
            act_limit=1.0,
        )
        self._model.eval()

        self._policy_version: int | None = None
        self._policy_fetcher = PolicyFetcher(
            param_server=self.cfg.s3_parameter_server,
            actor=self._model,
            device=gs.device,
        )

    @property
    def policy(self):
        return {
            "loaded": self._policy_version is not None,
            "version": self._policy_version,
        }

    def maybe_refresh(self, force: bool = False, version: int | None = None) -> bool:
        if self._policy_fetcher is None:
            if force:
                self.log.info("No S3 parameter server configured; using local weights.")
            return False

        try:
            updated = self._policy_fetcher.maybe_refresh(version=version)
            if updated is None:
                return False
            self._policy_version = updated
            self.log.info("Loaded policy version %s", self._policy_version)
            return True
        except Exception as exc:
            self.log.exception("Failed to refresh policy from S3: %s", exc)
            return False

    def get_actions(
        self, observation: torch.Tensor, exploit: bool = False
    ) -> torch.Tensor:
        if observation.ndim != 2:
            raise ValueError("Observation batch must be rank-2 [batch, obs_dim]")

        with torch.no_grad():
            model_obs = observation.to(dtype=torch.float32, device=gs.device)
            model_action, _ = self._model(
                model_obs,
                deterministic=exploit,
                with_logprob=False,
            )
            action = model_action.to(device=observation.device, dtype=observation.dtype)

        return torch.clamp(action, -self.action_clip, self.action_clip)
