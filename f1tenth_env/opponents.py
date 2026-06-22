"""Generic, pluggable opponent framework for 1v1 racing.

There is always exactly one opponent (hard 1v1; no multi-agent generality). The
environment talks only to an :class:`OpponentController`; whether the opponent is
a scripted controller or a neural policy is invisible to the env.

Concrete controllers:
- :class:`ScriptedCenterlineOpponent` - a centerline-following P-controller with
  closed-loop longitudinal speed control (shipped, the default 1v1 opponent).
- :class:`PolicyOpponent` - a frozen-policy opponent for self-play. The delayed
  snapshot/refresh training loop lives in ``standalone_trainer.py`` (``SelfPlayManager``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from genesis.utils.geom import quat_to_xyz


@dataclass
class OpponentContext:
    """Everything a controller needs to choose the opponent's actions this step.

    ``opp_obs`` is only populated when the controller declares
    ``requires_observation = True`` (e.g. the policy opponent), so the scripted
    opponent never pays for building a full egocentric observation.
    """

    step_state: dict[str, Any]
    opp_pos: torch.Tensor
    opp_vel: torch.Tensor
    opp_quat: torch.Tensor
    opp_last_actions: torch.Tensor
    env_cfg: dict[str, Any]
    device: torch.device
    ego_pos: torch.Tensor | None = None
    ego_vel: torch.Tensor | None = None
    ego_quat: torch.Tensor | None = None
    opp_obs: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class OpponentController:
    """Base class for the single 1v1 opponent.

    Subclasses return ``(num_envs, num_actions)`` actions in ``[-1, 1]`` from
    :meth:`act`. ``requires_observation`` tells the env whether to build the
    opponent's full egocentric observation before calling :meth:`act`.
    """

    requires_observation: bool = False

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, mask: torch.Tensor) -> None:
        """Optional hook for stateful controllers. No-op by default."""
        return None


class ScriptedCenterlineOpponent(OpponentController):
    """Centerline-following opponent: P-control on lateral error + heading error,
    with a fixed target speed (kept below the ego's pace so an overtake is
    feasible).
    """

    requires_observation = False

    def __init__(self, env_cfg: dict[str, Any]):
        self.kp_ey = float(env_cfg.get("opponent_kp_ey", 1.0))
        self.kh_heading = float(env_cfg.get("opponent_kh_heading", 1.0))
        self.delta_max = float(
            env_cfg.get("delta_max", env_cfg.get("max_steer", 0.44))
        )
        max_speed = max(float(env_cfg.get("max_speed", 15.0)), 1e-6)
        target_speed = float(env_cfg.get("opponent_target_speed", 3.0))
        self.throttle = float(min(max(target_speed / max_speed, 0.0), 1.0))

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        frenet = ctx.step_state["frenet"]
        boundary = ctx.step_state["boundary"]
        ey = boundary["ey"].reshape(-1)

        track_angle = torch.atan2(frenet["seg_dir"][:, 1], frenet["seg_dir"][:, 0])
        yaw = quat_to_xyz(ctx.opp_quat, rpy=True, degrees=False)[:, 2]
        heading_err = yaw - track_angle
        heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

        delta_max = max(self.delta_max, 1e-6)
        steer = -(self.kp_ey * ey + self.kh_heading * heading_err) / delta_max
        steer = torch.clamp(steer, -1.0, 1.0)

        throttle = torch.full_like(steer, self.throttle)
        return torch.stack([throttle, steer], dim=-1)


class PolicyOpponent(OpponentController):
    """Frozen-policy opponent for future self-play.

    Wraps a ``SquashedGaussianMLPActor`` and (optional) observation-normalization
    statistics; acts deterministically on the opponent's egocentric observation.
    The self-play *training loop* that periodically refreshes ``actor`` from the
    learner is deferred - the env only needs this controller to exist.
    """

    requires_observation = True

    def __init__(
        self,
        actor: torch.nn.Module,
        device: torch.device,
        obs_mean: torch.Tensor | None = None,
        obs_var: torch.Tensor | None = None,
        norm_eps: float = 1e-8,
        norm_clip: float = 10.0,
        act_clip: float = 1.0,
    ):
        self.actor = actor
        self.device = device
        self.norm_eps = float(norm_eps)
        self.norm_clip = float(norm_clip)
        self.act_clip = float(act_clip)
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_var = obs_var.to(device) if obs_var is not None else None
        self.actor.eval()

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_mean is None or self.obs_var is None:
            return obs
        normed = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.norm_eps)
        return torch.clamp(normed, -self.norm_clip, self.norm_clip)

    def load_snapshot(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        obs_mean: torch.Tensor,
        obs_var: torch.Tensor,
    ) -> None:
        """Hot-swap frozen actor weights and observation-normalization stats."""
        self.actor.load_state_dict(actor_state_dict)
        self.actor.to(device=self.device, dtype=torch.float32)
        self.actor.eval()
        self.obs_mean = obs_mean.to(self.device, dtype=torch.float32)
        self.obs_var = obs_var.to(self.device, dtype=torch.float32)

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        if ctx.opp_obs is None:
            raise ValueError(
                "PolicyOpponent.requires_observation is True but ctx.opp_obs is None; "
                "the env must build the opponent's egocentric observation."
            )
        with torch.no_grad():
            model_obs = self._normalize(ctx.opp_obs.to(self.device, dtype=torch.float32))
            action, _ = self.actor(model_obs, deterministic=True, with_logprob=False)
        return torch.clamp(action, -self.act_clip, self.act_clip).to(ctx.opp_obs.device)


def make_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> OpponentController | None:
    """Factory keyed on ``env_cfg['opponent_strategy']``.

    Returns ``None`` when no opponent is configured (solo / 1v0). Hard 1v1: this
    only ever returns a single controller.
    """
    strategy = env_cfg.get("opponent_strategy")
    if strategy is None:
        return None
    if strategy == "scripted":
        return ScriptedCenterlineOpponent(env_cfg)
    if strategy == "policy":
        return _make_policy_opponent(env_cfg, obs_cfg, device)
    raise ValueError(f"Unknown opponent_strategy: {strategy!r}")


def _make_policy_opponent(
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    device: torch.device,
) -> PolicyOpponent:
    # Imported lazily so the scripted path has no dependency on the RL stack.
    from qrsac import SquashedGaussianMLPActor

    obs_dim = int(obs_cfg["num_obs"])
    act_dim = int(env_cfg.get("num_actions", 2))
    hidden = list(env_cfg.get("opponent_hidden_layers", [512, 512, 512]))

    actor = SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden,
        activation=torch.nn.ReLU,
        act_limit=1.0,
    ).to(device=device, dtype=torch.float32)

    obs_mean = obs_var = None
    ckpt_path = env_cfg.get("opponent_ckpt")
    if ckpt_path:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(payload["actor"])
        if "obs_norm" in payload:
            obs_mean = payload["obs_norm"]["mean"].to(dtype=torch.float32)
            obs_var = payload["obs_norm"]["var"].to(dtype=torch.float32)

    actor.to(device=device, dtype=torch.float32).eval()

    return PolicyOpponent(
        actor=actor,
        device=device,
        obs_mean=obs_mean,
        obs_var=obs_var,
        norm_eps=float(obs_cfg.get("norm_eps", 1e-8)),
        norm_clip=float(obs_cfg.get("norm_clip", 10.0)),
    )
