#!/usr/bin/env python3
"""Single-process QRSAC trainer: F1tenthEnv + in-memory n-step replay, no Reverb/Redis/S3."""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import genesis as gs
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor

LOGGER_NAME = "standalone_trainer"
RECENT_EPISODES_MAX = 50


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record so lines appear promptly."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_trainer_logging(level: int = logging.INFO) -> logging.Logger:
    """Dedicated logger isolated from Genesis root-logger / FPS timer output."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = FlushingStreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def make_policy_network(cfg: dict) -> SquashedGaussianMLPActor:
    obs_dim = cfg["obs"]["num_obs"]
    action_dim = cfg["env"]["num_actions"]
    return SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg["model"]["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )


def make_q_network(cfg: dict) -> QuantileCritic:
    obs_dim = cfg["obs"]["num_obs"]
    action_dim = cfg["env"]["num_actions"]
    return QuantileCritic(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg["model"]["hidden_layers"],
        num_quantiles=cfg["model"]["num_quantiles"],
    )


def make_target_q_network(cfg: dict) -> QuantileCritic:
    target_q = make_q_network(cfg)
    for param in target_q.parameters():
        param.requires_grad = False
    return target_q


class NStepReplayBuffer:
    """Per-env n-step deques feeding a preallocated tensor ring buffer on device."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        n_step: int,
        gamma: float,
        num_envs: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.num_envs = num_envs
        self.device = device
        self.size = 0
        self.ptr = 0

        self.obs = torch.zeros(capacity, obs_dim, device=device, dtype=torch.float32)
        self.action = torch.zeros(
            capacity, act_dim, device=device, dtype=torch.float32
        )
        self.reward = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_obs = torch.zeros(
            capacity, obs_dim, device=device, dtype=torch.float32
        )
        self.done = torch.zeros(capacity, device=device, dtype=torch.float32)

        self._gamma_powers = torch.tensor(
            [gamma**k for k in range(n_step)], device=device, dtype=torch.float32
        )

        # Vectorized per-env n-step windows kept on device as circular buffers.
        # All envs advance in lockstep, so a single write column index ``w_pos``
        # is shared. ``w_len`` counts valid entries per env (reset to 0 on done).
        self.w_obs = torch.zeros(num_envs, n_step, obs_dim, device=device, dtype=torch.float32)
        self.w_act = torch.zeros(num_envs, n_step, act_dim, device=device, dtype=torch.float32)
        self.w_rew = torch.zeros(num_envs, n_step, device=device, dtype=torch.float32)
        self.w_len = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.w_pos = 0
        self._arange_n = torch.arange(n_step, device=device)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Vectorized n-step accumulation. Writes the current transition into each
        env's circular window, emits completed n-step samples for all full windows
        in a single batched scatter, then clears windows for done envs. The only
        host sync is one ``nonzero`` per step (independent of ``num_envs``)."""
        col = self.w_pos
        self.w_obs[:, col] = obs.detach()
        self.w_act[:, col] = actions.detach()
        self.w_rew[:, col] = rewards.detach()
        self.w_len = torch.clamp(self.w_len + 1, max=self.n_step)
        self.w_pos = (col + 1) % self.n_step

        # After advancing, column ``w_pos`` is the oldest entry of a full window;
        # ``order`` lists columns oldest -> newest for the discounted sum.
        oldest = self.w_pos
        order = (oldest + self._arange_n) % self.n_step
        n_step_reward = (self.w_rew[:, order] * self._gamma_powers).sum(dim=1)
        obs0 = self.w_obs[:, oldest]
        act0 = self.w_act[:, oldest]
        done_f = dones.detach().to(torch.float32)

        emit_mask = self.w_len == self.n_step
        idx = torch.nonzero(emit_mask, as_tuple=False).squeeze(-1)
        n_emit = int(idx.numel())
        if n_emit > 0:
            positions = (
                self.ptr + torch.arange(n_emit, device=self.device)
            ) % self.capacity
            self.obs[positions] = obs0[idx]
            self.action[positions] = act0[idx]
            self.reward[positions] = n_step_reward[idx]
            self.next_obs[positions] = next_obs[idx].detach()
            self.done[positions] = done_f[idx]
            self.ptr = int((self.ptr + n_emit) % self.capacity)
            self.size = min(self.size + n_emit, self.capacity)

        # Clear windows for done envs (sync-free masked write).
        self.w_len = torch.where(
            dones.bool(), torch.zeros_like(self.w_len), self.w_len
        )

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"Buffer has {self.size} samples, need at least {batch_size} to sample."
            )
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
        }


class ObsNormalizer:
    """Running mean/variance observation normalizer (Welford parallel update).

    Estimates per-feature mean and variance from observations actually experienced
    during training, then normalizes obs at network-input time. The replay buffer
    keeps RAW observations; normalization is applied with the current statistics
    wherever an observation enters a network, so there is no stale-normalization
    drift across the buffer. Stats are kept on-device in float32.
    """

    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        eps: float = 1e-8,
        clip: float = 10.0,
    ):
        self.device = device
        self.eps = float(eps)
        self.clip = float(clip)
        self.mean = torch.zeros(obs_dim, device=device, dtype=torch.float32)
        self.var = torch.ones(obs_dim, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Chan et al. parallel variance update from a (batch, obs_dim) tensor."""
        x = x.to(torch.float32)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.var = m2 / tot_count
        self.count = tot_count

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(torch.float32) - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].to(self.device, dtype=torch.float32)
        self.var = state["var"].to(self.device, dtype=torch.float32)
        self.count = float(state["count"])


class RunningStats:
    """Accumulates scalar means / min / max / totals for named diagnostics.

    Values are kept as on-device tensors and only synced to Python floats at
    log time to avoid a host sync on every environment step.
    """

    def __init__(self):
        self._sum: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}
        self._min: dict[str, torch.Tensor] = {}
        self._max: dict[str, torch.Tensor] = {}

    def add_mean(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.mean()
        self._count[key] = self._count.get(key, 0) + 1
        vmin, vmax = v.min(), v.max()
        self._min[key] = (
            vmin if key not in self._min else torch.minimum(self._min[key], vmin)
        )
        self._max[key] = (
            vmax if key not in self._max else torch.maximum(self._max[key], vmax)
        )

    def add_total(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.sum()

    def mean(self, key: str) -> float:
        if self._count.get(key, 0) == 0:
            return float("nan")
        return float(self._sum[key]) / self._count[key]

    def total(self, key: str) -> float:
        return float(self._sum[key]) if key in self._sum else 0.0

    def vmin(self, key: str) -> float:
        return float(self._min[key]) if key in self._min else float("nan")

    def vmax(self, key: str) -> float:
        return float(self._max[key]) if key in self._max else float("nan")

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()
        self._min.clear()
        self._max.clear()


def accumulate_step_diagnostics(
    diag: RunningStats,
    reward: torch.Tensor,
    actions: torch.Tensor,
    obs: torch.Tensor,
    extras: dict,
) -> None:
    """Fold one env step's reward terms, metrics and terminations into diag."""
    diag.add_mean("reward/step", reward)

    for name, value in extras.get("rewards", {}).get("terms", {}).items():
        if isinstance(value, torch.Tensor):
            diag.add_mean(f"reward_term/{name}", value)

    metrics = extras.get("metrics", {})
    for name in ("speed_xy", "lateral_error", "oob_mask", "progress_ds"):
        value = metrics.get(name)
        if isinstance(value, torch.Tensor):
            diag.add_mean(f"metric/{name}", value)

    for name, value in extras.get("termination", {}).items():
        if isinstance(value, torch.Tensor):
            diag.add_total(f"term/{name}", value)

    if actions.ndim == 2 and actions.shape[1] >= 2:
        diag.add_mean("action/throttle", actions[:, 0])
        diag.add_mean("action/steer", actions[:, 1])
    diag.add_mean("obs/abs", obs.abs())


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    if args.n_step is not None:
        cfg["model"]["n_step"] = args.n_step
    return cfg


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_models(cfg: dict, device: torch.device) -> tuple[Models, QRSACTrainer]:
    # Networks/optimizers stay float32 even when Genesis runs in precision="64"
    # (which flips torch's default dtype to float64); env outputs are bridged to
    # float32 at the boundary. Pin dtype explicitly so module creation under a
    # float64 default still yields float32 weights.
    net_dtype = torch.float32
    models = Models(
        actor=make_policy_network(cfg).to(device=device, dtype=net_dtype),
        critic1=make_q_network(cfg).to(device=device, dtype=net_dtype),
        critic2=make_q_network(cfg).to(device=device, dtype=net_dtype),
        critic1_target=make_target_q_network(cfg).to(device=device, dtype=net_dtype),
        critic2_target=make_target_q_network(cfg).to(device=device, dtype=net_dtype),
    )
    models.critic1_target.load_state_dict(models.critic1.state_dict())
    models.critic2_target.load_state_dict(models.critic2.state_dict())

    trainer = QRSACTrainer(
        models,
        device=device,
        gamma=cfg["model"]["rew_gamma"],
        n_step=cfg["model"]["n_step"],
        alpha=0.1,
        smooth_factor=0.005,
    )
    return models, trainer


def save_checkpoint(
    models: Models,
    step: int,
    ckpt_dir: Path,
    normalizer: "ObsNormalizer | None" = None,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step}.pt"
    payload = {
        "step": step,
        "actor": models.actor.state_dict(),
        "critic1": models.critic1.state_dict(),
        "critic2": models.critic2.state_dict(),
    }
    if normalizer is not None:
        # Eval/deploy MUST apply these same obs stats (e.g. ros2_deploy) since the
        # policy was trained on normalized observations.
        payload["obs_norm"] = normalizer.state_dict()
    torch.save(payload, path)
    logging.getLogger(LOGGER_NAME).info("Saved checkpoint to %s", path)
    return path


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Standalone QRSAC trainer (1v0, single process)")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=cfg["model"]["batch_size"])
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help=f"N-step horizon (default: {cfg['model']['n_step']} from config)",
    )
    parser.add_argument("--track", type=str, default=cfg["env"]["track"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="64",
        choices=["32", "64"],
        help="Genesis float precision. 64 is the strongest NaN-stability lever "
        "(slower / 2x memory); use 32 for fast iteration or GPU throughput.",
    )
    parser.add_argument("--ckpt-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="offline",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = build_config(args)
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": args.num_envs},
        **cfg["env"],
    }
    clip_actions = cfg["env"]["clip_actions"]
    control_interval = cfg["env"]["control_interval"]
    n_step = model_cfg["n_step"]

    device = select_device(args.device)

    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        precision=args.precision,
        performance_mode=True,
    )
    log = setup_trainer_logging()
    log.info("Using device: %s", device)

    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        enable_recording=False,
    )

    models, trainer = build_models(cfg, device)
    buffer = NStepReplayBuffer(
        capacity=args.buffer_capacity,
        obs_dim=obs_cfg["num_obs"],
        act_dim=cfg["env"]["num_actions"],
        n_step=n_step,
        gamma=model_cfg["rew_gamma"],
        num_envs=args.num_envs,
        device=device,
    )
    normalizer = ObsNormalizer(
        obs_dim=obs_cfg["num_obs"],
        device=device,
        eps=float(obs_cfg.get("norm_eps", 1e-8)),
        clip=float(obs_cfg.get("norm_clip", 10.0)),
    )

    run_id = args.run_id or uuid.uuid4().hex[:8]
    ckpt_dir = Path("outputs/standalone") / run_id
    log.info("Run id: %s  checkpoints: %s", run_id, ckpt_dir)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="f1tenth-genesis",
            name=f"standalone_{run_id}",
            config={**cfg, **vars(args)},
            mode=args.wandb_mode,
        )

    obs, _ = env.reset()
    obs = obs.to(torch.float32)
    normalizer.update(obs)
    act_dim = cfg["env"]["num_actions"]
    global_step = 0
    train_updates = 0
    episode_rewards = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    recent_episode_rewards: deque[float] = deque(maxlen=RECENT_EPISODES_MAX)
    policy_loss_accum = 0.0
    critic_loss_accum = 0.0
    loss_count = 0
    diag = RunningStats()
    last_batch: dict[str, torch.Tensor] | None = None
    t_start = time.perf_counter()

    try:
        while global_step < args.total_steps:
            if buffer.size < args.min_train_samples:
                actions = (
                    torch.rand(
                        args.num_envs, act_dim, device=device, dtype=torch.float32
                    )
                    * 2
                    * clip_actions
                    - clip_actions
                )
            else:
                with torch.no_grad():
                    actions, _ = models.actor(
                        normalizer.normalize(obs),
                        deterministic=False,
                        with_logprob=False,
                    )
                actions = actions.clamp(-clip_actions, clip_actions)

            try:
                next_obs, reward, done, extras = env.step(
                    actions.to(gs.tc_float), n_steps=control_interval
                )
            except gs.GenesisException as exc:
                log.error(
                    "Genesis raised at step %d (likely NaN constraint forces): %s. "
                    "Saving checkpoint and stopping.",
                    global_step,
                    exc,
                )
                save_checkpoint(models, global_step, ckpt_dir, normalizer)
                break
            next_obs = next_obs.to(torch.float32)
            reward = reward.to(torch.float32)
            episode_rewards += reward

            done_bool = done.bool()
            completed_returns = episode_rewards[done_bool]
            if completed_returns.numel() > 0:
                recent_episode_rewards.extend(completed_returns.tolist())
            episode_rewards = torch.where(
                done_bool, torch.zeros_like(episode_rewards), episode_rewards
            )

            accumulate_step_diagnostics(diag, reward, actions, obs, extras)
            diag.add_mean("obs/norm_abs", normalizer.normalize(obs).abs())

            finite_ok = bool(
                torch.isfinite(reward).all() and torch.isfinite(next_obs).all()
            )
            if finite_ok:
                buffer.add(obs, actions, reward, next_obs, done)
                # Update running stats only from finite observations so a NaN/Inf
                # spin transient can never corrupt the normalizer.
                normalizer.update(next_obs)
            else:
                n_bad_reward = int((~torch.isfinite(reward)).sum().item())
                n_bad_obs_envs = int(
                    (~torch.isfinite(next_obs)).any(dim=1).sum().item()
                )
                log.warning(
                    "Non-finite step at %d (reward_bad=%d obs_bad_envs=%d); "
                    "skipping buffer add.",
                    global_step,
                    n_bad_reward,
                    n_bad_obs_envs,
                )
            obs = next_obs
            global_step += 1

            if buffer.size >= args.min_train_samples:
                for _ in range(args.updates_per_step):
                    batch = buffer.sample(args.batch_size)
                    # Buffer stores RAW obs; normalize with current stats at input.
                    batch["obs"] = normalizer.normalize(batch["obs"])
                    batch["next_obs"] = normalizer.normalize(batch["next_obs"])
                    losses = trainer.update(batch)
                    train_updates += 1
                    policy_loss_accum += losses.policy_loss
                    critic_loss_accum += losses.critic_loss
                    loss_count += 1
                last_batch = batch

            if global_step % args.log_interval == 0:
                elapsed = time.perf_counter() - t_start
                steps_per_sec = global_step / max(elapsed, 1e-6)
                mean_ep_reward = (
                    sum(recent_episode_rewards) / len(recent_episode_rewards)
                    if recent_episode_rewards
                    else float("nan")
                )
                mean_policy_loss = (
                    policy_loss_accum / loss_count if loss_count else float("nan")
                )
                mean_critic_loss = (
                    critic_loss_accum / loss_count if loss_count else float("nan")
                )
                buffer_fill_pct = 100.0 * buffer.size / buffer.capacity
                log.info(
                    "step=%d buffer=%d/%d (%.1f%%) train_updates=%d steps/s=%.1f "
                    "policy_loss=%.4f critic_loss=%.4f mean_ep_reward=%.4f (n=%d)",
                    global_step,
                    buffer.size,
                    buffer.capacity,
                    buffer_fill_pct,
                    train_updates,
                    steps_per_sec,
                    mean_policy_loss,
                    mean_critic_loss,
                    mean_ep_reward,
                    len(recent_episode_rewards),
                )
                mean_q = float("nan")
                if last_batch is not None:
                    with torch.no_grad():
                        mean_q = (
                            models.critic1(last_batch["obs"], last_batch["action"])
                            .mean()
                            .item()
                        )
                nstep_buf_reward_mean = (
                    float(buffer.reward[: buffer.size].mean())
                    if buffer.size > 0
                    else float("nan")
                )

                log.info(
                    "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                    "progress=%.4f speed=%.4f oob_penalty=%.4f tyre_slip=%.4f "
                    "smooth=%.4f | nstep_buf_reward=%.4f mean_Q=%.4f",
                    diag.mean("reward/step"),
                    diag.vmin("reward/step"),
                    diag.vmax("reward/step"),
                    diag.mean("reward_term/progress"),
                    diag.mean("reward_term/speed"),
                    diag.mean("reward_term/oob_penalty"),
                    diag.mean("reward_term/tyre_slip_penalty"),
                    diag.mean("reward_term/smoothness"),
                    nstep_buf_reward_mean,
                    mean_q,
                )
                log.info(
                    "  env: speed=%.3f lat_err=%.3f oob_frac=%.3f progress_ds=%.4f | "
                    "throttle[%.2f..%.2f] steer[%.2f..%.2f] obs_absmax=%.2f "
                    "norm_obs_absmax=%.2f",
                    diag.mean("metric/speed_xy"),
                    diag.mean("metric/lateral_error"),
                    diag.mean("metric/oob_mask"),
                    diag.mean("metric/progress_ds"),
                    diag.vmin("action/throttle"),
                    diag.vmax("action/throttle"),
                    diag.vmin("action/steer"),
                    diag.vmax("action/steer"),
                    diag.vmax("obs/abs"),
                    diag.vmax("obs/norm_abs"),
                )
                log.info(
                    "  terminations: time_out=%d oob=%d not_moving=%d invalid=%d lap=%d",
                    int(diag.total("term/time_out")),
                    int(diag.total("term/out_of_bounds")),
                    int(diag.total("term/not_moving")),
                    int(diag.total("term/invalid_state")),
                    int(diag.total("term/lap_finished")),
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "step": global_step,
                            "buffer/size": buffer.size,
                            "train/updates": train_updates,
                            "train/policy_loss": mean_policy_loss,
                            "train/critic_loss": mean_critic_loss,
                            "train/mean_ep_reward": mean_ep_reward,
                            "train/mean_Q": mean_q,
                            "train/nstep_buf_reward": nstep_buf_reward_mean,
                            "perf/steps_per_sec": steps_per_sec,
                            "reward/total_mean": diag.mean("reward/step"),
                            "reward/total_min": diag.vmin("reward/step"),
                            "reward/total_max": diag.vmax("reward/step"),
                            "reward/progress": diag.mean("reward_term/progress"),
                            "reward/speed": diag.mean("reward_term/speed"),
                            "reward/oob_penalty": diag.mean(
                                "reward_term/oob_penalty"
                            ),
                            "reward/tyre_slip_penalty": diag.mean(
                                "reward_term/tyre_slip_penalty"
                            ),
                            "reward/smoothness": diag.mean(
                                "reward_term/smoothness"
                            ),
                            "env/speed_xy": diag.mean("metric/speed_xy"),
                            "env/lateral_error": diag.mean("metric/lateral_error"),
                            "env/oob_frac": diag.mean("metric/oob_mask"),
                            "env/progress_ds": diag.mean("metric/progress_ds"),
                            "action/throttle_max": diag.vmax("action/throttle"),
                            "action/steer_max": diag.vmax("action/steer"),
                            "obs/absmax": diag.vmax("obs/abs"),
                            "term/time_out": diag.total("term/time_out"),
                            "term/out_of_bounds": diag.total("term/out_of_bounds"),
                            "term/not_moving": diag.total("term/not_moving"),
                            "term/invalid_state": diag.total("term/invalid_state"),
                            "term/lap_finished": diag.total("term/lap_finished"),
                        },
                        step=global_step,
                    )
                policy_loss_accum = 0.0
                critic_loss_accum = 0.0
                loss_count = 0
                diag.reset()

            if global_step % args.ckpt_interval == 0:
                save_checkpoint(models, global_step, ckpt_dir, normalizer)

        save_checkpoint(models, global_step, ckpt_dir, normalizer)
    finally:
        try:
            env.close()
        except Exception as exc:
            log.warning("env.close() failed during shutdown: %s", exc)
        if wandb_run is not None:
            wandb_run.finish()

    log.info("Training finished after %d steps (%d updates).", global_step, train_updates)


if __name__ == "__main__":
    main()
