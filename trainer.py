from qrsac import QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor, Models
from qrsac.replay import ReplayServer
import torch.nn as nn
from config import Config
import logging
import wandb
from task import TaskServer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def make_policy_network(cfg):
    obs_dim = cfg.obs["num_obs"]
    action_dim = cfg.env["num_actions"]
    return SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg.model["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )


def make_q_network(cfg):
    obs_dim = cfg.obs["num_obs"]
    action_dim = cfg.env["num_actions"]
    return QuantileCritic(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg.model["hidden_layers"],
        num_quantiles=32,
    )


def make_target_q_network(cfg):
    target_q = make_q_network(cfg)
    for param in target_q.parameters():
        param.requires_grad = False
    return target_q


def train_loop(cfg: "Config"):
    log = logging.getLogger("Trainer")
    models = Models(
        actor=make_policy_network(cfg),
        critic1=make_q_network(cfg),
        critic2=make_q_network(cfg),
        critic1_target=make_target_q_network(cfg),
        critic2_target=make_target_q_network(cfg),
    )

    trainer = QRSACTrainer(
        models,
        gamma=cfg.model["rew_gamma"],
        n_step=cfg.model["n_step"],
        alpha=0.1,
        smooth_factor=0.005,
    )

    replay_server = ReplayServer(
        cfg,
        tables=cfg.model["replay_tables"],
        obs_dim=cfg.obs["num_obs"],
        act_dim=cfg.env["num_actions"],
    )

    param_server = cfg.s3_parameter_server
    current_version = param_server.latest_version() or 0

    task_server = TaskServer(cfg)

    replay_server.start()

    log.info("Waiting for replay server to have minimum samples...")
    replay_server.block_and_wait(minimum_samples=cfg.model["minimum_train_samples"])

    try:
        log.info("Starting training loop")
        global_train_step = 0
        with wandb.init(
            project="f1tenth-genesis",
            name=f"trainer_{cfg.session_id}",
            config=cfg.dict(),
        ) as run:
            # run.watch(models.actor, log="gradients", log_freq=500)
            while True:
                epoch_policy_loss = 0.0
                epoch_critic_loss = 0.0
                for _ in range(cfg.model["batches_per_epoch"]):
                    transition_data = replay_server.uniform_sample_batch()
                    losses = trainer.update(transition_data)
                    global_train_step += 1

                    run.log(
                        {
                            "policy_loss": losses.policy_loss,
                            "critic_loss": losses.critic_loss,
                        }
                    )
                    # reward = transition_data.get("rew", transition_data.get("reward"))
                    # done = transition_data["done"]
                    # action = transition_data.get("act", transition_data.get("action"))
                    # obs = transition_data["obs"]

                    # if reward is None or action is None:
                    #     raise KeyError(
                    #         "Batch must contain reward/rew and action/act fields for logging."
                    #     )

                    # if not torch.is_tensor(reward):
                    #     reward = torch.as_tensor(reward)
                    # if not torch.is_tensor(done):
                    #     done = torch.as_tensor(done)
                    # if not torch.is_tensor(action):
                    #     action = torch.as_tensor(action)
                    # if not torch.is_tensor(obs):
                    #     obs = torch.as_tensor(obs)

                    # replay_sizes = {
                    #     f"train/replay/size_{table_name}": float(table.size)
                    #     for table_name, table in replay_server.tables.items()
                    # }
                    # replay_total = float(
                    #     sum(table.size for table in replay_server.tables.values())
                    # )

                    # epoch_policy_loss += losses.policy_loss
                    # epoch_critic_loss += losses.critic_loss

                    # run.log(
                    #     {
                    #         "train/step": global_train_step,
                    #         "train/loss/policy": losses.policy_loss,
                    #         "train/loss/critic": losses.critic_loss,
                    #         "train/batch/reward_mean": float(reward.mean().item()),
                    #         "train/batch/reward_std": float(
                    #             reward.std(unbiased=False).item()
                    #         ),
                    #         "train/batch/reward_min": float(reward.min().item()),
                    #         "train/batch/reward_max": float(reward.max().item()),
                    #         "train/batch/done_rate": float(done.float().mean().item()),
                    #         "train/batch/action_mean_abs": float(
                    #             action.abs().mean().item()
                    #         ),
                    #         "train/batch/action_std": float(
                    #             action.std(unbiased=False).item()
                    #         ),
                    #         "train/batch/obs_mean_abs": float(obs.abs().mean().item()),
                    #         "train/batch/obs_std": float(
                    #             obs.std(unbiased=False).item()
                    #         ),
                    #         "train/replay/size_total": replay_total,
                    #         **{f"train/{k}": v for k, v in losses.diagnostics.items()},
                    #         **replay_sizes,
                    #     },
                    #     step=global_train_step,
                    # )

                epoch_batches = float(cfg.model["batches_per_epoch"])
                current_version += 1
                param_server.publish_actor(
                    actor_state_dict=models.actor.state_dict(),
                    version=current_version,
                    metadata={
                        "source": "trainer",
                        "session_id": cfg.session_id,
                    },
                )
                logging.info(
                    f"Published new policy version {current_version} to S3 parameter server."
                )
                run.log(
                    {
                        "train/step": global_train_step,
                        "train/publish/version": current_version,
                        "train/epoch/policy_loss_mean": epoch_policy_loss
                        / epoch_batches,
                        "train/epoch/critic_loss_mean": epoch_critic_loss
                        / epoch_batches,
                    },
                    step=global_train_step,
                )
                task_server.warm_up = False

                task_server.needs_eval = True
                logging.info("Signaled workers to run evaluation episodes.")
    except KeyboardInterrupt:
        pass
    replay_server.stop()


if __name__ == "__main__":
    cfg = Config(session_id="0", redis_uri="redis://localhost:6379")
    train_loop(cfg)
