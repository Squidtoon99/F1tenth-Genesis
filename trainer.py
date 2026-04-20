import torch

from qrsac import QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor, Models
from qrsac.replay import ReplayServer
import torch.nn as nn
from config import Config
import logging
import wandb
from task import TaskServer
from tqdm import tqdm
import time 
from db.models import TrainingSession

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = Models(
        actor=make_policy_network(cfg).to(device),
        critic1=make_q_network(cfg).to(device),
        critic2=make_q_network(cfg).to(device),
        critic1_target=make_target_q_network(cfg).to(device),
        critic2_target=make_target_q_network(cfg).to(device),
    )

    models.critic1_target.load_state_dict(models.critic1.state_dict())
    models.critic2_target.load_state_dict(models.critic2.state_dict())

    trainer = QRSACTrainer(
        models,
        gamma=cfg.model["rew_gamma"],
        n_step=cfg.model["n_step"],
        alpha=0.1,
        smooth_factor=0.005,
        device=device
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
    task_server.warm_up = True

    try:
        log.info("Waiting for replay server to have minimum samples...")
        global_train_step = 0
        with wandb.init(
            project="f1tenth-genesis",
            name=f"trainer_{cfg.session_id}",
            config=cfg.dict(),
        ) as run:
            task_server.wandb_id = run.id
            replay_server.block_and_wait(minimum_samples=cfg.model["minimum_train_samples"])
            log.info("Starting training loop")
            while True:
                epoch_policy_loss = 0.0
                epoch_critic_loss = 0.0
                for _ in tqdm(range(cfg.model["batches_per_epoch"]), desc="Training Batches"):
                    transition_data = replay_server.uniform_sample_batch()
                    losses = trainer.update(transition_data)
                    global_train_step += 1

                    run.log(
                        {
                            "policy_loss": losses.policy_loss,
                            "critic_loss": losses.critic_loss,
                        }
                    )
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

                if current_version <= 5 or current_version % 5 == 0:
                    task_server.needs_eval = True
                    logging.info("Signaled workers to run evaluation episodes.")
    except KeyboardInterrupt:
        pass
    replay_server.stop()


if __name__ == "__main__":
    cfg = Config()
    cfg.redis.set("done", "0")

    if cfg.db_session_factory is not None:
        with cfg.db_session_factory() as session:
            # find training session for current session_id or create if not exists
            training_session = (
                session.query(TrainingSession).filter_by(id=cfg.session_id).first()
            )

            if training_session is None:
                training_session = TrainingSession(
                    id=cfg.session_id,
                    name=f"Session {cfg.session_id}",
                    tracks=[cfg.env["track"]],
                )
                session.add(training_session)
                session.commit()
                logging.info(f"Created new training session with ID {cfg.session_id}")
    try:
        train_loop(cfg)
    finally:
        cfg.redis.set("done", "1")
