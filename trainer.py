from qrsac import QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor, Models
from qrsac.replay import ReplayServer
import torch
import torch.nn as nn
from config import Config


def make_policy_network(cfg):
    obs_dim = cfg.obs["obs_dim"]
    action_dim = cfg.env["action_dim"]
    return SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg.model["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )


def make_q_network(cfg):
    obs_dim = cfg.obs["obs_dim"]
    action_dim = cfg.env["action_dim"]
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
    models = Models(
        actor=make_policy_network(cfg),
        critic1=make_q_network(cfg),
        critic2=make_q_network(cfg),
        critic1_target=make_target_q_network(cfg),
        critic2_target=make_target_q_network(cfg),
    )

    trainer = QRSACTrainer(models)

    replay_server = ReplayServer(cfg, tables=cfg.model["replay_tables"])
