"""Pure-Python tests for the vendored actor + checkpoint loading."""

import os
import tempfile

import torch
import torch.nn as nn

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.policy_model import SquashedGaussianMLPActor, load_actor


def _make_actor():
    return SquashedGaussianMLPActor(
        obs_dim=ifc.NUM_OBS,
        act_dim=ifc.NUM_ACTIONS,
        hidden_sizes=ifc.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=ifc.ACT_LIMIT,
    )


def test_forward_shape_and_range():
    actor = _make_actor()
    actor.eval()
    obs = torch.zeros(4, ifc.NUM_OBS)
    with torch.no_grad():
        action, logp = actor(obs, deterministic=True, with_logprob=False)
    assert action.shape == (4, ifc.NUM_ACTIONS)
    assert logp is None
    assert bool((action.abs() <= ifc.ACT_LIMIT + 1e-5).all())


def test_load_checkpoint_actor_key():
    actor = _make_actor()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        # standalone_trainer.save_checkpoint format
        torch.save({"step": 10, "actor": actor.state_dict()}, path)
        loaded = load_actor(
            checkpoint_path=path,
            obs_dim=ifc.NUM_OBS,
            act_dim=ifc.NUM_ACTIONS,
            hidden_sizes=ifc.HIDDEN_LAYERS,
            act_limit=ifc.ACT_LIMIT,
            state_dict_key="actor",
            device=torch.device("cpu"),
        )
    obs = torch.randn(2, ifc.NUM_OBS)
    with torch.no_grad():
        a0, _ = actor(obs, deterministic=True, with_logprob=False)
        a1, _ = loaded(obs, deterministic=True, with_logprob=False)
    assert torch.allclose(a0, a1, atol=1e-6)


def test_load_raw_state_dict():
    actor = _make_actor()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "raw.pt")
        torch.save(actor.state_dict(), path)
        loaded = load_actor(
            checkpoint_path=path,
            obs_dim=ifc.NUM_OBS,
            act_dim=ifc.NUM_ACTIONS,
            hidden_sizes=ifc.HIDDEN_LAYERS,
            act_limit=ifc.ACT_LIMIT,
            state_dict_key="actor",
            device=torch.device("cpu"),
        )
    assert isinstance(loaded, SquashedGaussianMLPActor)
