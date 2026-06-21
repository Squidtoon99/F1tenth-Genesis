"""Tests for 1v1 collision termination (terminations.collision_mask) and the
guarantee that the existing 1v0 termination outputs are unchanged.

``terminations.py`` uses package-relative imports, so (as in test_passing_reward)
we register a small package shim pointing ``.utils`` at the loaded audit module.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def term_mod(real_modules):
    pkg_name = "f1tenth_env_term_under_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, "f1tenth_env")]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.utils"] = real_modules.utils

    path = os.path.join(_REPO_ROOT, "f1tenth_env", "terminations.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.terminations", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.terminations"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --- collision threshold ------------------------------------------------------
def test_collision_threshold_just_inside_and_outside(term_mod):
    ego = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    opp = torch.tensor(
        [[0.39, 0.0], [0.41, 0.0], [10.0, 0.0]], dtype=torch.float32
    )
    mask = term_mod.collision_mask(ego, opp, 0.4)
    assert mask.tolist() == [True, False, False]


def test_collision_symmetric_distance(term_mod):
    ego = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    opp = torch.tensor([[1.2, 2.1]], dtype=torch.float32)
    # distance = sqrt(0.04 + 0.01) ~ 0.2236
    assert term_mod.collision_mask(ego, opp, 0.3).item() is True
    assert term_mod.collision_mask(ego, opp, 0.2).item() is False


def test_collision_batched(term_mod):
    g = torch.Generator().manual_seed(1)
    ego = torch.rand(64, 2, generator=g)
    opp = torch.rand(64, 2, generator=g)
    sep = torch.linalg.norm(ego - opp, dim=-1)
    expected = sep < 0.5
    assert torch.equal(term_mod.collision_mask(ego, opp, 0.5), expected)


# --- 1v0 no-regression --------------------------------------------------------
def test_existing_terminations_have_no_collision_key(term_mod):
    """compute_terminations (the shared 1v0/1v1 core) never adds a collision key;
    collision is layered on by the env only when an opponent exists."""
    n = 4
    boundary = {
        "ey": torch.zeros(n),
        "w_l_s": torch.full((n,), 1.5),
        "w_r_s": torch.full((n,), 1.5),
        "boundary_dist": torch.full((n,), 1.5),
    }
    step_state = {
        "boundary": boundary,
        "frenet": {"seg_dir": torch.tensor([[1.0, 0.0]] * n)},
        "progress_ds": torch.full((n,), 0.1),
    }
    term_params = {
        "term_oob_margin_m": 0.15,
        "term_oob_max_consecutive": 10,
        "term_speed_threshold": 0.2,
        "term_not_moving_min_ds": 1e-3,
        "not_moving_steps_threshold": 20,
        "term_heading_error_rad": 3.0,
        "target_laps": 0,
    }
    term_state = {
        "oob_consecutive_buf": torch.zeros(n, dtype=torch.int32),
        "not_moving_steps_buf": torch.zeros(n, dtype=torch.int32),
    }
    base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
    reset, extras, _ = term_mod.compute_terminations(
        step_state=step_state,
        episode_steps_buf=torch.zeros(n, dtype=torch.int32),
        max_episode_steps=250,
        base_pos=torch.zeros(n, 3),
        base_quat=base_quat,
        base_lin_vel=torch.tensor([[5.0, 0.0, 0.0]] * n),
        base_ang_vel=torch.zeros(n, 3),
        lap_count_buf=torch.zeros(n, dtype=torch.int32),
        term_state=term_state,
        term_params=term_params,
    )
    assert "collision" not in extras
    assert not bool(reset.any())  # moving, on-track, not timed out -> no termination
