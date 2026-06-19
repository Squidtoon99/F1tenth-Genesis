"""Parity test: obs_core vs the real f1tenth_env observation pipeline.

This guarantees the deployed observation equals the one the policy trained on. It
loads the original ``f1tenth_env/observations.py`` and ``f1tenth_env/utils.py`` as
standalone modules (so we do not pull in the genesis-heavy ``env.py``) and compares
their ``build_observation`` output against ``obs_core.ObservationBuilder``.

Requires ``genesis`` to be importable (it is in the training venv). The test
configures the few module-level ``genesis`` constants (tc_float, device, EPS) that
the pure geometry helpers need, without a full ``gs.init()``.
"""

import importlib.util
import os
import sys
import tempfile
import types

import numpy as np
import pytest

# Keep matplotlib (imported transitively by genesis) from complaining.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

gs = pytest.importorskip("genesis")

import torch  # noqa: E402

from f1tenth_rl_agent import obs_core  # noqa: E402
from f1tenth_rl_agent.interfaces import default_obs_cfg  # noqa: E402

# Repo root: ros2_deploy/f1tenth_rl_agent/test -> up 3.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _configure_genesis():
    gs.tc_float = torch.float32
    gs.device = torch.device("cpu")
    if getattr(gs, "EPS", None) is None:
        gs.EPS = 1e-12


def _load_module(name: str, relpath: str):
    """Load a repo module by file path without running f1tenth_env/__init__.py."""
    path = os.path.join(_REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_requests():
    """Stub `requests` so utils.py module-level load_tracks() returns immediately."""
    if "requests" in sys.modules:
        return

    class _Resp:
        status_code = 404

        def json(self):
            return {}

    stub = types.ModuleType("requests")
    stub.get = lambda *a, **k: _Resp()
    sys.modules["requests"] = stub


@pytest.fixture(scope="module")
def real_modules():
    _configure_genesis()
    _stub_requests()
    real_utils = _load_module("real_f1tenth_utils", "f1tenth_env/utils.py")
    real_obs = _load_module("real_f1tenth_observations", "f1tenth_env/observations.py")
    return real_utils, real_obs


def _make_track(n=240):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    cl = np.stack([25.0 * np.cos(th), 14.0 * np.sin(th)], axis=-1).astype(np.float32)
    wl = np.full(n, 1.6, dtype=np.float32)
    wr = np.full(n, 1.4, dtype=np.float32)
    return cl, wl, wr


def _yaw_quat_wxyz(yaw: np.ndarray) -> torch.Tensor:
    half = 0.5 * yaw
    w = np.cos(half)
    z = np.sin(half)
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    quat[:, 0] = w
    quat[:, 3] = z
    return torch.tensor(quat, dtype=torch.float32)


def test_obs_parity(real_modules):
    real_utils, real_obs = real_modules
    device = torch.device("cpu")
    cl, wl, wr = _make_track()
    obs_cfg = default_obs_cfg()

    builder = obs_core.ObservationBuilder(cl, wl, wr, obs_cfg, device=device)

    track_state = {
        "centerline": cl,
        "w_tr_left_torch": torch.tensor(wl, dtype=torch.float32),
        "w_tr_right_torch": torch.tensor(wr, dtype=torch.float32),
        "track_geom_cache": {},
        "frenet_step_cache": {},
    }

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(20):
        b = 4
        # sample positions near the track with random lateral offset
        idx = rng.integers(0, cl.shape[0], size=b)
        base = cl[idx]
        offset = rng.uniform(-1.0, 1.0, size=(b, 2)).astype(np.float32)
        pos_xy = base + offset
        pos = np.concatenate([pos_xy, np.zeros((b, 1), np.float32)], axis=1)
        base_pos = torch.tensor(pos, dtype=torch.float32)

        yaw = rng.uniform(-np.pi, np.pi, size=b).astype(np.float32)
        base_quat = _yaw_quat_wxyz(yaw)

        base_lin_vel = torch.tensor(
            rng.uniform(-1, 6, size=(b, 3)).astype(np.float32)
        )
        base_lin_vel[:, 2] = 0.0
        base_ang_vel = torch.tensor(
            rng.uniform(-1, 1, size=(b, 3)).astype(np.float32)
        )
        base_lin_acc = torch.tensor(
            rng.uniform(-2, 2, size=(b, 3)).astype(np.float32)
        )
        base_lin_acc[:, 2] = 0.0
        last_actions = torch.tensor(
            rng.uniform(-1, 1, size=(b, 2)).astype(np.float32)
        )

        # fresh caches per sample so frenet recomputes
        track_state["track_geom_cache"] = {}
        track_state["frenet_step_cache"] = {}
        episode_steps = torch.zeros(b, dtype=torch.int32)
        step_state = real_utils.build_step_state(
            base_pos=base_pos,
            episode_steps_buf=episode_steps,
            track_state=track_state,
            device=device,
            cache_id="parity",
        )

        real = real_obs.build_observation(
            num_obs=372,
            num_envs=b,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat=base_quat,
            centerline=cl,
            obs_cfg=obs_cfg,
            step_state=step_state,
            device=device,
        )

        mine = builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
        )

        diff = (real - mine).abs().max().item()
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, f"obs parity diff too large: {max_diff}"
