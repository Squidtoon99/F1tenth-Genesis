"""Genesis <-> IRL vehicle physics alignment toolkit.

A self-contained package for profiling the real F1TENTH car and the Genesis
simulation through the *same* open-loop maneuver schedule, then fitting Genesis
physics parameters so the two agree in the high-speed racing band. Low-speed IRL
artifacts (VESC cogging / stiction / crawl "crunching") are deliberately excluded
from the fit objective via a ``v_fit_min`` speed gate.

Run via ``python -m vehicle_calibration <subcommand>`` from the repo root.
"""

from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
MANEUVERS_DIR = os.path.join(PACKAGE_DIR, "maneuvers")
RUNS_DIR = os.path.join(PACKAGE_DIR, "runs")

DEFAULT_PROFILE_YAML = os.path.join(MANEUVERS_DIR, "carpet_profile.yaml")
DEFAULT_SETTINGS_YAML = os.path.join(MANEUVERS_DIR, "defaults.yaml")


def run_dir(run_id: str, create: bool = False) -> str:
    """Return the artifact directory for ``run_id`` (optionally creating it)."""
    path = os.path.join(RUNS_DIR, run_id)
    if create:
        os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    return path
