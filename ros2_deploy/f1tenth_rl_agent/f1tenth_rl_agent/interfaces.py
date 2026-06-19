"""Shared topic names, array layouts, and constants for the f1tenth_rl_agent stack.

This module is the code mirror of ``ros2_deploy/INTERFACES.md``. Every node imports
from here so the contract stays consistent across independently-developed nodes.
"""

from __future__ import annotations

# --- Simulator topics (f1tenth_gym_ros) ---------------------------------------
TOPIC_ODOM = "/ego_racecar/odom"
TOPIC_MAP = "/map"
TOPIC_DRIVE = "/drive"
TOPIC_INITIALPOSE = "/initialpose"

# --- Internal agent topics ----------------------------------------------------
TOPIC_TRACK_CENTERLINE = "/rl/track/centerline"
TOPIC_TRACK_WIDTHS = "/rl/track/widths"
TOPIC_TRACK_MARKERS = "/rl/track/markers"
TOPIC_OBSERVATION = "/rl/observation"
TOPIC_ACTION = "/rl/action"
TOPIC_FUTURE_POINTS = "/rl/obs_debug/future_points"
TOPIC_METRICS = "/rl/metrics"

# --- Frames -------------------------------------------------------------------
FRAME_MAP = "map"
FRAME_BASE_LINK = "ego_racecar/base_link"

# --- Dimensions ---------------------------------------------------------------
NUM_OBS = 372
NUM_ACTIONS = 2

# Observation field slices (start, stop) within the 372-dim vector.
OBS_LIN_VEL = (0, 2)
OBS_ANG_VEL = (2, 3)
OBS_LIN_ACC = (3, 5)
OBS_LAST_ACTION = (5, 7)
OBS_TRACK_PROGRESS = (7, 9)
OBS_CENTERLINE_ANGLE = (9, 10)
OBS_CENTERLINE_DISTANCE = (10, 11)
OBS_CONTACT_FLAG = (11, 12)
OBS_FUTURE_POINTS = (12, 372)

# Metrics layout.
METRICS_LEN = 6
METRICS_LAP_COUNT = 0
METRICS_LAST_LAP_TIME = 1
METRICS_MAX_PROGRESS = 2
METRICS_LATERAL_ERROR = 3
METRICS_OOB_FLAG = 4
METRICS_SPEED = 5

# --- Policy / vehicle constants (DEFAULT_CONFIG in config.py) -----------------
MAX_SPEED = 10.0
MAX_STEER = 0.4189
CLIP_ACTIONS = 1.0
CONTACT_MARGIN_M = 0.08
FUTURE_TRACK_NUM_POINTS = 60
FUTURE_TRACK_HORIZON_S = 6.0
FUTURE_TRACK_WIDTH = 2.2
HIDDEN_LAYERS = [512, 512, 512]
ACT_LIMIT = 1.0
CONTROL_HZ = 10.0


def default_obs_cfg() -> dict:
    """Return the obs_cfg dict expected by obs_core.build_observation."""
    return {
        "num_obs": NUM_OBS,
        "contact_margin_m": CONTACT_MARGIN_M,
        "future_track_num_points": FUTURE_TRACK_NUM_POINTS,
        "future_track_horizon_s": FUTURE_TRACK_HORIZON_S,
        "future_track_width": FUTURE_TRACK_WIDTH,
    }
