import json
import logging
from redis import Redis
from functools import wraps


class RedisWrapper:
    def __init__(self, redis_client: Redis, session_id: str):
        self.redis_client = redis_client
        self.session_id = session_id

    def _prefix_key(self, key: str) -> str:
        return f"{self.session_id}:{key}"

    def __getattr__(self, name):
        attr = getattr(self.redis_client, name)

        if callable(attr):

            @wraps(attr)
            def wrapper(*args, **kwargs):
                if args and isinstance(args[0], str):
                    args = (self._prefix_key(args[0]),) + args[1:]
                return attr(*args, **kwargs)

            return wrapper
        else:
            return attr


DEFAULT_CONFIG = {
    "obs": {
        "num_obs": 371,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        },
        "contact_margin_m": 0.08,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        "future_track_width": 2.0,
    },
    "env": {
        "num_actions": 2,
        "episode_length": 25.0,
        "clip_actions": 1.0,
        "simulate_action_latency": True,
        "term_oob_margin_m": 0.15,
        "term_oob_max_consecutive": 15,
        "term_speed_threshold": 0.2,
        "term_not_moving_time_s": 2.0,
        "term_not_moving_min_ds": 1e-3,
        "term_heading_error_rad": 3.0,
        "target_laps": 0,
        "car_spawn_pos": (0.0, 0.0, 0.01),
        "car_spawn_rot": (0.0, 0.0, 0.0),
        "joint_names": [
            "left_rear_wheel_joint",
            "right_rear_wheel_joint",
        ],
        "default_joint_angles": {
            "left_rear_wheel_joint": 0.0,
            "right_rear_wheel_joint": 0.0,
        },
        "max_speed": 7.0,  # m/s
        "max_steer": 0.4189,  # radianss
        "wheelbase": 0.325,
        "track_width": 0.20,
        "wheel_radius": 0.05,
        "track": "Montreal",
    },
    "reward": {
        "progress_k_fwd": 5.0,
        "progress_k_back": 5.0,
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.5,
        "oob_k": 10.0,
        "reward_scales": {
            "progress": 1.4,
            "oob_penalty": 1.0,
        },
    },
    "model": {
        "hidden_layers": [256, 256],
        "num_quantiles": 32,
    },
}


class Config:
    """Configuration class for the application."""

    def __init__(self, session_id: str, redis_uri: str | None):
        self.session_id = session_id
        self.redis_uri = redis_uri or "redis://localhost:6379/0"
        self.redis_client = Redis.from_url(self.redis_uri, decode_responses=True)
        self.redis_binary_client = Redis.from_url(
            self.redis_uri, decode_responses=False
        )
        self.redis = RedisWrapper(self.redis_client, self.session_id)
        self.redis_b = RedisWrapper(self.redis_binary_client, self.session_id)
        self.logger = logging.getLogger("Genesis")
        self._cfg = {}
        # Load redis overrides

        for key in ["obs", "env", "reward"]:
            redis_key = f"config:{key}"
            if self.redis.exists(redis_key):
                try:
                    if value := self.redis.get(redis_key):
                        self._cfg[key] = {
                            **json.loads(str(value)),
                            **DEFAULT_CONFIG[key],
                        }  # Override defaults with Redis values
                        self.logger.info(
                            f"Loaded config override for '{key}' from Redis."
                        )
                    else:
                        self._cfg[key] = DEFAULT_CONFIG[key]
                        self.logger.info(
                            f"No value found for '{redis_key}' in Redis. Using default config."
                        )
                except Exception as e:
                    self.logger.error(
                        f"Failed to load config override for '{key}': {e}"
                    )

    @property
    def obs(self):
        return self._cfg["obs"]

    @property
    def env(self):
        return self._cfg["env"]

    @property
    def reward(self):
        return self._cfg["reward"]

    @property
    def model(self):
        return self._cfg["model"]
