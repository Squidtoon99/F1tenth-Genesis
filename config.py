import json
import logging
import os

from dotenv import load_dotenv
from redis import Redis
from functools import wraps

from db import bootstrap_database
from param import S3ParameterServer
import wandb

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
        "num_obs": 380,
        # Observation normalization is now done with empirical running statistics in
        # the trainer (ObsNormalizer), driven by values actually experienced. Keep
        # the env-side fixed scales at 1.0 so near-raw obs reach the normalizer.
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "lin_acc": 1.0,
        },
        # Loose guard only: bound a rare spin/contact transient before it can skew
        # the running variance. Real scaling is handled by the trainer normalizer.
        "clip_obs": 50.0,
        # Trainer-side ObsNormalizer parameters.
        "norm_clip": 10.0,
        "norm_eps": 1e-8,
        "contact_margin_m": 0.08,
        # 1v1: when enabled, an opponent-relative block of size opponent_obs_dim is
        # appended to the observation (num_obs becomes 380 + opponent_obs_dim). Off
        # by default so the solo (1v0) observation stays 380-dim and unchanged.
        "enable_opponent_obs": False,
        "opponent_obs_dim": 7,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        # Floor for the speed-scaled lookahead so the policy still sees the upcoming
        # track while stopped/crawling (speed*horizon -> 0 collapses all samples
        # onto the current point). Only affects speeds below
        # future_track_min_lookahead_m / future_track_horizon_s.
        "future_track_min_lookahead_m": 5.0,
        # Deprecated: corridor edges use per-vertex w_tr_left_m / w_tr_right_m.
        "future_track_width": 2.2,
    },
    "env": {
        "num_actions": 2,
        "episode_length": 25.0,
        # control_dt = sim_dt * control_interval = 0.1s (10 Hz control). When
        # changing sim_dt, adjust control_interval inversely to keep control_dt
        # and episode_length semantics constant.
        "control_interval": 10,
        "sim_dt": 0.01,
        # Tuned via scripts/sweep_physics_integration.py: substep counts 2..10 pass
        # the physics_check stability gate for normal upright driving. Raising
        # substeps does NOT cure the high-speed spin-out contact NaN (confirmed at
        # 8 substeps / 1.25ms); that instability is driven by the policy being
        # steered into the boundary, so it is addressed upstream (clean centerline,
        # obs clipping, speed-capped reward, fast OOB termination), not here.
        "sim_substeps": 4,
        "solver_iterations": 50,
        "solver_ls_iterations": 50,
        "show_fps": False,
        # Opt-in torch.compile of the pure-tensor observation math. Off by default
        # (no warmup/recompile risk); enable on the GPU training target after
        # confirming a steady-state throughput gain that outweighs compile warmup.
        "compile_obs": False,
        "clip_actions": 1.0,
        "simulate_action_latency": True,
        "term_oob_margin_m": 0.15,
        # Strict OOB: chicane cuts end the episode quickly so skipping S-bends
        # cannot amortize off-track time against on-track progress.
        "term_oob_max_consecutive": 2,
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
        # Observation/reset throttle scaling only — longitudinal cap comes from power+drag.
        "max_speed": 15.0,
        "max_steer": 0.44,  # radians (alias for delta_max)
        "delta_max": 0.44,  # radians
        "wheelbase": 0.325,
        "track_width": 0.20,
        "wheel_radius": 0.05,
        "f_drive_max": 23.0,
        "f_brake_max": 23.0,
        "power_max": 255.0,
        "k_drive_front": 0.5,
        "t_delta": 0.1,
        "c_roll": 0.0,
        "dragcoeff": 0.075,
        "tire_friction": 0.65,
        "v_eps": 0.1,
        "enable_aero_drag": True,
        "drive_torque_sign": 1.0,
        # Competition sim track (dfr_f1tenth_gym dev-humble maps/IV_2026_SIM).
        "track": "IV_2026_SIM",
        # --- 1v1 opponent (hard 1v1: exactly one opponent) ---
        # opponent_strategy: None (1v0 / solo), "scripted" (centerline follower),
        # or "policy" (frozen-policy self-play opponent; deferred training loop).
        "opponent_strategy": None,
        # Scripted opponent: centerline follower kept below ego pace so an overtake
        # is feasible. target speed is in m/s; spawn gap is meters ahead of the ego.
        "opponent_target_speed": 3.0,
        "opponent_spawn_gap_m": 7.0,
        "opponent_kp_ey": 1.0,
        "opponent_kh_heading": 1.0,
        # Collision termination: end the episode when the cars are within
        # collision_dist_m. No shaped collision penalty (forfeited progress is the
        # avoidance incentive).
        "term_on_collision": True,
        "collision_dist_m": 0.4,
    },
    "reward": {
        # GT Sophy-aligned reward: course progress (primary), off-course penalty
        # (~time-off x speed^2), tyre-slip penalty, and a small smoothness shaping
        # term. No explicit speed reward: speed is induced purely by progress per
        # step under the gamma=0.9896 discount, exactly as in GT Sophy.
        "progress_k_fwd": 5.0,
        "progress_k_back": 5.0,
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.2,
        # GT Sophy off-course penalty R_soc = -(time off course) * speed^2. The
        # per-step time off course is constant and folds into oob_k; tuned so the
        # penalty at racing speed (~6 m/s) is comparable to the previous shaping
        # while escalating quadratically with speed for fast excursions.
        "oob_k": 0.3,
        "lateral_k": 0.5,
        # 1v1 passing reward gain: per-step reward = passing_k * (ego_ds - opp_ds),
        # i.e. track position gained on the opponent. Only active when a "passing"
        # entry is added to reward_scales (the trainer does this for 1v1), so 1v0 is
        # unaffected.
        "passing_k": 5.0,
        # Global downscale applied to the summed reward to keep per-step total and
        # value targets O(1) (progress alone was ~9/step before). Preserves the
        # relative balance between the individual reward terms.
        "global_reward_scale": 0.2,
        "reward_scales": {
            "progress": 5.0,
            "lateral": 1.0,
            "oob_penalty": 0.6,
            "tyre_slip_penalty": 0.05,
            # Mild jerk penalty to curb bang-bang throttle/steer.
            "smoothness": 0.05,
        },
    },
    "model": {
        "hidden_layers": [512, 512, 512],
        "num_quantiles": 32,
        "rew_gamma": 0.9896,
        "n_step": 7,
        "replay_tables": ["1v0"],  # "1v1", "mistake_learning"],
        "minimum_train_samples": 40000,
        "batches_per_epoch": 6000,
        "replay_buffer_limit": 10**7,
        "batch_size": 1024,
        "update_to_data_ratio": 0.01,
    },
    "selfplay": {
        "snapshot_interval": 20_000,
        "refresh_interval": 5_000,
        "pool_size": 5,
        "sample_mode": "mixed",
        "mixed_latest_prob": 0.8,
    },
}


class Config:
    """Configuration class for the application."""

    def __init__(
        self,
        session_id: str | None = None,
        bootstrap_db: bool = True,
    ):
        load_dotenv()
        self.session_id = session_id or os.getenv("SESSION_ID", "1")

        # Redis Config

        self.redis_uri = os.getenv("REDIS_URI") or "redis://localhost:6379/0"

        self._redis_client = Redis.from_url(self.redis_uri, decode_responses=True)
        self._redis_binary_client = Redis.from_url(
            self.redis_uri, decode_responses=False
        )
        self.redis = RedisWrapper(self._redis_client, self.session_id)
        self.redis_b = RedisWrapper(self._redis_binary_client, self.session_id)
        self.logger = logging.getLogger("Genesis")

        # Postgres
        self.db_engine = None 
        self.db_session_factory = None
        if database_url := os.getenv("POSTGRES_URI"):
            self.db_engine = None
            self.db_session_factory = None

            if bootstrap_db:
                try:
                    self.db_engine, self.db_session_factory = bootstrap_database(
                        database_url=database_url
                    )
                    self.logger.info("Database bootstrap completed successfully.")
                except Exception as exc:
                    if "the database system is starting up" in str(exc):
                        self.logger.warning(
                            "Database is still starting up."
                        )
                    else:
                        self.logger.exception(f"Database bootstrap failed: {exc}")

        
        self._cfg = DEFAULT_CONFIG.copy()
        # Load redis overrides

        for key in ["obs", "env", "reward", "model"]:
            redis_key = f"config:{key}"
            try:
                if value := self.redis.get(redis_key):
                    self._cfg[key] = {
                        **DEFAULT_CONFIG[key],
                        **json.loads(str(value)),
                    }
                    self.logger.info(f"Loaded config override for '{key}' from Redis.")
                else:
                    self._cfg[key] = DEFAULT_CONFIG[key]
                    self.logger.info(
                        f"No value found for '{redis_key}' in Redis. Using default config."
                    )
            except Exception as e:
                self.logger.error(f"Failed to load config override for '{key}': {e}")

        # Load S3 config from environment variables
        self._param_server = S3ParameterServer.from_env()

    def rkey(self, key: str) -> str:
        """Helper method to get a Redis key with session prefix."""
        return f"{self.session_id}:{key}"

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

    @property
    def s3_parameter_server(self):
        return self._param_server

    def dict(self):
        return self._cfg

    def db_session(self):
        if self.db_session_factory is None:
            raise RuntimeError(
                "Database session factory is not initialized. "
                "Check database URL/configuration or enable bootstrap_db."
            )
        return self.db_session_factory()
