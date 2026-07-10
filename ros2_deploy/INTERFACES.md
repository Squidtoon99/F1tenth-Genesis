# f1tenth_rl_agent - Frozen Topic / Message Contract

This document is the source of truth for the ROS 2 interface between every node in
the `f1tenth_rl_agent` package and the `f1tenth_gym_ros` simulator bridge. It is
frozen first so that each node can be developed and tested independently against
mock publishers/subscribers.

All topic names and array layouts are mirrored in code in
`f1tenth_rl_agent/interfaces.py`. If you change anything here, change it there too.

## Conventions

- Frame conventions follow REP-103/REP-105. The world frame is `map`. The car body
  frame is `ego_racecar/base_link` (the odometry child frame).
- Observation / action vectors are transported as `std_msgs/Float32MultiArray` to
  avoid a custom message package. The semantic layout is documented below and is
  identical to the training pipeline in `f1tenth_env/observations.py`.
- Control cadence is **10 Hz** (`control_interval=10 * sim_dt=0.01`), matching the
  rate the policy was trained at.

## Topics consumed from the simulator (`f1tenth_gym_ros`)

| Topic | Type | Notes |
| --- | --- | --- |
| `/ego_racecar/odom` | `nav_msgs/Odometry` | Ground-truth ego pose (`map` frame) and twist (body frame). |
| `/map` | `nav_msgs/OccupancyGrid` | Occupancy grid, used for visualization only. |

## Topics produced to the simulator

| Topic | Type | Notes |
| --- | --- | --- |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Ego drive command (`speed` m/s, `steering_angle` rad). |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Episode reset pose (used by `evaluation_node`). |

## Internal agent topics

### `/rl/track/centerline` - `nav_msgs/Path`
- Frame `map`, QoS `transient_local` (latched) + periodic republish.
- `poses[i].pose.position.{x,y}` are the centerline points in order; `z = 0`.
- Length `N` equals the number of rows in the track CSV.

### `/rl/track/widths` - `std_msgs/Float32MultiArray`
- QoS `transient_local`.
- `layout.dim = [ {label: "point", size: N, stride: 2N}, {label: "lr", size: 2, stride: 2} ]`.
- `data` length `2N`, interleaved per centerline point: `[w_tr_left_0, w_tr_right_0, w_tr_left_1, w_tr_right_1, ...]`.

### `/rl/track/markers` - `visualization_msgs/MarkerArray`
- Frame `map`. LINE_STRIP markers: id 0 = centerline, id 1 = left boundary, id 2 = right boundary.

### `/rl/observation` - `std_msgs/Float32MultiArray`
- `data` length **380**, exact order of `f1tenth_env.observations.build_observation`:

| Index | Field | Source |
| --- | --- | --- |
| `0:2` | body-frame linear velocity `(vx, vy)` × `obs_scales.lin_vel` (1.0) | odom twist |
| `2:3` | body-frame yaw rate `wz` × `obs_scales.ang_vel` (1.0) | odom twist |
| `3:5` | body-frame linear accel `(ax, ay)` × `obs_scales.lin_acc` (1.0) | finite difference of body velocity |
| `5:7` | last action `[throttle, steering]` | last `/rl/action` |
| `7:9` | track progress `[cos, sin]` of Frenet `s/L` | Frenet |
| `9` | centerline heading error (yaw - track tangent, wrapped) | Frenet |
| `10` | signed lateral error `ey` | Frenet |
| `11` | wall-contact flag (`boundary_dist < 0.08`) | Frenet |
| `12:372` | future track points: center/left/right x 60 pts x 2D, **ego frame** | centerline |
| `372:380` | tyre slip `[slip_ratio x4, slip_angle x4]` | **zeros in gym deploy** (no wheel state in f1tenth_gym_ros) |

When the 1v1 opponent observation is enabled (`enable_opponent_obs`), a 7-dim block
is appended and `data` length becomes **387**:

| Index | Field | Source |
| --- | --- | --- |
| `380:382` | opponent position relative to ego, **ego body frame** `(x, y)` (m) | opponent detect/odom |
| `382:384` | opponent velocity relative to ego, **ego body frame** `(vx, vy)` (m/s) | opponent detect/odom |
| `384` | signed along-track gap `s_opp - s_ego` wrapped to `[-L/2, L/2]`, normalized by `L/2` | Frenet |
| `385` | opponent signed lateral offset `ey_opp` (m) | Frenet |
| `386` | presence flag (`1.0` present, else `0.0`) | opponent detect/odom |

The whole block (including the presence flag) is the exact zero sentinel when the
opponent is absent. This matches `f1tenth_env.observations.obs_opponent`.

- The env-side scales are now `1.0` and the assembled observation is loosely clipped
  to `[-50, 50]` (`clip_obs` in training config). Per-feature standardization is done
  by the trainer's `ObsNormalizer`; `policy_inference_node` applies the checkpoint's
  saved `obs_norm` (standardize by running mean/var, then clamp to `norm_clip = 10`)
  before the actor. A checkpoint with no `obs_norm` runs unnormalized.

### `/rl/action` - `std_msgs/Float32MultiArray`
- `data` length **2**, `[throttle, steering]`, both in `[-1, 1]`.
- Published by `policy_inference_node`; consumed by `drive_command_node` and (for the
  `last action` obs field) `observation_builder_node`.

### `/rl/obs_debug/future_points` - `visualization_msgs/MarkerArray`
- Frame `map`. The future center/left/right points transformed back to world for visualization.

### `/rl/obs_debug/scalars` - `std_msgs/Float32MultiArray`
- `data` length **24**, decoded by `obs_debug_node` directly from `/rl/observation` so every
  field the policy sees can be plotted as a time series. Indices mirror `interfaces.py`
  (`OBS_DEBUG_*`):

| Index | Field | Source (obs slice) |
| --- | --- | --- |
| `0:2` | `lin_vel_x`, `lin_vel_y` | `[0:2]` |
| `2` | `ang_vel_z` | `[2]` |
| `3:5` | `lin_acc_x`, `lin_acc_y` | `[3:5]` |
| `5:7` | `last_throttle`, `last_steer` | `[5:7]` |
| `7:9` | `progress_cos`, `progress_sin` | `[7:9]` |
| `9` | `heading_err` | `[9]` |
| `10` | `lateral_err` | `[10]` |
| `11` | `contact_flag` | `[11]` |
| `12` | `speed` (derived `hypot(vx, vy)`) | `[0:2]` |
| `13` | `min_left_margin` (m, observed corridor ahead) | derived from `[12:372]` |
| `14` | `min_right_margin` (m, observed corridor ahead) | derived from `[12:372]` |
| `15:22` | opponent block (`rel_x, rel_y, rel_vx, rel_vy, gap_norm, lateral, present`) | `[380:387]` |
| `22` | `max_slip_ratio` (max abs of first 4 slip values) | `[372:376]` |
| `23` | `max_slip_angle` (max abs of last 4 slip values) | `[376:380]` |

- The corridor margins are computed in the **ego frame**: the future block is reshaped to
  `(3, 60, 2)` (center / left / right), and per sample `k` the margins are
  `left[k].y - center[k].y` and `center[k].y - right[k].y`; the published value is the
  minimum over all 60 samples. They show how tight the corridor the policy perceives is.
- The opponent fields are zero when the opponent obs is masked or absent (presence flag at
  index 21 is `0.0`).

### `/rl/obs_debug/opponent` - `visualization_msgs/Marker`
- Frame `map`. A sphere at the opponent position reconstructed from the ego-frame relative
  position in the observation, published only when the presence flag is set; otherwise a
  `DELETE` marker. Debug only.

### `/rl/metrics` - `std_msgs/Float32MultiArray`
- `data` length **6**: `[lap_count, last_lap_time_s, max_progress_ratio, lateral_error_m, oob_flag, speed_mps]`.

### `/rl/opponent/odom` - `nav_msgs/Odometry`
- Frame `map`. Published by the real-car `opponent_detector` (LiDAR detection) when a
  confirmed opponent is tracked. `pose.pose.position.{x,y}` is the opponent centroid;
  `twist.twist.linear.{x,y}` is the **world-frame** velocity (≈0 for a stationary
  opponent). Consumed by `vehicle_obs` (real car) / `observation_builder` (sim) to
  fill the `[380:387]` opponent block. Absence is signalled by silence + a consumer
  timeout (`opponent_timeout_s`), not by a separate flag.

### `/rl/opponent/marker` - `visualization_msgs/Marker`
- Frame `map`. A sphere at the detected opponent position for Foxglove (debug only).

## Observation / action constants (from `config.py` DEFAULT_CONFIG)

- `num_obs = 380`, `num_actions = 2`
- `max_speed = 15.0` m/s, `max_steer = 0.44` rad, `clip_actions = 1.0`
- `contact_margin_m = 0.08`
- `future_track_num_points = 60`, `future_track_horizon_s = 6.0`, `future_track_min_lookahead_m = 5.0`
- Track corridor edges use per-point `w_tr_left_m` / `w_tr_right_m` from the centerline CSV (~1.33 m total mean on IV_2026_SIM). The legacy `future_track_width = 2.2` config key is deprecated and unused for boundary geometry.
- hidden layers `[512, 512, 512]`, activation ReLU, `act_limit = 1.0`
