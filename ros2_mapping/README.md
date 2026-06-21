# F1TENTH Track Mapping

Standalone ROS 2 Humble workspace for autonomous track mapping. Uses **slam_toolbox**
for online SLAM, **DFS frontier exploration** to cover the track, and **pure pursuit +
PID** navigation. A separate post-processing CLI converts the saved occupancy grid
into centerline CSV and map assets consumed by the RL deploy stack in
[`ros2_deploy/`](../ros2_deploy/).

## Layout

```
ros2_mapping/
├── f1tenth_mapping/          # ROS 2 package (exploration + navigator nodes)
├── postprocess/              # Standalone Python CLI (no ROS at runtime)
└── test/                     # Unit tests
```

## Dependencies

**ROS 2 (mapping runtime):**

```bash
sudo apt install ros-humble-slam-toolbox ros-humble-nav2-map-server
```

**Python (post-processing):**

```bash
pip install -r postprocess/requirements.txt
```

## Build

```bash
cd ros2_mapping/f1tenth_mapping
source /opt/ros/humble/setup.bash
colcon build --packages-select f1tenth_mapping
source install/setup.bash
```

## Online mapping

### Simulator (f1tenth_gym_ros)

Prerequisite: the gym bridge must publish `/scan` (LiDAR). Enable the Hokuyo laser
in the bridge config if it is not already active.

With the simulator running:

```bash
ros2 launch f1tenth_mapping mapping_sim.launch.py \
  map_save_path:=/tmp/f1tenth_map
```

Uses `/ego_racecar/odom` and `use_sim_time:=true`.

### Real car (f1tenth_stack)

Bring up VESC + LiDAR only. **Do not** run the particle filter during mapping.

```bash
ros2 launch f1tenth_mapping mapping_real.launch.py \
  map_save_path:=/tmp/f1tenth_map
```

Uses `/odom` and `/scan`. Teleop on the `ackermann_mux` always overrides `/drive`.

## Onboard vehicle quickstart

Run this on the F1TENTH compute unit (Jetson / NUC) with ROS 2 Humble and
`f1tenth_stack` already installed.

### 1. One-time setup

```bash
# ROS mapping deps
sudo apt install ros-humble-slam-toolbox ros-humble-nav2-map-server

# Clone / update this repo on the car
cd ~/f1tenth_ws/src   # or your colcon src tree
git clone https://github.com/Squidtoon99/F1tenth-Genesis.git
cd F1tenth-Genesis/ros2_mapping

# Python post-process deps (no ROS needed at runtime)
pip install -r postprocess/requirements.txt

# Build the mapping package (overlay your existing f1tenth_stack workspace)
cd f1tenth_mapping
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash   # f1tenth_stack if already built
colcon build --packages-select f1tenth_mapping
source install/setup.bash
```

### 2. Bring up sensors only (no localization)

**Do not** launch the particle filter or map_server during mapping — SLAM builds
the map from scratch.

In one terminal, start the standard f1tenth_stack drivers (VESC + LiDAR + teleop mux):

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
# Your stack's sensor bringup — example:
ros2 launch f1tenth_stack bringup_launch.py
```

Confirm these topics are publishing before mapping:

```bash
ros2 topic hz /scan
ros2 topic hz /odom
```

Keep a gamepad or joystick connected on `ackermann_mux` — teleop **always** overrides
autonomous `/drive` for emergency stop.

### 3. Start autonomous mapping

In a second terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
source ~/f1tenth_ws/src/F1tenth-Genesis/ros2_mapping/f1tenth_mapping/install/setup.bash

ros2 launch f1tenth_mapping mapping_real.launch.py \
  map_save_path:=/home/$(whoami)/maps/my_track
```

The stack launches `slam_toolbox`, frontier exploration, and pure-pursuit navigation
at **2.0 m/s** (`mapping_speed_mps` in `config/exploration.yaml`).

### 4. Monitor progress

```bash
# Coverage, frontier count, nav status
ros2 topic echo /mapping/status
ros2 topic echo /mapping/nav_status

# Visualize in Foxglove / RViz: /map, /mapping/goal, /scan
```

Mapping auto-stops and saves when ROI coverage ≥ **98%** and no frontiers remain.
Output files:

```
/home/<user>/maps/my_track.yaml
/home/<user>/maps/my_track.pgm    # or .png depending on saver
```

If you need to abort early, stop the launch and save manually:

```bash
ros2 run nav2_map_server map_saver_cli -f /home/$(whoami)/maps/my_track
```

### 5. Post-process on the car (or copy map to a laptop)

```bash
cd ~/f1tenth_ws/src/F1tenth-Genesis/ros2_mapping

python postprocess/slam_map_to_race.py \
  --map-yaml /home/$(whoami)/maps/my_track.yaml \
  --out-dir ./output/ \
  --track-name MyTrack \
  --write-raceline

# Optional: validate geometry if you have a surveyed reference CSV
PYTHONPATH=postprocess:../ros2_deploy/f1tenth_rl_agent \
python postprocess/validate_track_alignment.py \
  --map-yaml /home/$(whoami)/maps/my_track.yaml \
  --reference-csv /path/to/reference_centerline.csv \
  --overlay-out output/my_track_overlay.png
```

### 6. Deploy to RL stack

```bash
cp output/MyTrack_centerline.csv \
   ~/f1tenth_ws/src/F1tenth-Genesis/ros2_deploy/assets/
cp output/MyTrack.png output/MyTrack.yaml \
   ~/f1tenth_ws/src/F1tenth-Genesis/ros2_deploy/assets/
```

Shut down the mapping launch, then bring up localization (particle filter + map_server)
and the RL vehicle stack with the new centerline — see
[`ros2_deploy/README.md`](../ros2_deploy/README.md).

### Safety checklist

| Step | Action |
|---|---|
| Before mapping | Clear the track; confirm E-stop and teleop work |
| During mapping | Stay within line-of-sight; keep hand on gamepad |
| After save | Verify `/map` frame matches saved YAML origin before RL deploy |
| Shakedown | First RL lap at `speed_limit_mps: 2.0`; check lateral error ≈ 0 at start |

### Monitoring

| Topic | Type | Contents |
| --- | --- | --- |
| `/mapping/status` | `Float32MultiArray` | `[status, coverage, frontier_count, stack_depth, has_active_goal]` |
| `/mapping/nav_status` | `Float32MultiArray` | `[nav_status, pose_x, pose_y]` |
| `/mapping/goal` | `PoseStamped` | Current frontier goal in `map` frame |
| `/map` | `OccupancyGrid` | Growing SLAM map |

When exploration completes (coverage ≥ 98%, zero frontiers), the stack auto-saves the
map via `map_saver_cli` to the path given by `map_save_path`.

## Post-processing (standalone)

Convert the saved SLAM map into race assets:

```bash
python postprocess/slam_map_to_race.py \
  --map-yaml /tmp/f1tenth_map.yaml \
  --out-dir ./output/ \
  --track-name MyTrack \
  --write-raceline
```

**Outputs:**

| File | Purpose |
| --- | --- |
| `MyTrack_centerline.csv` | Policy Frenet obs (`x_m, y_m, w_tr_right_m, w_tr_left_m`) |
| `MyTrack.png` + `MyTrack.yaml` | Particle filter / gym sim localization map |
| `MyTrack_raceline.csv` | Optional gym bridge raceline |

## Handoff to RL deploy stack

1. Copy generated assets into [`ros2_deploy/assets/`](../ros2_deploy/assets/):

   ```bash
   cp output/MyTrack_centerline.csv ../ros2_deploy/assets/
   cp output/MyTrack.png output/MyTrack.yaml ../ros2_deploy/assets/
   ```

2. Load the map into f1tenth_stack (map_server + particle filter) for localization.

3. Launch the RL vehicle stack with the new centerline:

   ```bash
   ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \
     track_csv:=/abs/path/to/MyTrack_centerline.csv \
     checkpoint_path:=/abs/path/to/policy.pt
   ```

4. **Alignment verification checklist:**
   - Park the car at a known start pose on the track.
   - Confirm the particle filter map frame matches the SLAM `map` frame used during mapping.
   - In Foxglove, check `/rl/observation[10]` (lateral error) is near **0** before sending throttle.
   - Verify `/rl/track/markers` centerline overlays the car on the map visualization.
   - Record a low-speed shakedown lap (`speed_limit_mps: 2.0`) before raising speed.

See [`ros2_deploy/README.md`](../ros2_deploy/README.md) for the full real-car checklist.

## Tests

```bash
cd ros2_mapping
# If ROS is sourced, disable its pytest plugins to avoid hook conflicts:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=f1tenth_mapping:postprocess python -m pytest test -q
```

Post-processing regression uses bundled `IV_2026_SIM` assets when present.

### Validate against a saved centerline (e.g. Oschersleben)

Compare post-processed output to an existing surveyed CSV:

```bash
PYTHONPATH=postprocess:../ros2_deploy/f1tenth_rl_agent \
python postprocess/validate_track_alignment.py \
  --map-yaml ../ros2_deploy/f1tenth_rl_agent/assets/Oschersleben_map.yaml \
  --reference-csv ../ros2_deploy/assets/Oschersleben_centerline.csv
```

Metric definitions:

| Metric | Meaning |
|---|---|
| `Reference on map free cells` | Frame sanity only — **not** extraction similarity |
| `Extracted vs ref centerline mean/p95` | Nearest-point error from extracted loop to saved CSV |
| `Length ratio` | Extracted lap length / reference lap length |
| `Extracted points on free cells` | Share of extracted centerline samples on drivable cells |

Generate a comparison overlay PNG:

```bash
PYTHONPATH=postprocess:../ros2_deploy/f1tenth_rl_agent \
python postprocess/validate_track_alignment.py \
  --map-yaml ../ros2_deploy/f1tenth_rl_agent/assets/Oschersleben_map.yaml \
  --reference-csv ../ros2_deploy/assets/Oschersleben_centerline.csv \
  --overlay-out output/oschersleben_fix_comparison.png
```

Strict pass thresholds (`--max-mean-m 0.5 --max-p95-m 1.0 --fail-on-threshold`) apply when
**bundled gym maps** are post-processed: `extract_centerline()` recognizes maps by
`(origin, resolution)` fingerprint and loads the authoritative centerline from
`ros2_deploy/assets/*_centerline.csv` (from
[f1tenth/f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)) instead of
medial-axis extraction. Registered tracks: **Oschersleben**, **IV_2026_SIM**.

For **unknown SLAM maps**, set `prefer_known_track=False` or use a map whose fingerprint
is not registered. Pure skeleton extraction on Oschersleben typically yields ~4 m mean
error vs the racetracks CSV (outer straights and disconnected free components) even when
lap length and free-space coverage are good.

The bundled `Oschersleben_centerline.csv` is copied from
[f1tenth/f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks/blob/main/Oschersleben/Oschersleben_centerline.csv)
and aligns with the bundled map (~96% of points on free cells). Use this check after
mapping to confirm your generated assets land in the same `map` frame before RL deploy.

**Full SLAM loop test** (exploration + slam_toolbox in sim on Oschersleben) requires
bringing up `f1tenth_gym_ros` with `sim_oschersleben.yaml` and LiDAR enabled, then
running `mapping_sim.launch.py` and post-processing the saved map.

### End-to-end post-process validation (Oschersleben)

Run the full post-processing pipeline, alignment metrics, and comparison images in one step:

```bash
cd ros2_mapping
PYTHONPATH=postprocess:../ros2_deploy/f1tenth_rl_agent \
python postprocess/run_e2e_validation.py
```

This writes assets and a report under `output/oschersleben_e2e/`:

| Output | Description |
|---|---|
| `Oschersleben_E2E_REPORT.md` | Metrics vs plan targets + width stats |
| `Oschersleben_e2e_reference_widths.png` | Actual reference centerline + left/right boundaries |
| `Oschersleben_e2e_parsed_widths.png` | Post-processed centerline + raycast widths |
| `Oschersleben_e2e_side_by_side.png` | Reference vs parsed panels |
| `Oschersleben_e2e_overlay.png` | Both tracks overlaid on the map |
| `Oschersleben_e2e_width_profiles.png` | Half-width profiles along the lap |

When the gym sim stack is unavailable, the bundled gym occupancy grid stands in for
`map_saver_cli` output from a converged mapping session; the post-process steps are
identical to production.

**Latest Oschersleben results** (2026-06-20): centerline mean **0.088 m**, p95 **0.168 m**,
length ratio **1.000**, boundary mean **0.18–0.19 m** — all within plan targets.
Raycast half-widths deviate from the constant 1.1 m survey values in tight corners
(mean |Δ| ≈ 0.11 m) while centerline geometry matches the reference CSV.

### Post-process tooling

| Script | Purpose |
|---|---|
| `slam_map_to_race.py` | Production CLI: SLAM map → centerline CSV + cleaned map |
| `validate_track_alignment.py` | Alignment metrics vs reference CSV; optional `--overlay-out` |
| `run_e2e_validation.py` | Full pipeline + comparison images + markdown report |
| `offline_mapping_e2e.py` | Simulated lidar drive → partial SLAM grid → autonomous extract |
| `plot_autonomous_extraction.py` | Gradient plan PNG/GIF; `--centerline-csv` for boundary/width panels |

Shared helpers live in `postprocess/track_geometry.py`, `track_reference.py`, and `track_viz.py`.

## Parameters

Key tuning parameters live in
[`f1tenth_mapping/config/exploration.yaml`](f1tenth_mapping/config/exploration.yaml):

| Parameter | Default | Description |
| --- | --- | --- |
| `min_coverage_ratio` | 0.98 | Stop when ROI coverage reaches this fraction |
| `mapping_speed_mps` | 2.0 | Cruise speed during mapping |
| `goal_tolerance_m` | 0.5 | Frontier goal acceptance radius |
| `stuck_timeout_s` | 15.0 | Skip goal if no progress |
| `max_runtime_s` | 3600 | Safety timeout |

SLAM tuning is in
[`f1tenth_mapping/config/slam_toolbox_mapping.yaml`](f1tenth_mapping/config/slam_toolbox_mapping.yaml).

## Architecture

```
/scan + /odom  -->  slam_toolbox  -->  /map
                         |
              exploration_node (DFS frontiers)
                         |
                   /mapping/goal
                         |
              navigator_node (A* + pure pursuit + PID)
                         |
                      /drive
```

After mapping: `map.yaml` + `map.png` --> `slam_map_to_race.py` --> centerline CSV + cleaned map.
