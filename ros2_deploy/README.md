# f1tenth_rl_agent - ROS 2 deployment of the trained QRSAC policy

**First time with Docker + a trained checkpoint?** See [`DOCKER_QUICKSTART.md`](DOCKER_QUICKSTART.md).

This folder contains a ROS 2 Humble stack that runs the trained
`SquashedGaussianMLPActor` policy in closed loop against the
[`f1tenth_gym_ros`](https://github.com/RMin280/dfr_f1tenth_gym/tree/dev-humble)
simulator on the **IV 2026 competition** track (`IV_2026_SIM` from
[`dfr_f1tenth_gym` dev-humble](https://github.com/RMin280/dfr_f1tenth_gym/tree/dev-humble)),
visualized in Foxglove.

The policy does **not** consume LiDAR. Its 380-dim observation is built from the
car pose (odometry) projected onto the training track centerline (Frenet frame)
plus sampled future track points. So the obs builder must use the *same* centerline
the policy trained on, and the simulator must load the *same* map, so odometry and
centerline share one coordinate frame. See `INTERFACES.md`.

## Layout

- `INTERFACES.md` - frozen topic/message contract shared by all nodes.
- `f1tenth_rl_agent/` - the ROS 2 ament_python package (5 nodes + obs_core).
- `assets/` - track maps + centerlines (`IV_2026_SIM_*`, legacy `Oschersleben_*`),
  `sim_iv2026.yaml` / `sim_oschersleben.yaml` (bridge config overrides), `foxglove_layout.json`.
- `f1tenth_rl_agent/assets/` - copies installed with the package for Docker mounts.
- `docker/` - `Dockerfile.agent` and `docker-compose.agent.yml`.

## Nodes (see `INTERFACES.md` for full topic contract)

| Node | Subscribes | Publishes |
| --- | --- | --- |
| `track_server` | - | `/rl/track/centerline`, `/rl/track/widths`, `/rl/track/markers` |
| `observation_builder` | `/ego_racecar/odom`, track topics, `/rl/action` | `/rl/observation`, `/rl/obs_debug/future_points` |
| `policy_inference` | `/rl/observation` | `/rl/action` |
| `drive_command` | `/rl/action` | `/drive` |
| `evaluation` | `/ego_racecar/odom`, track topics | `/rl/metrics`, `/initialpose` |

## 1. Bring up the simulator (Docker, macOS / no NVIDIA -> noVNC)

```bash
# sibling workspace, not inside this repo
git clone -b dev-humble https://github.com/RMin280/dfr_f1tenth_gym.git
cd dfr_f1tenth_gym
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
docker build -t f1tenth_gym_ros -f Dockerfile .

# IV 2026 competition map + config (default)
cp /path/to/this/repo/ros2_deploy/assets/IV_2026_SIM.png maps/
cp /path/to/this/repo/ros2_deploy/assets/IV_2026_SIM.yaml maps/
cp /path/to/this/repo/ros2_deploy/assets/IV_2026_SIM_raceline.csv maps/
# merge the values from ros2_deploy/assets/sim_iv2026.yaml into config/sim.yaml

# Legacy Oschersleben (ckpt trained on this track): use sim_oschersleben.yaml and
# agent_oschersleben.yaml instead of the defaults above.

docker compose up        # starts `sim` + `novnc`
```

- noVNC GUI: http://localhost:8080/vnc.html
- Foxglove: https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://localhost:8765
  (import `assets/foxglove_layout.json` as the layout)

## 2. Run the agent

### Option A (recommended): inside the sim container

Mount this package into the bridge repo before `docker compose up`, e.g. add to the
bridge's `docker-compose.yml` `sim` service volumes:

```yaml
      - /path/to/this/repo/ros2_deploy/f1tenth_rl_agent:/sim_ws/src/f1tenth_rl_agent
      - /path/to/checkpoints:/checkpoints:ro
```

Then exec into the sim container and launch:

```bash
docker exec -it dfr_f1tenth_gym-sim-1 /bin/bash
source /opt/ros/humble/setup.bash
source /sim_ws/.venv/bin/activate
cd /sim_ws
colcon build --packages-select f1tenth_rl_agent
source install/setup.bash
ros2 launch f1tenth_rl_agent bringup_agent_launch.py \
    checkpoint_path:=/checkpoints/policy.pt
```

### Option B: separate agent container

```bash
cd ros2_deploy/docker
SIM_NETWORK=dfr_f1tenth_gym_x11 CHECKPOINT_DIR=/abs/path/to/ckpts \
  docker compose -f docker-compose.agent.yml up --build
```

(`SIM_NETWORK` must match the bridge compose project's `x11` network name; check with
`docker network ls`.)

## Track selection

| Track | Genesis training | Gym sim config | ROS agent config |
| --- | --- | --- | --- |
| **IV 2026** (default) | `config.py` → `"track": "IV_2026_SIM"` or `--track IV_2026_SIM` | `sim_iv2026.yaml` | `config/agent.yaml` |
| Oschersleben (legacy) | `--track Oschersleben` | `sim_oschersleben.yaml` | `config/agent_oschersleben.yaml` |

Centerline CSVs are derived from `IV_2026_SIM_smooth.csv` in
[dfr_f1tenth_gym dev-humble](https://github.com/RMin280/dfr_f1tenth_gym/tree/dev-humble)
via `scripts/build_centerline_from_raceline.py` (671 points). Corridor half-widths
are **per vertex** from CSV columns `w_tr_left_m` / `w_tr_right_m` (~0.65–0.68 m
each side, ~1.33 m total mean width — not a uniform 2.2 m band).

**Note:** `ckpt_30000.pt` was trained on Oschersleben — retrain on IV 2026 before
expecting competitive lap times on the new map.

To launch the agent on Oschersleben with an old checkpoint:

```bash
ros2 launch f1tenth_rl_agent bringup_agent_launch.py \
  params_file:=/sim_ws/src/f1tenth_rl_agent/config/agent_oschersleben.yaml \
  checkpoint_path:=/checkpoints/ckpt_30000.pt
```

## 3. Checkpoint

`policy_inference` loads a local `.pt` (mounted at `/checkpoints/policy.pt` by default).
It accepts either a raw actor `state_dict` or the `standalone_trainer` checkpoint format
`{"step", "actor", ...}` (loads the `actor` key). With no checkpoint it falls back to a
random-init actor so the rest of the pipeline can still be exercised.

## 4. Validation

- Unit + parity tests (run in the venv or the container):

```bash
cd ros2_deploy/f1tenth_rl_agent
python -m pytest test -q          # pure tests + (in-container) rclpy tests
```

  `test/test_obs_parity.py` asserts `obs_core` matches the real
  `f1tenth_env.build_observation` within `1e-4` - this guarantees the deployed
  observation equals the one the policy trained on.

- Closed-loop success criteria (watch `/rl/metrics` + Foxglove):
  - `/rl/observation` length 380 and finite, `/rl/action` in `[-1, 1]` at ~10 Hz,
    `/drive` populated.
  - The car completes >= 1 full lap without going out of bounds.
  - `lateral_error` stays within the track half-width; `max_progress` increases
    monotonically per lap; `last_lap_time` becomes finite.

- If the car drives off immediately, first re-check obs parity and the
  `twist_in_world_frame` assumption (set it `true` in `config/agent.yaml` if the
  simulator reports odom twist in the world frame).

## Notes / assumptions

- Pose source is the sim ground-truth `/ego_racecar/odom` (no localization node).
- `drive_command` maps negative throttle to a stop (`brake_behavior: stop`); set
  `reverse` to allow reverse.
- Control loop runs at 10 Hz to match the training `control_interval`.

---

# Real-car deployment (`f1tenth_rl_vehicle`)

The sim stack above is for `f1tenth_gym_ros`. For a physical F1TENTH car running the
standard `f1tenth_stack` (particle filter + VESC + `ackermann_mux`), use the lean C++
package [`f1tenth_rl_vehicle/`](f1tenth_rl_vehicle/). It runs **three** nodes on the
car instead of the five sim nodes:

| Node | Package | Lang | Role |
| --- | --- | --- | --- |
| `vehicle_obs` | `f1tenth_rl_vehicle` | C++ | particle-filter pose + VESC twist + track CSV -> `/rl/observation` @ fixed 10 Hz |
| `policy_inference` | `f1tenth_rl_agent` | Python | `/rl/observation` -> `/rl/action` (applies the checkpoint's `obs_norm`) |
| `drive` | `f1tenth_rl_vehicle` | C++ | `/rl/action` -> `/drive` (Ackermann) with a stop watchdog |
| `opponent_detector` | `f1tenth_rl_vehicle` | C++ | `/scan` + ego pose -> `/rl/opponent/odom` (1v1 only, `enable_opponent:=true`) |
| `obs_debug` | `f1tenth_rl_agent` | Python | `/rl/observation` -> `/rl/obs_debug/scalars` + markers (read-only diagnostics, `enable_obs_debug:=true`, default on) |

`vehicle_obs` merges the sim `track_server` + `observation_builder` (it loads the
training centerline CSV directly), so there is no `/rl/track/*` plumbing on the car.
Tyre-slip obs dims `[372:380]` are published as zeros (no per-wheel sensing), matching
the gym deploy. `evaluation` is sim-only (`/initialpose` teleport) and is not run.

## Build

```bash
# in your f1tenth_stack colcon workspace
colcon build --packages-select f1tenth_rl_agent f1tenth_rl_vehicle
source install/setup.bash
```

The C++ parity test (`test_rl_obs_core`) compares `rl_obs_core` against a fixture
generated from the Python pipeline (`test/obs_fixture.txt`), guaranteeing the on-car
observation equals the one the policy trained on within `1e-4`.

## Diagnosing wall collisions (obs_debug)

`bringup_vehicle.launch.py` runs the read-only `obs_debug` node by default
(`enable_obs_debug:=false` to disable). It decodes the exact `/rl/observation` the
policy sees and republishes it for plotting and 3D overlay, without touching the
drive command:

- `/rl/obs_debug/scalars` (`Float32MultiArray`, 24 fields) -- lateral error, contact
  flag, speed, last action, and the **minimum observed corridor margins** ahead
  (see `INTERFACES.md` for the index table).
- `/rl/obs_debug/future_points` -- the center/left/right future track curves the
  policy sees, transformed into the `map` frame.
- `/rl/obs_debug/opponent` -- a sphere at the opponent position reconstructed from the
  ego-frame relative offset (only when the presence flag is set).

Open `assets/foxglove_layout.json` in Foxglove (connected to the car's ROS bridge)
and watch the `obs_debug` plot when the car hits a wall:

- **`contact_flag`** rising to 1 before impact means the policy already perceives the
  wall (a control/timing problem, not a perception one).
- **`min_left_margin` / `min_right_margin`** near zero while steering is saturated means
  the policy is driving outside the corridor it perceives.
- **`lateral_err`** diverging while the margins stay healthy points at a localization or
  track-frame mismatch -- confirm by checking that `/rl/obs_debug/future_points` overlays
  the real track in the 3D panel. Misaligned future points are the most common root cause.

## 1v1 opponent detection

For head-to-head racing the car can detect the opponent from its LiDAR and feed the
7-dim opponent block into the observation (the obs grows from 380 to **387** dims,
matching a checkpoint trained with `enable_opponent_obs=true`). Enable it at launch:

```bash
ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \
    checkpoint_path:=/abs/path/to/policy_1v1.pt \
    enable_opponent:=true
```

This starts the extra `opponent_detector` node and sets `enable_opponent_obs` on both
`vehicle_obs` and `policy_inference`. The detector:

- clusters `/scan` returns, transforms them to the `map` frame using the ego pose and
  a **static laser-mount offset** (`lidar_offset_{x,y,yaw}` in `config/vehicle.yaml`;
  set these from your URDF -- no tf2 dependency), then
- gates clusters to the drivable corridor (centerline + width CSV) to reject walls, and
- tracks the surviving cluster across frames to estimate a world-frame velocity,
  publishing the opponent as `nav_msgs/Odometry` on `/rl/opponent/odom`.

Confirmation is persistence-based, so a **stationary opponent is still detected**
(its reported velocity is ~0); motion is only used to fast-track confirmation, never
to reject a slow/static track. The corridor gate assumes the drivable band is
otherwise clear (no static cones/debris inside it). Tune the gating thresholds
(`cluster_gap_m`, `min/max_opponent_size_m`, `boundary_margin_m`, ...) in
`config/vehicle.yaml`.

The opponent obs `[380:387]` is parity-checked against the training
`obs_opponent`. Regenerate the fixture (in the venv/container with torch) when the
observation math changes:

```bash
cd ros2_deploy/f1tenth_rl_vehicle/test
PYTHONPATH=../../f1tenth_rl_agent python gen_obs_fixture.py \
    ../../f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv obs_fixture.txt
```

### Validation in `f1tenth_gym_ros`

The detector was validated against the live two-car `f1tenth_gym_ros` sim. The gym
renders the opponent into the ego LiDAR, so it can be detected from `/scan` alone.
Copy the package into the sim container, build, run the unit suite, then run the
detector against the bridge topics (`/scan` + ground-truth ego pose `/ego_racecar/odom`)
and compare `/rl/opponent/odom` to the ground-truth opponent pose `/opp_racecar/odom`:

```bash
# from ros2_deploy/, with the gym sim container running:
docker cp f1tenth_rl_vehicle <sim_container>:/sim_ws/src/f1tenth_rl_vehicle
docker exec <sim_container> bash -lc '
  source /opt/ros/humble/setup.bash && cd /sim_ws &&
  colcon build --packages-select f1tenth_rl_vehicle &&
  colcon test --packages-select f1tenth_rl_vehicle --event-handlers console_direct+'

# run the detector against the live sim:
docker exec <sim_container> bash -lc '
  source /opt/ros/humble/setup.bash && source /sim_ws/install/setup.bash &&
  ros2 run f1tenth_rl_vehicle opponent_detector --ros-args \
    -p track_csv:=/sim_ws/src/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv \
    -p scan_topic:=/scan -p pose_topic:=/ego_racecar/odom'
# then: ros2 topic echo /rl/opponent/odom   (vs /opp_racecar/odom)
```

Measured against ground truth over a multi-lap race:

- **Visible opponent** (in the front FoV, unoccluded): ~95% recall, ~0.5 m mean
  position error.
- **Occluded / behind** (around a corner, nothing valid in view): ~15%
  false-positive rate after the corridor + foreground + range gates (down from ~55%
  with the corridor gate alone). The consumer's `opponent_timeout_s` and the obs
  presence flag absorb these brief spurious detections.

## Launch

```bash
ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \
    checkpoint_path:=/abs/path/to/policy.pt
```

Useful launch arguments:

- `track_csv:=` an absolute centerline path (defaults to the installed
  `IV_2026_SIM_centerline.csv`). For a real track this must be **surveyed in the same
  frame as the localization map**.
- `params_file:=` overrides [`config/vehicle.yaml`](f1tenth_rl_vehicle/config/vehicle.yaml)
  (topics, `speed_limit_mps`, `twist_in_world_frame`, ...).

Verify the topic names against your install (defaults assume `/pf/pose/odom` for
map-frame pose and `/odom` for VESC body twist).

## Map alignment

The policy's Frenet features (progress, lateral error, look-ahead points) are only
correct if the localization `map` frame matches the centerline CSV coordinates.

1. Build the centerline from a surveyed raceline with
   [`scripts/build_centerline_from_raceline.py`](../scripts/build_centerline_from_raceline.py).
2. Park the car at a known start pose and confirm in Foxglove that the lateral error
   obs (`/rl/observation[10]`) is near zero before sending any throttle.

## First-run checklist (staged shakedown)

The only safety layer is the `drive` watchdog plus the f1tenth_stack `ackermann_mux`
(teleop always overrides the autonomous `/drive` input). Bring speed up in stages:

1. Stationary: `/rl/observation` is length 380 and finite; lateral error ~0 at the
   known start pose.
2. Low speed: keep `speed_limit_mps: 2.0` (the `vehicle.yaml` default) and confirm
   teleop override on the mux cuts the car instantly.
3. Full speed: raise `speed_limit_mps` toward the training `max_speed` (15.0) once the
   low-speed lap is clean.

Record a bag (`ros2 bag record /rl/observation /rl/action /drive /pf/pose/odom /odom`)
for post-run debugging. If the car drives off immediately, re-check the map alignment
and the `twist_in_world_frame` setting first.
