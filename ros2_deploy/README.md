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
via `scripts/build_centerline_from_raceline.py` (671 points, 2.2 m width).

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
