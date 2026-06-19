# f1tenth_rl_agent - ROS 2 deployment of the trained QRSAC policy

This folder contains a ROS 2 Humble stack that runs the trained
`SquashedGaussianMLPActor` policy in closed loop against the
[`f1tenth_gym_ros`](https://github.com/RMin280/dfr_f1tenth_gym/tree/dev-humble)
simulator on the **Oschersleben** track, visualized in Foxglove.

The policy does **not** consume LiDAR. Its 372-dim observation is built from the
car pose (odometry) projected onto the training track centerline (Frenet frame)
plus sampled future track points. So the obs builder must use the *same* Oschersleben
centerline the policy trained on, and the simulator must load the *same* Oschersleben
map, so odometry and centerline share one coordinate frame. See `INTERFACES.md`.

## Layout

- `INTERFACES.md` - frozen topic/message contract shared by all nodes.
- `f1tenth_rl_agent/` - the ROS 2 ament_python package (5 nodes + obs_core).
- `assets/` - Oschersleben map (`.png`/`.yaml`) + `Oschersleben_centerline.csv`,
  `sim_oschersleben.yaml` (bridge config overrides), `foxglove_layout.json`.
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

# Oschersleben map + config
cp /path/to/this/repo/ros2_deploy/assets/Oschersleben_map.png maps/
cp /path/to/this/repo/ros2_deploy/assets/Oschersleben_map.yaml maps/
# merge the values from ros2_deploy/assets/sim_oschersleben.yaml into config/sim.yaml

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
  - `/rl/observation` length 372 and finite, `/rl/action` in `[-1, 1]` at ~10 Hz,
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
