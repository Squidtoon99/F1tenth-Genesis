# Docker quickstart: run a trained policy in f1tenth_gym_ros

This guide is for someone who **already has** [dfr_f1tenth_gym](https://github.com/RMin280/dfr_f1tenth_gym/tree/dev-humble) (or equivalent `f1tenth_gym_ros` Docker stack) running and wants to close the loop with a checkpoint from this repo.

**What you get:** the gym simulator drives the car from a trained QRSAC policy (`SquashedGaussianMLPActor`, 380-dim observation). Visualization via noVNC and Foxglove.

**Repos involved:**

| Repo | Role |
| --- | --- |
| `dfr_f1tenth_gym` | Docker sim + `f1tenth_gym_ros` bridge |
| `F1tenth-Genesis` | Trained checkpoints + `ros2_deploy/f1tenth_rl_agent` ROS package |

---

## Prerequisites

- Docker Desktop (or Docker Engine) running
- `dfr_f1tenth_gym` cloned and built at least once (`docker compose up` works)
- A policy checkpoint (`.pt`) from `standalone_trainer.py`, e.g.  
  `outputs/standalone/<run_name>/ckpt_30000.pt`
- **Track / checkpoint must match.** Example: `ckpt_30000` was trained on **Oschersleben**. The default agent config targets **IV 2026** — use the Oschersleben overrides below for that checkpoint.

---

## 1. One-time Docker volume setup

Edit your gym repo’s `docker-compose.yml` (`sim` service `volumes`). Add three mounts (adjust host paths):

```yaml
services:
  sim:
    volumes:
      - .:/sim_ws/src/f1tenth_gym_ros
      # F1tenth-Genesis ROS agent package (Python nodes)
      - /path/to/F1tenth-Genesis/ros2_deploy/f1tenth_rl_agent:/sim_ws/src/f1tenth_rl_agent
      # Track centerlines + map PNGs used by track_server
      - /path/to/F1tenth-Genesis/ros2_deploy/assets:/sim_ws/src/f1tenth_rl_agent/assets
      # Read-only folder of .pt checkpoints inside the container
      - /path/to/F1tenth-Genesis/outputs/standalone/local_longrun_1407:/checkpoints:ro
```

Restart the stack after editing:

```bash
cd /path/to/dfr_f1tenth_gym
docker compose down
docker compose up -d
```

Confirm containers are up:

```bash
docker ps
# expect: dfr_f1tenth_gym-sim-1, dfr_f1tenth_gym-novnc-1 (names may vary)
```

---

## 2. One-time map + sim config

The policy observation is built from a **centerline CSV**. The gym must load the **same map** in the same coordinate frame.

### Option A — IV 2026 competition track (default in this repo)

Inside your gym checkout:

```bash
cd /path/to/dfr_f1tenth_gym
cp /path/to/F1tenth-Genesis/ros2_deploy/assets/IV_2026_SIM.png maps/
cp /path/to/F1tenth-Genesis/ros2_deploy/assets/IV_2026_SIM.yaml maps/
cp /path/to/F1tenth-Genesis/ros2_deploy/assets/IV_2026_SIM_raceline.csv maps/
```

Merge the bridge settings from  
`F1tenth-Genesis/ros2_deploy/assets/sim_iv2026.yaml`  
into `config/sim.yaml` (copy the `bridge` and `foxglove` keys).

Use `config/agent.yaml` when launching the agent (default).

### Option B — Oschersleben (for `ckpt_30000` and other Oschersleben-trained runs)

```bash
cp /path/to/F1tenth-Genesis/ros2_deploy/assets/Oschersleben_map.png maps/
cp /path/to/F1tenth-Genesis/ros2_deploy/assets/Oschersleben_map.yaml maps/
```

Merge `ros2_deploy/assets/sim_oschersleben.yaml` into `config/sim.yaml`.

When launching the agent, pass `agent_oschersleben.yaml` (see step 4).

Rebuild inside the container so installed config picks up `sim.yaml` changes:

```bash
docker exec -it dfr_f1tenth_gym-sim-1 bash -lc '
  source /opt/ros/humble/setup.bash
  source /sim_ws/.venv/bin/activate
  cd /sim_ws && colcon build
'
```

---

## 3. Build the agent package (first time + after code changes)

```bash
docker exec -it dfr_f1tenth_gym-sim-1 bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source /sim_ws/.venv/bin/activate
cd /sim_ws
colcon build --packages-select f1tenth_rl_agent
source install/setup.bash
```

Quick sanity check:

```bash
ros2 pkg list | grep f1tenth_rl_agent
ls /checkpoints/
ls /sim_ws/src/f1tenth_rl_agent/assets/
```

You should see your `.pt` file under `/checkpoints/` and centerline CSVs under `assets/`.

---

## 4. Launch sim + policy

### Recommended: one launch file (if your gym fork includes it)

Many setups add `launch/rl_agent_demo_launch.py` to the gym repo. That starts the sim bridge, RViz, Foxglove, and all five agent nodes together:

```bash
docker exec -it dfr_f1tenth_gym-sim-1 bash /sim_ws/src/f1tenth_gym_ros/scripts/start_rl_demo.sh
```

With a trained checkpoint (Oschersleben example):

```bash
docker exec -it dfr_f1tenth_gym-sim-1 bash -lc '
  source /opt/ros/humble/setup.bash
  source /sim_ws/.venv/bin/activate
  source /sim_ws/install/setup.bash
  ros2 launch f1tenth_gym_ros rl_agent_demo_launch.py \
    checkpoint_path:=/checkpoints/ckpt_30000.pt \
    params_file:=/sim_ws/src/f1tenth_rl_agent/config/agent_oschersleben.yaml
'
```

IV 2026 example (after retraining on that track):

```bash
ros2 launch f1tenth_gym_ros rl_agent_demo_launch.py \
  checkpoint_path:=/checkpoints/ckpt_30000.pt
```

### Alternative: agent only (sim already running)

If the gym bridge is already up in another terminal:

```bash
ros2 launch f1tenth_rl_agent bringup_agent_launch.py \
  checkpoint_path:=/checkpoints/ckpt_30000.pt \
  params_file:=/sim_ws/src/f1tenth_rl_agent/config/agent_oschersleben.yaml
```

Look for this log line:

```text
Loaded policy checkpoint: /checkpoints/ckpt_30000.pt
```

If you see `using random-init actor`, the checkpoint path is wrong or failed to load.

---

## 5. View the run

| Tool | URL |
| --- | --- |
| **noVNC** (RViz in browser) | http://localhost:8080/vnc.html |
| **Foxglove** | https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://localhost:8765 |

Optional: import `F1tenth-Genesis/ros2_deploy/assets/foxglove_layout.json` as a Foxglove layout.

Useful topics:

| Topic | What to check |
| --- | --- |
| `/rl/observation` | 380 floats, no NaNs |
| `/rl/action` | ~10 Hz, values in `[-1, 1]` |
| `/drive` | Non-zero speed / steering when policy runs |
| `/rl/metrics` | `speed`, `progress`, `lateral_error`, lap stats |
| `/ego_racecar/odom` | Car pose updating |

---

## 6. First-run checklist

1. **Track matches checkpoint** — Oschersleben ckpt → `agent_oschersleben.yaml` + Oschersleben map in `sim.yaml`.
2. **Checkpoint mounted** — `ls /checkpoints/` inside the container shows your `.pt`.
3. **Agent built** — `colcon build --packages-select f1tenth_rl_agent` succeeded.
4. **Policy loaded** — log says `Loaded policy checkpoint: ...` (not random-init).
5. **Car moves** — `/drive` publishes; if it instantly hits walls, see troubleshooting.

For an old 30k Oschersleben checkpoint, cap deploy speed to reduce wall crashes until you retrain:

```yaml
# in config/agent_oschersleben.yaml → drive_command.ros__parameters
max_speed: 5.0   # training never exceeded ~6 m/s; default gym cap is 10
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Failed to load checkpoint` | Wrong path or corrupt file | Check `/checkpoints/` mount and filename |
| Car shoots to ~10 m/s then hits wall | Deploy `max_speed: 10` vs training ~3–6 m/s | Lower `max_speed` to 5.0; retrain |
| Immediate off-track / nonsense steering | Wrong track centerline vs sim map | Align `sim.yaml` `map_path` with `track_csv` in agent YAML |
| `Observation length != 380` | Stale agent build | Rebuild `f1tenth_rl_agent` |
| Foxglove won’t connect | Port 8765 in use | Stop previous `rl_agent_demo_launch` (`pkill -f rl_agent_demo_launch`) |
| No `/drive` messages | Agent not running or watchdog timeout | Ensure `policy_inference` and `drive_command` nodes are up |

**Observation frame:** if lateral error looks wrong at standstill, try `twist_in_world_frame: true` in the agent YAML (`observation_builder` section).

**Parity test** (optional, on host venv):

```bash
cd F1tenth-Genesis/ros2_deploy/f1tenth_rl_agent
python -m pytest test/test_obs_parity.py -q
```

---

## 8. Stopping and restarting

```bash
# Inside container: Ctrl+C on the launch terminal

# Or from host:
docker exec dfr_f1tenth_gym-sim-1 pkill -f rl_agent_demo_launch.py

# Full stack restart:
cd /path/to/dfr_f1tenth_gym && docker compose restart
```

---

## Reference: what runs where

```
┌─────────────────────────────────────────────────────────┐
│  Docker container (dfr_f1tenth_gym-sim-1)               │
│                                                         │
│  f1tenth_gym_ros bridge  →  /ego_racecar/odom, /scan    │
│                                                         │
│  f1tenth_rl_agent:                                      │
│    track_server          →  /rl/track/*                 │
│    observation_builder   →  /rl/observation  (380-d)    │
│    policy_inference      →  /rl/action       (reads .pt) │
│    drive_command         →  /drive                      │
│    evaluation            →  /rl/metrics                   │
└─────────────────────────────────────────────────────────┘
         ▲                              │
         │  /checkpoints/*.pt (mount)   ▼
    F1tenth-Genesis                 f1tenth_gym sim
```

More detail: [`README.md`](README.md), [`INTERFACES.md`](INTERFACES.md).

---

## 1v1 opponent obs validation (387-dim)

With the gym 1v1 bridge running and `agent_1v1.yaml` / `agent_1v1_detector.yaml` in the container:

```bash
# GT wiring: observation_builder uses /ego_racecar/opp_odom (tight parity)
docker exec -it dfr_f1tenth_gym-sim-1 bash -lc '
  source /opt/ros/humble/setup.bash && source /sim_ws/.venv/bin/activate &&
  source /sim_ws/install/setup.bash &&
  cd /sim_ws/src/f1tenth_rl_agent/test &&
  python validate_opponent_obs_gym.py --duration-s 20 --min-samples 30 \
    --pos-tol 0.05 --vel-tol 0.5 --opp-reference-topic /ego_racecar/opp_odom'

# Detector path: observation_builder uses /rl/opponent/odom (LiDAR detector)
# Run a single opponent_detector + observation_builder with agent_1v1_detector.yaml first.
python validate_opponent_obs_gym.py --duration-s 25 --min-samples 50 \
  --pos-tol 0.1 --vel-tol 3.0 --opp-reference-topic /rl/opponent/odom
```

Measured on IV_2026_SIM (Jun 2026): GT wiring p95 position error < 0.05 m on all opponent
channels; detector world position vs `/ego_racecar/opp_odom` mean ~0.56 m (p95 ~0.89 m);
detector→obs wiring p95 position error < 0.1 m when referencing `/rl/opponent/odom`.
