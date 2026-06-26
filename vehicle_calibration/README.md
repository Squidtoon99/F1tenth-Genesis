# vehicle_calibration

Tooling to align Genesis simulation physics with the real F1TENTH car by driving
both through the **same** open-loop maneuver schedule, measuring the response, and
fitting Genesis parameters to match — in the **high-speed racing band only**.

## Why high-speed-first

The real car exhibits low-speed "crunching": jerky crawl, VESC cogging/stiction,
and speed-tracking deadband below ~1–2 m/s. That is a deploy/actuator artifact, not
something Genesis should reproduce, and racing happens at higher speeds. Every
metric here is gated by `v_fit_min` (default 2.5 m/s, in `maneuvers/defaults.yaml`):
samples below it are logged and plotted but never enter the fit objective. The
fitter is not even allowed to touch `c_roll`, stiction, or a speed-tracking lag —
the only knobs that would let it chase low-speed behavior.

## Layout

```
vehicle_calibration/
├── maneuvers/carpet_profile.yaml   # shared maneuver schedule (Genesis + car)
├── maneuvers/defaults.yaml         # v_fit_min, fit weights, search space
├── profile_genesis.py              # in-sim open-loop profiler
├── parse_bag.py                    # rosbag2 -> aligned 10 Hz CSV
├── compare.py                      # sim vs IRL plots + alignment_report.json
├── fit.py                          # parameter search -> fitted_env.json
├── lap_compare.py                  # closed-loop policy-lap validation
├── track_align.py                  # Phase 0 centerline alignment helpers
├── ros/profile_maneuver_node.py    # on-car scripted /rl/action publisher
└── runs/<run_id>/                  # gitignored artifacts (CSV, JSON, plots, bag)
```

## Workflow

All commands run from the repo root via `python -m vehicle_calibration`.

### 0. Track alignment (only needed for policy-lap validation)

```bash
python -m vehicle_calibration track-align check       # on-car gate checklist
python -m vehicle_calibration track-align validate --map map.yaml --reference ref.csv
python -m vehicle_calibration track-align bundle --csv /abs/<TRACK>_centerline.csv
```

Open-loop profiling (steps 1–4) does not need a finished map; run it in any open
area of the carpet in parallel.

### 1. Profile Genesis

```bash
python -m vehicle_calibration profile genesis --run-id carpet_baseline
```

Writes `runs/carpet_baseline/profile_genesis.csv` and `summary_genesis.json`.

### 2. Profile the real car

```bash
python -m vehicle_calibration profile irl --run-id carpet_baseline   # prints how-to
```

On the car, launch the profiler (it owns `/rl/action`; do not run
`policy_inference` at the same time), arm teleop, and record a bag:

```bash
ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py enable_profiler:=true
ros2 bag record -o vehicle_calibration/runs/carpet_baseline/bag \
    /rl/action /drive /odom /pf/pose/odom /rl/observation /calib/maneuver /calib/role
```

Stage `speed_limit_mps` (in `vehicle.yaml`): start at 2.0 for the first checks,
then raise to 5–6 m/s so the ≥3 m/s fit windows reach race speed.

### 3. Parse the bag

```bash
python -m vehicle_calibration parse bag \
    vehicle_calibration/runs/carpet_baseline/bag --run-id carpet_baseline
```

### 4. Compare

```bash
python -m vehicle_calibration compare --run-id carpet_baseline
```

Per-maneuver overlay plots (sub-`v_fit_min` shaded out-of-fit) and a fit-band
diff table in `alignment_report.json`.

### 5. Fit Genesis to the car

```bash
python -m vehicle_calibration fit --run-id carpet_baseline
```

Searches `f_drive_max, dragcoeff, power_max, f_brake_max, tire_friction, t_delta`,
re-checks the winner against `scripts/physics_check.py`, and writes
`runs/carpet_baseline/fitted_env.json`. Merge those values into
`config.py["env"]` (or keep them as a carpet overlay) once validated.

### 6. Policy-lap validation (after Phase 0)

```bash
python -m vehicle_calibration lap-compare profile-genesis --run-id lap01 --ckpt ckpt.pt
python -m vehicle_calibration lap-compare parse-irl       --run-id lap01 lap_bag/
python -m vehicle_calibration lap-compare compare         --run-id lap01
```

Speed and lateral-error agreement is scored only on lap segments at or above
`lap_v_gate` (3 m/s), matching the high-speed-first intent.

## Schema

`parse_bag` and `profile_genesis` emit the identical per-step CSV schema (see
`schema.py`): `t, maneuver, role, throttle, steer, x, y, yaw, vx, vy, speed, ax,
ay, omega_z, steer_state, source`. `steer_state` is the internal lagged steer
angle and is only available in Genesis (NaN on the car).
