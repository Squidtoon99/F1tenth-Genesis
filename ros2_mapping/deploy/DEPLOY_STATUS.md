# Real-World 1v1 Deploy Status (2026-06-22)

Staging complete except **GPU torch install** (background worker) and **operator on-track placement**.

## Summary

| Item | Status | Notes |
|------|--------|-------|
| 1. Centerline on car | **Done** | `~/maps/f1tenth_map_centerline.csv` (823 pts), synced from worktree |
| 1b. Map-frame alignment | **Acceptable** | Overlay 92.5% free pixels; **no transform applied** (same `map` origin as `f1tenth_map.yaml`) |
| 2. Localization | **Staged** | `iv2026_localize_launch.py` + `f1tenth_map.yaml`; **must run `set_initial_pose.sh` after each PF restart** |
| 3. Opponent detector | **Defaults set** | Tuned params in `rl_real.yaml`; **no opponent car** → sporadic false positives on `/rl/opponent/odom`; fallback documented |
| 4. Obs parity (387 + obs_norm) | **Done** | `vehicle_obs` publishes **387-dim @ 10 Hz**; `policy_inference` applies `obs_norm`; **CPU torch 2.2.2** installed, inference **4.6 ms/step (~216 Hz)** |
| 5. Bringup staging | **Done** | `~/deploy/run_go_live.sh` + params in `rl_real.yaml` |
| 6. Dry-run topics | **Partial** | Sensors + obs OK; `/rl/action` absent until policy runs |

---

## 1. Centerline + map alignment

- **Files:** `~/maps/f1tenth_map.{pgm,yaml}`, `f1tenth_map_centerline.csv`, `f1tenth_map_overlay.png`
- **Map origin:** `(28.0, -11.5)`, resolution `0.05`
- **Centerline extent:** x ∈ [35.0, 55.1], y ∈ [-7.4, 14.8] m (map frame)
- **Verify:** `python3 ~/deploy/verify_centerline_align.py ~/maps` → 100% in bounds, 92.5% on free pixels (WARN: some points on wall pixels — acceptable for narrow corridor)
- **Transform applied:** none (CSV extracted in same frame as saved SLAM map)

**Blocker for racing:** PF must be seeded at a centerline point. Without `set_initial_pose.sh`, PF drifted to ~(0, -3) m (**36 m** from track). After initial pose: PF ≈ **(54.9, -7.36)**, **0.36 m** from centerline.

---

## 2. Localization

**Launch:**
```bash
bash ~/deploy/run_sensors.sh          # urg + vesc + mux + joy deadman
bash ~/deploy/run_localization.sh     # map_server + particle_filter
bash ~/deploy/set_initial_pose.sh     # REQUIRED — publishes /initialpose at CSV index 0
```

**Topics (when healthy):**
| Topic | Rate | Frame |
|-------|------|-------|
| `/scan` | ~40 Hz | `laser` |
| `/odom` | ~50 Hz | body twist from VESC |
| `/pf/pose/odom` | ~12–38 Hz | map-frame pose (after scan+odom+map) |
| `/map` | latched | occupancy grid |

**Check:** `python3 ~/deploy/check_alignment.py` — target `|obs[10]| < 0.3` before live.

**Note:** Do **not** run second `particle_filter localize_launch.py` (conflicts with iv2026 stack). Duplicate was killed surgically (PID-only, no broad pkill).

---

## 3. Opponent detector

**Params** (`rl_real.yaml`): `lidar_offset_x=0.27`, `max_range_m=8`, `min_track_age_frames=4`, corridor gating on real centerline.

**Without opponent car:** detector may publish **false positives** on `/rl/opponent/odom` (wall/occlusion). Consumer timeout `0.5 s` zeros opponent block `[380:387]`.

**Fallbacks:**
| Mode | Command | When |
|------|---------|------|
| 1v1 zero block | keep `enable_opponent:=true`, no `/rl/opponent/odom` | opponent absent >0.5 s |
| 1v0 solo | `enable_opponent:=false` + `ckpt_50000.pt` (380-dim) | detector not trusted |

---

## 4. Observation parity

- **`vehicle_obs`:** 387-dim vector @ **10.0 Hz** (380 base + 7 opponent block)
- **Tyre slip `[372:380]`:** zeros (matches gym deploy)
- **`policy_inference`:** loads `obs_norm` mean/var `[387]`, `norm_clip=10`, `norm_eps=1e-8`
- **torch:** CPU build `torch==2.2.2` (aarch64 wheel), `torch.cuda.is_available()==False`; standalone actor-MLP forward = **4.6 ms** on 6 CPU threads

**GPU-free dry-run:**
```bash
bash ~/deploy/run_obs_dryrun.sh   # vehicle_obs + opponent_detector only
```

---

## 5. Staged bringup commands

**Full stack dry-run (no motion):**
```bash
bash ~/deploy/run_sensors.sh
bash ~/deploy/run_localization.sh && sleep 5 && bash ~/deploy/set_initial_pose.sh
DRY_RUN=1 AGENT_PARAMS=~/deploy/agent_1v1_detector.yaml bash ~/deploy/run_rl_stack.sh
# drive speed_limit_mps=0; policy will fail until torch ready
```

**Go live (CPU torch; GPU deferred):**
```bash
LIVE_SPEED_LIMIT=2.0 LIVE_MIN_SPEED=1.0 bash ~/deploy/run_go_live.sh   # DEVICE defaults to cpu
```

**Params wired in `rl_real.yaml`:**
- `checkpoint_path: ~/checkpoints/ckpt_500000.pt`
- `enable_opponent_obs: true`
- `speed_limit_mps: 2.0`, `min_speed_mps: 1.0` (VESC deadband)
- Mux: teleop **priority 100** (`/teleop`, deadman **button 5**), RL `/drive` **priority 10**

---

## 6. Topic dry-run checklist

| Topic | Expected | Observed (latest staging) |
|-------|----------|---------------------------|
| `/scan` | ~40 Hz | OK |
| `/odom` | ~50 Hz | OK |
| `/pf/pose/odom` | >10 Hz after seed | OK after sensors+initialpose |
| `/map` | latched | map_server active |
| `/rl/observation` | 10 Hz, len 387 | OK |
| `/rl/opponent/odom` | sporadic if no opponent | optional / false positives |
| `/rl/action` | 10 Hz | **waiting on policy_inference** |
| `/drive` | 10 Hz when policy runs | dry-run: speed_limit=0 |

---

## Single remaining step to go live

1. **torch ready (CPU):** `python3 -c "import torch; print(torch.__version__)"` → `2.2.2` (GPU deferred)
2. **Operator:** place car on track (or confirm PF pose + `check_alignment.py` shows `|ey| < 0.5`)
3. **Run:** `bash ~/deploy/run_go_live.sh`  (DEVICE defaults to `cpu`)
4. **Hold deadman (button 5)** until satisfied; **release** for RL at **2.0 m/s** cap
5. **E-stop ready**

---

## User action required

- [ ] Confirm car is physically on/near centerline start after `set_initial_pose.sh`
- [ ] Do not start competing `localize_launch.py` on the car (conflicts with iv2026 PF)
- [ ] For real 1v1: park opponent car ahead for detector validation, or accept zero-opponent block until tuned
