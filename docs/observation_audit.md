# Observation accuracy audit

Independent verification of every value in the `num_obs = 380` observation vector
consumed by the policy. This document is both the **specification** (what each
value is supposed to be) and the **findings report** (what the tests actually
show).

Scope of the original audit: audit + tests + findings only. The two confirmed
issues (tyre-slip frame, speed-zero future-point collapse) were subsequently fixed
and verified - see the "Resolution" section at the end of this document.

How the audit is performed (three layers):

1. Analytic known-answer unit tests on synthetic tracks (`tests/test_observation_geometry.py`,
   `tests/test_tyre_slip.py`) - no Genesis sim, closed-form ground truth.
2. Brute-force Frenet cross-check (`tests/test_frenet_projection.py`) - the
   windowed/coarse projection vs an exhaustive all-segment projection.
3. In-sim instrumentation harness (`scripts/audit_observations.py`) - the real
   Genesis env driven through scripted maneuvers, every observation slice
   compared against an independent re-derivation from raw Genesis state.

Status legend: CORRECT (verified), SUSPECT (flagged, under test), WRONG
(confirmed inaccurate), DESIGN (works as written but semantically questionable).

---

## Observation layout (built in `f1tenth_env/observations.py::build_observation`)

Order matches the concatenation in `build_observation`.

### `[0:2]` body-frame linear velocity (vx, vy)
- Source: `inv_transform_by_quat(car.get_vel(), quat)[:, :2]` (`env.py`).
- Intended: ego/body-frame planar velocity, x = forward, y = left, m/s.
- Verify: rotate world velocity into body frame by hand (quaternion) and compare.
- Status: CORRECT (in-sim max abs err 3.8e-6 vs independent quaternion rotation).

### `[2:3]` body-frame yaw rate (omega_z)
- Source: `inv_transform_by_quat(car.get_ang(), quat)[:, 2]`.
- Intended: yaw angular velocity about body z, rad/s.
- Verify: independent quaternion rotation of world angular velocity.
- Status: CORRECT (in-sim max abs err ~1e-6).

### `[3:5]` body-frame linear acceleration (ax, ay)
- Source: `inv_transform_by_quat(get_links_acc(base_link), quat)[:, :2]`.
- Intended: proper body-frame planar acceleration, m/s^2.
- Risk: `get_links_acc` may carry the gravity term (engine `cacc` convention),
  which leaks into ax/ay under pitch/roll and puts ~g on az.
- Verify: at rest and at constant velocity ax,ay ~ 0; inspect raw az.
- Status: CORRECT (no gravity leak) with a caveat. Body az read from the env's own
  buffer stays near 0 (-0.5..+0.6 m/s^2 under accel/brake), so gravity is NOT
  leaking a ~9.81 offset into the planar accel. The body-frame rotation itself is
  trusted (vx/vy/omega all verified to ~1e-6 with the same transform). CAVEAT: the
  raw magnitude could not be re-derived independently because `get_links_acc` is
  stateful (re-calling it after the env already read it returns inconsistent
  values), and a finite-difference cross-check differs by definition
  (instantaneous link accel vs control-step average). The obs faithfully echoes the
  env's stored accel except where `clip_obs` clips large spikes.

### `[5:7]` last actions (throttle, steer)
- Source: `self.last_actions` (the executed action, post-clip, normalized [-1, 1]).
- Verify: compare to the action stream the harness applies.
- Status: CORRECT. In a no-reset run the echo matches the previously applied action
  exactly (err 0). With resets enabled there is an expected one-step value at reset
  steps because reset re-seeds `last_actions[:,0]` with `reset_throttle` before the
  obs is built - intended behavior, not an error.

### `[7:9]` track progress (cos, sin)
- Source: `obs_track_progress` = [cos, sin](2*pi*s/L).
- Verify: car placed at known arc length s -> exact cos/sin.
- Status: CORRECT. Analytic circle test passes; in-sim mean abs err 3.8e-3. Rare
  large values (max ~1.7) coincide with sharp-corner segment-vertex ties where s
  jumps between adjacent segments (same artifact as centerline angle below).

### `[9:10]` centerline heading error (SUSPECT)
- Source: `obs_centerline_angle` = wrap(yaw - atan2(seg_dir_y, seg_dir_x)).
- Intended: signed heading of the car relative to the forward track tangent, rad.
- Verify: straight track (tangent angle 0) -> err == yaw; circle -> err ==
  wrap(yaw - (phi + pi/2)); plus brute-force projection cross-check on the real track.
- Status: CORRECT in the trained (on-track) regime, with two caveats. Analytic
  tests pass exactly. On the real track, when the projection lands in a segment
  INTERIOR the heading error matches brute force to 3.5e-7 (max). The only large
  on-track discrepancies (up to ~pi) occur AT segment vertices, where the env and
  the brute-force projection legitimately pick different adjacent segments (a tie);
  this is a discretization wart shared by both, not an env bug, but it does make the
  heading observation briefly discontinuous at sharp corners.
  SECOND caveat: when the car runs FAR off-track (stress run, resets disabled) the
  windowed/coarse projection picks a wrong segment and the heading error blows up
  (interior max 2.9). During training, OOB termination prevents this regime.

### `[10:11]` signed lateral error ey (SUSPECT)
- Source: `obs_centerline_distance` = boundary `ey` = (pos - proj) . n_hat,
  n_hat = [-t_y, t_x] (left normal). ey > 0 => left of centerline.
- Verify: straight track car at (x, y0) -> ey == y0; circle radius r -> ey == r - R;
  sign cross-checked against the CSV width columns; brute-force projection cross-check.
- Status: CORRECT in the trained regime. Analytic straight/circle tests pass
  (sign + magnitude). In-sim on-track max abs err vs brute force 0.066 m (mean
  7e-5). Like the heading term, ey degrades only when the car is FAR off-track
  (stress max 25 m) because the windowed projection then selects a wrong segment;
  OOB termination keeps training out of that regime.

### `[11:12]` contact flag
- Source: `obs_contact_flag` = (boundary_dist < contact_margin_m).float().
- Verify: place car at a known distance inside/outside the margin band.
- Status: CORRECT (analytic test passes; in-sim err 0 vs brute-force boundary
  distance). Note its accuracy inherits from ey, so it shares ey's far-off-track
  caveat.

### `[12:372]` future track points (center/left/right x 60 x (x,y), ego frame)
- Source: `obs_future_track_points`. Samples `s0 + speed*horizon_s*k/N` along the
  centerline, builds +/- half-width offsets, rotates into the ego frame.
- Risks: (a) when speed ~ 0 the lookahead is 0 and all 60 samples collapse to the
  current point (no preview); (b) `s`/`L` use the loop-closed cache while the
  future-point arclength uses the open-polyline `cumlen`, which can mis-index near
  the start/finish seam.
- Verify: straight track -> center points on +x at expected spacing, left/right at
  +/- half-width; speed=0 degeneracy; seam behavior.
- Status: MOSTLY CORRECT, one DESIGN issue. Analytic layout test passes
  (ordering [center, left, right], ego transform, +/- half-width offsets). The
  closed-vs-open arc-length seam does NOT bite on IV_2026_SIM: the centerline CSV
  is already a closed loop, so closed_loop_len == open_polyline_len (133.771 m,
  gap 0) and there is no mis-index. DESIGN ISSUE (confirmed by unit test): at
  speed ~ 0 the lookahead is 0, so all 60 samples collapse onto the current point -
  the policy gets no track preview while stopped/crawling.

### `[372:380]` tyre slip (4 slip ratios + 4 slip angles) (SUSPECT)
- Source: `compute_tyre_slip` (`car.py`), wheel order [LR, RR, LF, RF].
  slip_angle = atan2(v_lat, |v_fwd|); slip_ratio = (r*omega - v_fwd) / max(|r*omega|, |v_fwd|).
- Risk: `wheel_state["frame_quat"]` (base_link for rear, steering hinge for front)
  is collected but never used; the wheel velocity from `get_links_vel(ref="link_com")`
  is treated as wheel-frame [fwd, lat] but is most likely world-frame and never
  rotated. Signature: a rear wheel driving straight at car yaw psi reports
  slip_angle ~ psi instead of ~ 0.
- Verify: rear slip_angle ~ 0 driving straight; free-roll slip_ratio ~ 0; throttle
  -> positive ratio; recompute the corrected (frame-rotated) slip and quantify the gap.
- Status: CONFIRMED WRONG. The wheel velocity fed to `compute_tyre_slip` is in the
  WORLD frame, not the wheel frame, so slip angle/ratio are computed against the
  wrong axes. Smoking-gun in-sim diagnostic (LR wheel): raw `motion_link_vel.x`
  tracks WORLD vx (0.31 vs world 0.32) while the frame-corrected value tracks BODY
  vx (2.74 vs body 2.76). Env-vs-frame-corrected slip differs by up to 3.07 (mean
  ~0.7). The unit test `test_world_frame_velocity_corrupts_slip_angle` reproduces
  the signature: a wheel rolling straight reports slip_angle == car yaw instead of
  ~ 0. Root cause: `wheel_state["frame_quat"]` is collected in env.py but never
  applied in `car.py::compute_tyre_slip`. The slip FORMULA itself is correct
  (verified by the wheel-frame unit tests).

---

## Findings

Evidence base:
- `tests/` (33 tests, all passing): analytic geometry, brute-force Frenet
  cross-check, tyre-slip formula + frame sensitivity.
- `scripts/audit_observations.py` on the real track `IV_2026_SIM`: in-sim
  comparison of every slice vs an independent re-derivation, both in the normal
  (terminations on) regime and a `--stress` (far off-track) regime. Raw numbers in
  `outputs/standalone/obs_audit/obs_audit.csv`.

### Headline result
Exactly one value is genuinely computed wrong: the **tyre slip** block. Everything
the user also suspected about the **centerline angle/distance** turned out to be
correct in the trained (on-track) regime; those terms only degrade when the car is
far off-track, which OOB termination already prevents during training.

### Suspect verdicts
- **Tyre slip - CONFIRMED WRONG.** Wheel velocity is left in the world frame;
  `frame_quat` is never applied. Slip angle/ratio are therefore meaningless during
  any cornering/yaw. This is the one fix that matters.
- **Centerline heading error - CORRECT (on-track).** Exact in segment interiors
  (max 3.5e-7). The only large values are (a) segment-vertex ties at sharp corners
  (a discretization wart shared with brute force) and (b) far-off-track windowing
  failures (outside the trained regime).
- **Signed lateral error ey - CORRECT (on-track).** Max 0.066 m vs brute force;
  sign convention (left = +) verified analytically. Same far-off-track caveat.

### Secondary leads
- **Acceleration gravity/frame leakage - NOT observed.** Body az stays near 0; no
  ~9.81 offset. Body-frame transform trusted. Raw magnitude not independently
  re-derivable (stateful `get_links_acc`); see the [3:5] caveat.
- **Closed vs open arc-length seam (future points) - NOT a problem on this track.**
  The centerline is a closed loop, so closed and open lengths are equal (gap 0 m).
  Would only matter on a track whose CSV is an open polyline.
- **Speed=0 future-point collapse - CONFIRMED (design issue).** No track preview
  while stopped/crawling. Plausibly reinforced the early "crawl" local optimum.

### Per-observation summary table

| Obs (index)                  | Status            | Max abs err (on-track) | Root cause / note | Recommended fix (NOT applied) | Retraining impact |
|------------------------------|-------------------|------------------------|-------------------|-------------------------------|-------------------|
| lin_vel [0:2]                | CORRECT           | 3.8e-6                 | -                 | none                          | -                 |
| ang_vel [2]                  | CORRECT           | 1e-6                   | -                 | none                          | -                 |
| lin_acc [3:5]                | CORRECT (caveat)  | n/a (stateful getter)  | no gravity leak; magnitude not independently verified | optionally add an independent accel cross-check | none |
| last_actions [5:7]           | CORRECT           | 0 (no-reset)           | reset re-seeds throttle (intended) | none | - |
| progress [7:9]               | CORRECT           | 3.8e-3 (mean)          | rare vertex/seam ties | none | - |
| centerline_angle [9]         | CORRECT on-track  | 3.5e-7 (interior)      | vertex ties at corners; fails far off-track | optional: finer projection / interpolate tangent across vertices | none if obs dim unchanged |
| centerline_dist [10]         | CORRECT on-track  | 0.066 m                | fails far off-track only | optional: brute-force/finer projection for robustness | none |
| contact [11]                 | CORRECT           | 0                      | inherits ey | none | - |
| future_track_points [12:372] | DESIGN issue      | layout correct         | speed~0 -> all samples collapse (no preview) | use a minimum lookahead distance (floor speed*horizon) so preview persists at low speed | changes obs VALUES at low speed; retrain recommended |
| tyre_slip [372:380]          | CONFIRMED WRONG   | up to 3.07 vs corrected| world-frame wheel vel; `frame_quat` never applied in `compute_tyre_slip` | rotate `motion_link_vel` by `frame_quat` (world->wheel) before computing slip | changes obs VALUES; retrain required |

### Recommended fixes (described, intentionally not applied)
1. **Tyre slip frame (priority).** In `f1tenth_env/car.py::compute_tyre_slip`, rotate
   each wheel's `motion_link_vel` by the inverse of `wheel_state["frame_quat"]`
   (world -> wheel frame) before taking column 0 as forward and column 1 as lateral.
   The corrected reference is already implemented in `scripts/audit_observations.py`
   and matches body velocity. Obs values change -> retrain.
2. **Speed=0 future-point preview.** Floor the lookahead distance (e.g.
   `max(speed*horizon_s, d_min)`) in `obs_future_track_points` so the policy still
   sees the upcoming track while crawling. Obs values change at low speed -> retrain.
3. **(Optional) Off-track / vertex robustness of the Frenet projection.** Not
   required for current training (OOB termination keeps the car on-track), but if
   off-track observations ever matter, widen/curvature-adapt the search window or
   interpolate the tangent across vertices to remove the corner discontinuity.

### How to reproduce
```bash
source venv/bin/activate
python -m pytest tests/ -q                       # analytic/cross-check tests
python scripts/audit_observations.py             # in-sim, trained regime
python scripts/audit_observations.py --stress    # in-sim, far off-track regime
```

---

## Resolution (fixes applied + verification)

Both confirmed issues were fixed on the training side (deployment left unchanged,
see follow-up below). Both fixes change observation VALUES, so prior checkpoints
are invalid and a retrain is required.

### Fix 1 - Tyre slip frame (`[372:380]`)
- Change: [f1tenth_env/car.py](f1tenth_env/car.py) `compute_tyre_slip` now rotates
  `wheel_state["motion_link_vel"]` (world frame) into each wheel's frame via
  `gu.inv_transform_by_quat(lin_vel, frame_quat)` before splitting forward/lateral.
  `frame_quat` is optional (absent -> velocity treated as already wheel-frame, for
  the analytic unit tests). The env already supplies `frame_quat`
  ([f1tenth_env/env.py](f1tenth_env/env.py):562).
- Verification:
  - Unit: `tests/test_tyre_slip.py::test_frame_quat_rotation_recovers_zero_slip`
    (a wheel rolling straight at any yaw -> slip_angle/ratio ~ 0) and
    `test_frame_quat_rotation_preserves_lateral_slip`.
  - In-sim ([scripts/audit_observations.py](scripts/audit_observations.py)):
    `tyre_slip env vs frame-corrected` max abs err dropped from ~3.07 to
    **2.6e-5 (PASS)**; frame diagnostic now shows the env value tracking BODY
    velocity (corrected.x ~ body vx) instead of world velocity.

### Fix 2 - Future-point preview at low speed (`[12:372]`)
- Change: new config key `future_track_min_lookahead_m` (default 5.0) in
  [config.py](config.py); the lookahead is floored with
  `torch.clamp(speed*horizon_s, min=min_lookahead)` identically in
  [f1tenth_env/observations.py](f1tenth_env/observations.py) and the ROS port
  [obs_core.py](ros2_deploy/f1tenth_rl_agent/f1tenth_rl_agent/obs_core.py)
  (plus the key in `interfaces.default_obs_cfg`). Only affects speeds below
  `min_lookahead/horizon_s` (~0.83 m/s).
- Verification:
  - Unit: `tests/test_observation_geometry.py::test_future_points_speed_zero_uses_min_lookahead`
    (samples now span the floor instead of collapsing).
  - Parity: `ros2_deploy/f1tenth_rl_agent/test/test_obs_parity.py` PASS (env and
    `obs_core` stay in lockstep).
  - In-sim: low-speed (<0.5 m/s) center-line spread min **1.19 m (PASS)** vs the
    previous ~0 collapse.

### Test status
- `pytest tests/` - 38 passed.
- `pytest ros2_deploy/f1tenth_rl_agent/test/test_obs_parity.py` - 1 passed.
  (Note: run the two test roots separately; both define a `conftest.py`, so a
  combined run hits a module-name collision.)

### Notes / unrelated observations
- The in-sim track data loaded this session was a longer open-polyline variant of
  `IV_2026_SIM` (closed_loop_len 269.6 m, seam gap 0.144 m) vs the earlier closed
  133.8 m loop. With a non-zero seam the previously-documented vertex/seam/
  far-off-track geometry caveats are more visible in the raw harness numbers, but
  they are independent of the two fixes (which both pass) and were already flagged
  as acceptable in the trained regime.

### Follow-up (out of scope, not implemented)
- Deployment slip is still zero: [observation_builder_node.py](ros2_deploy/f1tenth_rl_agent/f1tenth_rl_agent/observation_builder_node.py)
  calls `builder.build(...)` without `tyre_slip`, so the `[372:380]` block is zeros
  at inference. The training fix makes slip meaningful in sim; wiring real
  wheel-frame slip (encoders/IMU) into the ROS node is a separate task needed for
  full sim-to-real parity.
- Frenet projection far-off-track/vertex robustness (audit verdict: correct in the
  trained regime).
