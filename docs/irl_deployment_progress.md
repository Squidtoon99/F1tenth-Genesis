# IRL Deployment Progress Tracker

Living status for sim-to-real (IRL) readiness. Updated against the repo on branch
`cursor/cbfc99e3` after plan items **#11**, **#1**, and **#3** (this batch).

Severity scale: **Critical** (gates IRL) / **High** / **Medium** / **Low**.

---

## MODEL readiness (~45%)

Training stack, observation/reward contract, and 1v1 opponent behavior needed before
any checkpoint is deploy-safe.

| Area | ~Done | Blockers |
| --- | --- | --- |
| Observation / track geometry | 85% | Per-point corridor widths, min-lookahead, tyre-slip in training |
| Reward / termination stack | 70% | Collision penalty added (#3); clean retrain still pending (#2) |
| 1v1 opponent | 75% | Closed-loop speed control (#1); clean 1v1 retrain pending (#2b) |
| Self-play | 60% | Infra live in `standalone_trainer.py`; needs clean 1v1 warm-start (#4) |
| Domain randomization | 0% | No DR in `f1tenth_env` yet (#5) |
| Contact / NaN stability | 50% | Reset guard exists; long soak unproven (#6) |
| Clean checkpoints | 0% | All existing ckpts predate obs/reward fixes (#2, #2b) |

---

## HARDWARE / EVALUATION-GROUND (~25%)

Sim deploy parity and on-car validation path.

| Area | ~Done | Blockers |
| --- | --- | --- |
| Solo sim deploy (380-dim) | 90% | `obs_core` parity tests pass; IV_2026 map bundled |
| 1v1 sim deploy (387-dim) | 70% | `enable_opponent_obs` path in agent + vehicle stacks; detector uncommitted (#8) |
| Tyre-slip deploy parity | 60% | `--zero-tyre-slip-obs` ablates `[372:380]` in training (#7); on-car estimator deferred |
| Real-track centerline | 0% | No surveyed map aligned to PF frame (#9) |
| Solo IRL shakedown | 0% | No staged hardware run, bags, or lap logs (#10) |

---

## Issues table

| Issue | Area | Status | Evidence |
| --- | --- | --- | --- |
| Stale docs / missing tracker | Hygiene | **RESOLVED** (#11) | This file; `training_1v1.md`, ROS width docs, `test_track_io.py` updated |
| Immobile scripted opponent | Model | **RESOLVED** (#1) | `ScriptedCenterlineOpponent` P-speed control; `opponent_kp_speed`; Genesis test |
| No shaped collision penalty | Model | **RESOLVED** (#3) | Config-gated `collision_k` in `rewards.py`; Genesis collision test |
| Pre-fix checkpoints | Model | **OPEN** (#2) | No long 1v0 retrain on current obs/reward stack |
| Clean 1v1 baseline | Model | **OPEN** (#2b) | Blocked on #1 + #3 smoke; retrain not started |
| Self-play from stale ckpt | Model | **PARTIAL** (#4) | `SelfPlayManager` + `--self-play` in trainer; needs clean 1v1 init |
| Domain randomization | Model | **OPEN** (#5) | Not implemented |
| Genesis NaN under contact | Model | **PARTIAL** (#6) | Mitigation in env; no NaN-rate metric / long soak |
| Tyre-slip train/deploy gap | Deploy | **PARTIAL** (#7) | `zero_tyre_slip_obs` flag zeros `[372:380]` in training; on-car slip estimator hardware-gated/deferred |
| Opponent detector uncommitted | Deploy | **OPEN** (#8) | `opponent_detector_node.cpp` WIP; not in main commits |
| Real-track mapping | Deploy | **OPEN** (#9) | No aligned centerline CSV for physical track |
| Solo IRL shakedown | Deploy | **OPEN** (#10) | Hardware milestone not started |
| Per-point track width docs | Hygiene | **RESOLVED** (#11) | IV_2026 CSV ~1.33 m mean total width from `w_tr_left`/`w_tr_right` |

---

## Prioritized roadmap

Critical path (from plan):

1. ~~**#11 Docs/test cleanup**~~ — **DONE** (this batch)
2. ~~**#1 Mobile centerline opponent**~~ — **DONE** (this batch)
3. **#2 Clean 1v0 retrain** — long solo run on `IV_2026_SIM` with current stack
4. ~~**#3 Collision penalty**~~ — **DONE** (this batch)
5. **#2b Clean 1v1 retrain** — vs mobile scripted opponent
6. **#4 Self-play** — warm-start from clean 1v1 ckpt
7. **#8 Opponent detector** — commit node + gym validation of `[380:387]`
8. **#9 Real-track map** — SLAM + aligned centerline; verify `obs[10] ≈ 0`
9. **#10 Solo IRL shakedown** — staged low-speed run, bags, E-stop check
10. **#5 Domain randomization** — friction/mass/latency/obs noise
11. ~~**#7 Tyre-slip parity**~~ — training ablate via `zero_tyre_slip_obs` (on-car estimator deferred)
12. **#6 NaN soak** — metric + long 1v1 contact test

Parallel: #5 and #6 can run alongside retrains.

---

## Plan batch status (cursor/cbfc99e3)

| Plan item | Status after this batch |
| --- | --- |
| #11 Docs/test cleanup | **Complete** |
| #1 Mobile opponent | **Complete** |
| #3 Collision penalty | **Complete** |
| #2 / #2b / #4+ | **Not started** (next: clean 1v0 retrain) |
