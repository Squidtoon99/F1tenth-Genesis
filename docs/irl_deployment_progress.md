# IRL Deployment Progress Tracker

Living status for sim-to-real (IRL) readiness. Updated against the repo on branch
`cursor/cbfc99e3` after plan items **#6**, **#5**, **#7**, and **#8** (this batch).

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
| Domain randomization | 40% | Config-gated DR in `f1tenth_env` (#5); default off |
| Contact / NaN stability | 75% | Non-finite rate metrics + long 1v1 contact soak (#6) |
| Clean checkpoints | 0% | All existing ckpts predate obs/reward fixes (#2, #2b) |

---

## HARDWARE / EVALUATION-GROUND (~25%)

Sim deploy parity and on-car validation path.

| Area | ~Done | Blockers |
| --- | --- | --- |
| Solo sim deploy (380-dim) | 90% | `obs_core` parity tests pass; IV_2026 map bundled |
| 1v1 sim deploy (387-dim) | 80% | Detector + obs wiring gym-validated (#8); needs on-car run |
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
| Domain randomization | Model | **RESOLVED** (#5) | Config-gated `domain_randomization` block; Genesis tests |
| Genesis NaN under contact | Model | **RESOLVED** (#6) | Non-finite rate metrics in trainer; 600-step 1v1 contact soak test |
| Tyre-slip train/deploy gap | Deploy | **RESOLVED** (#7) | `zero_tyre_slip_obs` zeros `[372:380]` in training; on-car estimator hardware-gated/deferred |
| Opponent detector uncommitted | Deploy | **RESOLVED** (#8) | Node committed (`ad2c8e3`); gym validation harness PASS on IV_2026_SIM |
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
7. ~~**#8 Opponent detector**~~ — gym validation of `[380:387]` **DONE**
8. **#9 Real-track map** — SLAM + aligned centerline; verify `obs[10] ≈ 0`
9. **#10 Solo IRL shakedown** — staged low-speed run, bags, E-stop check
10. ~~**#5 Domain randomization**~~ — friction/mass/latency/obs noise **DONE**
11. ~~**#7 Tyre-slip parity**~~ — training ablate via `zero_tyre_slip_obs` (on-car estimator deferred)
12. ~~**#6 NaN soak**~~ — metric + long 1v1 contact test **DONE**

Parallel: #5 and #6 can run alongside retrains.

---

## Plan batch status (cursor/cbfc99e3)

| Plan item | Status after this batch |
| --- | --- |
| #11 Docs/test cleanup | **Complete** |
| #1 Mobile opponent | **Complete** |
| #3 Collision penalty | **Complete** |
| #6 NaN metric + soak | **Complete** |
| #5 Domain randomization | **Complete** |
| #7 Tyre-slip ablate | **Complete** |
| #8 Opponent obs gym validation | **Complete** |
| #2 / #2b / #4+ | **Not started** (next: clean 1v0 retrain) |
