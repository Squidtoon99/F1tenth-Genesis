# Training with opponents (1v1)

The standalone trainer supports head-to-head 1v1 racing in addition to the solo
(1v0) task. When an opponent is enabled the ego learns to **overtake** a second car
on the same track: the observation gains a 7-dim opponent-relative block, a
`passing` reward term rewards gaining track position, and the episode ends on a
car-to-car collision.

This is opt-in. With no opponent flag the trainer behaves exactly as before (solo,
380-dim observation) — see [observation_audit.md](observation_audit.md) for the full
observation spec, including the 1v1 block layout (`[380:387]`).

## TL;DR

```bash
# Solo baseline (unchanged)
python standalone_trainer.py --num-envs 512 --total-steps 500000

# 1v1 vs the scripted centerline opponent (the shipped 1v1 setup)
python standalone_trainer.py \
  --opponent scripted \
  --num-envs 512 \
  --total-steps 2000000 \
  --device cuda --precision 32
```

Enabling `--opponent scripted` automatically:
- sets `opponent_strategy = "scripted"` and spawns one opponent ahead of the ego,
- enables the opponent observation block (`num_obs` 380 → **387**),
- activates the `passing` reward term (`--passing-scale`, default `0.5`).

Because the observation dimension changes, **1v1 networks are sized fresh at 387** —
a 1v0 checkpoint is not load-compatible (and vice versa). Training is from scratch,
as intended.

## Opponent flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--opponent {none,scripted,policy}` | `none` | `none` = solo/1v0. `scripted` = centerline-following pace car (recommended). `policy` = frozen-policy opponent; use with `--self-play` for delayed snapshot self-play. |
| `--mixed-opponents` | off | GT Sophy-style mixed opponent population: each env row is randomly assigned scripted or self-play policy on reset. Implies `--self-play` (the policy half refreshes from the snapshot pool). Mix weights come from `config.env.opponent_mix`. |
| `--opponent-target-speed FLOAT` | `3.0` | Scripted opponent target speed (m/s). Keep it **below** the ego's achievable pace so an overtake is feasible (too fast → the ego can never pass and `passing` never goes positive). |
| `--opponent-spawn-gap FLOAT` | `7.0` | Meters the opponent spawns ahead of the ego along the centerline at every reset. Smaller = collisions/overtakes happen sooner; larger = more approach room. |
| `--passing-scale FLOAT` | `0.5` | Reward scale on the passing term `k * (ego_ds - opp_ds)`. Positive when the ego gains track position. Raise it to push overtaking harder; lower it if it dominates clean-driving terms. Ignored when `--opponent none`. |
| `--collision-scale FLOAT` | `1.0` | Reward scale on the GT Sophy any-collision term `Rc = -collision_k` (binary, per-step penalty on car-car box overlap, regardless of fault). Ignored when `--opponent none`. |
| `--rear-end-scale FLOAT` | `1.0` | Reward scale on the GT Sophy rear-end term `Rr = -rear_end_k * ‖v_ego - v_opp‖²`, applied when the ego collides with an opponent that is **ahead** on the centerline (scales with squared closing speed). `0.0` disables it. Ignored when `--opponent none`. |
| `--opponent-ckpt PATH` | `None` | Only for `--opponent policy`: the frozen actor checkpoint to drive the opponent. |

### Episode horizon flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--episode-length FLOAT` | auto | Episode horizon in **seconds**. Default: derived from the track centerline length (`episode_lap_multiplier` laps at `expected_lap_speed_mps`), so each track gets multi-lap racing time instead of the old single-lap 45 s. |
| `--episode-lap-multiplier FLOAT` | `3.0` | Number of laps the auto-derived horizon should cover. Ignored if `--episode-length` is set. |

The episode now ends on the **time** horizon (config `target_laps = 0`, disabling
single-lap cutoff) like GT Sophy's base scenarios, giving the agent multi-lap traffic
and overtaking exposure. The startup log prints the resolved horizon and control-step
count. Set `target_laps > 0` in [config.py](../config.py) to cap by lap count instead.

All the standard trainer flags still apply (`--num-envs`, `--total-steps`,
`--batch-size`, `--alpha`, `--device`, `--precision`, `--seed`, `--wandb`, …). The
underlying tunables (`opponent_kp_ey`, `opponent_kh_heading`, `opponent_kp_speed`,
`car_length`, `car_width`, `collision_margin_m`, `term_on_collision`,
`opponent_obs_dim`, `passing_k`, `collision_k`) live in [config.py](../config.py)
under the `env` / `obs` / `reward` sections and can be overridden there if needed.

## What changes under the hood

- **Observation** (`enable_opponent_obs = True`): a 7-dim block is appended —
  opponent position and velocity in the ego body frame, signed normalized
  along-track gap, opponent lateral offset, and a presence flag. When no opponent is
  present for a row the block is the exact zero sentinel. Full layout in
  [observation_audit.md](observation_audit.md).
- **Reward**: a `passing` term `passing_k * (ego_ds - opp_ds)` (per-step arc-length
  deltas) is added. It is reset-safe and wrap-safe (no start/finish-line spikes) and
  is **gated** by the presence of the `passing` reward scale, so the solo reward is
  byte-for-byte unchanged. A config-gated `collision` term (`collision_k` × overlap
  mask) adds a negative reward on contact when enabled for 1v1 training.
- **Rear-end penalty (`Rr`)**: the GT Sophy rear-end term (Wurman et al., Nature 2022)
  `-rear_end_k * ‖v_ego - v_opp‖²` fires only when the ego is in a car-car collision
  with an opponent that is **ahead** on the centerline, scaling with the squared
  closing speed so high-speed rear-ends are punished hardest. It stacks on top of `Rc`
  and is gated by the `rear_end` reward scale (`--rear-end-scale`).
- **Termination**: the episode ends on car-to-car overlap detected by an anisotropic
  ego-frame box using `car_length` (0.46 m), `car_width` (0.30 m), and optional
  `collision_margin_m` (see `terminations.collision_mask`). Episode reset forfeits
  future progress reward in addition to any configured collision penalty.

## Verify before a long run

Run the in-sim assertion harness first. It hard-fails on a wrong observation shape,
non-finite obs/reward, a missing/spiky passing term, or a collision path that never
triggers, and writes a passing-reward plot to `outputs/verify_1v1/`:

```bash
# Genesis's numba cache cannot write into site-packages under some sandboxes;
# point it at a writable dir.
NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/verify_1v1.py --num-envs 16 --steps 250

# Verify the mixed opponent population (per-row scripted + policy) as well:
NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/verify_1v1.py --opponent mixed --num-envs 16 --steps 400
```

Expected: `obs_shape=(16, 387)`, `collisions > 0`, a non-zero `rear_end_min` on the
chase rollout, and the 1v0 regression check reports `obs_shape=(16, 380)`. The
`--opponent mixed` run additionally confirms both opponent modes are assigned and
scripted-mode rows reach cruise speed.

You can also run the deterministic unit tests (no simulator needed):

```bash
python -m pytest tests/test_opponent_obs.py tests/test_passing_reward.py tests/test_collision_term.py -q
```

## Checkpoints and deployment

Checkpoints are written to `outputs/standalone/<run_id>/ckpt_<step>.pt` with `actor`,
`critic1`, `critic2`, and `obs_norm` (observation mean/var). A 1v1 checkpoint expects
a **387-dim** observation at inference.

The ROS deploy stack supports both observation sizes: solo policies use 380 dims;
1v1 policies use **387** dims when `enable_opponent_obs` is set (sim
`observation_builder`, car `vehicle_obs`, and `policy_inference` all append the
7-dim opponent block from `/rl/opponent/odom` or LiDAR detection). The exact
contract is documented in [observation_audit.md](observation_audit.md) under
"1v1 opponent observation block".

## Self-play (`--self-play`)

Delayed self-play is implemented in `standalone_trainer.py` via `SelfPlayManager`:
the learner is snapshotted into a pool and the frozen `PolicyOpponent` is refreshed
periodically. Enable with `--self-play` (implies `--opponent policy`); tune cadence
with `--selfplay-snapshot-interval`, `--selfplay-refresh-interval`, and
`--selfplay-pool-size`. Warm-start with `--init-ckpt` to seed both the learner and
the opponent pool. The checkpoint format (`actor` + `obs_norm`) is what
`PolicyOpponent` loads. The system is hard-limited to a single opponent (1v1).

## Mixed opponent population (`--mixed-opponents`)

GT Sophy trains against a **mixed population** (built-in slower AI plus curated policy
snapshots) rather than pure self-play, which the paper found important to avoid
overfitting to a single opponent style. The hard-1v1 analogue here assigns each
parallel env row, on reset, to either the scripted centerline follower or the
frozen self-play policy:

```bash
python standalone_trainer.py \
  --mixed-opponents \
  --rear-end-scale 1.0 \
  --init-ckpt outputs/runs/foundation_1v1_v1/checkpoints/ckpt_500000.pt \
  --num-envs 512 --total-steps 2000000 \
  --run-id mixed_selfplay_v1
```

`--mixed-opponents` sets `opponent_strategy = "mixed"` and implies `--self-play`, so
the policy half of the population is refreshed from the learner snapshot pool exactly
like pure self-play (`env.refresh_opponent_policy` forwards snapshots to the inner
policy). The per-row sampling weights live in `config.env.opponent_mix`
(`scripted_weight`, `policy_weight`); warm-start the policy half with `--init-ckpt`.
