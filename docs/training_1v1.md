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
| `--opponent {none,scripted,policy}` | `none` | `none` = solo/1v0. `scripted` = centerline-following pace car (recommended). `policy` = frozen-policy self-play opponent (interface only; training loop deferred). |
| `--opponent-target-speed FLOAT` | `3.0` | Scripted opponent target speed (m/s). Keep it **below** the ego's achievable pace so an overtake is feasible (too fast → the ego can never pass and `passing` never goes positive). |
| `--opponent-spawn-gap FLOAT` | `7.0` | Meters the opponent spawns ahead of the ego along the centerline at every reset. Smaller = collisions/overtakes happen sooner; larger = more approach room. |
| `--passing-scale FLOAT` | `0.5` | Reward scale on the passing term `k * (ego_ds - opp_ds)`. Positive when the ego gains track position. Raise it to push overtaking harder; lower it if it dominates clean-driving terms. Ignored when `--opponent none`. |
| `--opponent-ckpt PATH` | `None` | Only for `--opponent policy`: the frozen actor checkpoint to drive the opponent. |

All the standard trainer flags still apply (`--num-envs`, `--total-steps`,
`--batch-size`, `--alpha`, `--device`, `--precision`, `--seed`, `--wandb`, …). The
underlying tunables (`opponent_kp_ey`, `opponent_kh_heading`, `collision_dist_m`,
`term_on_collision`, `opponent_obs_dim`, `passing_k`) live in
[config.py](../config.py) under the `env` / `obs` / `reward` sections and can be
overridden there if needed.

## What changes under the hood

- **Observation** (`enable_opponent_obs = True`): a 7-dim block is appended —
  opponent position and velocity in the ego body frame, signed normalized
  along-track gap, opponent lateral offset, and a presence flag. When no opponent is
  present for a row the block is the exact zero sentinel. Full layout in
  [observation_audit.md](observation_audit.md).
- **Reward**: a `passing` term `passing_k * (ego_ds - opp_ds)` (per-step arc-length
  deltas) is added. It is reset-safe and wrap-safe (no start/finish-line spikes) and
  is **gated** by the presence of the `passing` reward scale, so the solo reward is
  byte-for-byte unchanged. There is intentionally no shaped collision penalty.
- **Termination**: the episode ends when the two cars are within `collision_dist_m`
  (default 0.4 m). This is the entire collision-handling story — ending the episode
  forfeits future progress reward, which is the only "don't crash" signal needed.

## Verify before a long run

Run the in-sim assertion harness first. It hard-fails on a wrong observation shape,
non-finite obs/reward, a missing/spiky passing term, or a collision path that never
triggers, and writes a passing-reward plot to `outputs/verify_1v1/`:

```bash
# Genesis's numba cache cannot write into site-packages under some sandboxes;
# point it at a writable dir.
NUMBA_CACHE_DIR=/tmp/numba_cache python scripts/verify_1v1.py --num-envs 16 --steps 250
```

Expected: `obs_shape=(16, 387)`, `collisions > 0`, and the 1v0 regression check
reports `obs_shape=(16, 380)`.

You can also run the deterministic unit tests (no simulator needed):

```bash
python -m pytest tests/test_opponent_obs.py tests/test_passing_reward.py tests/test_collision_term.py -q
```

## Checkpoints and deployment

Checkpoints are written to `outputs/standalone/<run_id>/ckpt_<step>.pt` with `actor`,
`critic1`, `critic2`, and `obs_norm` (observation mean/var). A 1v1 checkpoint expects
a **387-dim** observation at inference.

The ROS port (`ros2_deploy/.../obs_core.py`) currently builds only the 380-dim base
vector; deploying a 1v1 policy requires it to also emit the 7-dim opponent block from
perception (with the same zero sentinel when no opponent is detected). The exact
contract it must satisfy is documented in
[observation_audit.md](observation_audit.md) under "1v1 opponent observation block".

## Future: self-play (`--opponent policy`)

The `PolicyOpponent` controller and `--opponent-ckpt` plumbing exist so a frozen
policy can drive the opponent today. The self-play *training loop* — periodically
snapshotting the learner into the opponent — is intentionally deferred; the
checkpoint format above (`actor` + `obs_norm`) is already what `PolicyOpponent`
loads, so wiring it up later needs no changes to the env or observation contract.
The system is hard-limited to a single opponent (1v1); there is no multi-agent path.
