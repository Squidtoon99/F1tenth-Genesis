#!/usr/bin/env bash
# Durable self-chaining orchestrator: waits for clean_1v0_v1 to finish, then runs
# clean_1v1_v1 (scripted opponent) and selfplay_clean_v1 (self-play) sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

VENV_PYTHON="$ROOT/venv/bin/python"
TRAINER="$ROOT/standalone_trainer.py"
CHAIN_LOG="$ROOT/outputs/irl_training_chain.log"
POLL_INTERVAL=60
POST_EXIT_WAIT=300  # seconds to wait for final ckpt after trainer exits

STAGE_1V0_RUN_ID="clean_1v0_v1"
STAGE_1V1_RUN_ID="clean_1v1_v1"
STAGE_SELFPLAY_RUN_ID="selfplay_clean_v1"
TOTAL_STEPS=500000
WANDB_GROUP="${WANDB_GROUP:-irl_mac_$(date +%Y-%m-%d)}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

mkdir -p "$ROOT/outputs"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$CHAIN_LOG"
}

run_dir() {
  echo "$ROOT/outputs/runs/$1"
}

final_ckpt() {
  echo "$(run_dir "$1")/checkpoints/ckpt_${TOTAL_STEPS}.pt"
}

is_trainer_running() {
  local run_id="$1"
  pgrep -f "standalone_trainer\.py.*--run-id ${run_id}" >/dev/null 2>&1
}

stage_done() {
  local run_id="$1"
  [[ -f "$(final_ckpt "$run_id")" ]] && ! is_trainer_running "$run_id"
}

wait_for_stage() {
  local run_id="$1"
  local label="$2"
  local post_exit_start=0

  if $DRY_RUN; then
    if stage_done "$run_id"; then
      log "DRY-RUN ${label}-done: $(final_ckpt "$run_id") (already complete)"
    else
      log "DRY-RUN would wait for ${label} ($(final_ckpt "$run_id"))"
    fi
    return 0
  fi

  while true; do
    if stage_done "$run_id"; then
      log "${label}-done: $(final_ckpt "$run_id")"
      return 0
    fi

    if ! is_trainer_running "$run_id"; then
      if [[ -f "$(final_ckpt "$run_id")" ]]; then
        log "${label}-done: $(final_ckpt "$run_id")"
        return 0
      fi

      if [[ $post_exit_start -eq 0 ]]; then
        post_exit_start=$(date +%s)
        log "${label}: trainer exited, waiting up to ${POST_EXIT_WAIT}s for final checkpoint..."
      elif (( $(date +%s) - post_exit_start > POST_EXIT_WAIT )); then
        log "FAILED: ${label} trainer exited but $(final_ckpt "$run_id") never appeared"
        exit 1
      fi
    else
      post_exit_start=0
    fi

    sleep "$POLL_INTERVAL"
  done
}

run_training_stage() {
  local label="$1"
  local run_id="$2"
  local init_ckpt="$3"
  shift 3
  local -a extra_args=("$@")

  local stage_log
  stage_log="$(run_dir "$run_id")/chain.log"
  mkdir -p "$(run_dir "$run_id")"

  local -a cmd=(
    caffeinate -i -s
    "$VENV_PYTHON" "$TRAINER"
    --track IV_2026_SIM
    --num-envs 8
    --total-steps "$TOTAL_STEPS"
    --precision 64
    --seed 42
    --run-id "$run_id"
    --ckpt-interval 10000
    --init-ckpt "$init_ckpt"
    --wandb
    --wandb-mode online
    --wandb-group "$WANDB_GROUP"
  )
  cmd+=("${extra_args[@]}")

  if $DRY_RUN; then
    log "DRY-RUN ${label}-launch: ${cmd[*]}"
    return 0
  fi

  log "${label}-launched: run_id=${run_id} init_ckpt=${init_ckpt}"
  log "${label}-cmd: ${cmd[*]}"

  # Foreground: the orchestrator screen blocks here until this stage completes.
  "${cmd[@]}" >>"$stage_log" 2>&1

  wait_for_stage "$run_id" "$label"
}

log "armed: waiting for ${STAGE_1V0_RUN_ID} to finish ($(final_ckpt "$STAGE_1V0_RUN_ID"))"

wait_for_stage "$STAGE_1V0_RUN_ID" "1v0"

INIT_1V1="$(final_ckpt "$STAGE_1V0_RUN_ID")"
run_training_stage "1v1" "$STAGE_1V1_RUN_ID" "$INIT_1V1" \
  --opponent scripted

INIT_SELFPLAY="$(final_ckpt "$STAGE_1V1_RUN_ID")"
run_training_stage "selfplay" "$STAGE_SELFPLAY_RUN_ID" "$INIT_SELFPLAY" \
  --self-play --opponent policy

log "chain-complete: all stages finished"
