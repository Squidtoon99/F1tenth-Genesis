#!/usr/bin/env bash
# Poll origin for new commits, pull fast-forward updates, and launch a lightweight
# "bleeding edge" trainer when training-related files change.
#
# Intended to run alongside the overnight orchestrator: the long-running trainer
# keeps going on the code it was started with, while this watcher keeps the
# checkout current and spins up smaller validation runs for new commits.
#
# Environment overrides:
#   GIT_REMOTE          default: origin
#   GIT_BRANCH          default: current branch
#   POLL_INTERVAL_S     default: 900 (15 min)
#   BLEEDING_NUM_ENVS   default: 128 when another trainer is active, else 256
#   BLEEDING_TOTAL_STEPS default: 100000
#   BLEEDING_GPU_FREE_MIB minimum free GPU memory to launch (default: 6000)
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATE_DIR="${ROOT}/outputs/runs/_git_sync"
STATE_FILE="${STATE_DIR}/state.env"
LOG_FILE="${STATE_DIR}/watcher.log"
mkdir -p "$STATE_DIR"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-900}"
BLEEDING_TOTAL_STEPS="${BLEEDING_TOTAL_STEPS:-100000}"
BLEEDING_GPU_FREE_MIB="${BLEEDING_GPU_FREE_MIB:-6000}"

WATCH_PREFIXES=(
  standalone_trainer.py
  config.py
  scripts/
  f1tenth_env/
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] git_sync $*" | tee -a "$LOG_FILE"
}

load_state() {
  LAST_TRAINED_SHA=""
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  cat >"$STATE_FILE" <<EOF
LAST_TRAINED_SHA=${LAST_TRAINED_SHA:-}
EOF
}

trainer_pids() {
  pgrep -f "standalone_trainer.py" 2>/dev/null || true
}

bleeding_trainer_running() {
  pgrep -af "[s]tandalone_trainer.py.*bleeding_" >/dev/null 2>&1
}

other_trainer_running() {
  pgrep -af "[s]tandalone_trainer.py" | grep -v "bleeding_" >/dev/null 2>&1
}

gpu_free_mib() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 999999
    return
  fi
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

relevant_changes() {
  local from_sha="$1"
  local to_sha="$2"
  if [ -z "$from_sha" ]; then
    return 0
  fi
  local changed
  changed=$(git diff --name-only "$from_sha" "$to_sha" 2>/dev/null || true)
  if [ -z "$changed" ]; then
    return 1
  fi
  local path
  for path in $changed; do
    for prefix in "${WATCH_PREFIXES[@]}"; do
      if [[ "$path" == "$prefix" || "$path" == ${prefix}* ]]; then
        return 0
      fi
    done
  done
  return 1
}

launch_bleeding_trainer() {
  local sha="$1"
  local short_sha="${sha:0:8}"
  local run_id="bleeding_${short_sha}_p64"
  local run_dir="${ROOT}/outputs/runs/${run_id}"
  local num_envs="${BLEEDING_NUM_ENVS:-}"

  if other_trainer_running; then
    num_envs="${num_envs:-128}"
  else
    num_envs="${num_envs:-256}"
  fi

  local free_mib
  free_mib=$(gpu_free_mib)
  if [ "${free_mib:-0}" -lt "$BLEEDING_GPU_FREE_MIB" ]; then
    log "skip launch run_id=${run_id}: GPU free ${free_mib}MiB < ${BLEEDING_GPU_FREE_MIB}MiB"
    return 1
  fi

  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
  # shellcheck disable=SC1091
  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate genesis
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a

  local group="bleeding_$(date +%Y-%m-%d)"
  log "launch run_id=${run_id} envs=${num_envs} steps=${BLEEDING_TOTAL_STEPS} sha=${short_sha} group=${group}"

  mkdir -p "$run_dir"
  nohup python "${ROOT}/standalone_trainer.py" \
    --run-id "$run_id" \
    --wandb-group "$group" \
    --hypothesis "Bleeding-edge validation for ${short_sha}" \
    --num-envs "$num_envs" \
    --total-steps "$BLEEDING_TOTAL_STEPS" \
    --alpha 0.01 \
    --min-train-samples 40000 \
    --track IV_2026_SIM \
    --precision 64 \
    --ckpt-interval 10000 \
    --seed 42 \
    --buffer-capacity 1000000 \
    --log-interval 100 \
    --wandb-mode online \
    --self-play \
    --wandb \
    --run-dir "$run_dir" \
    >> "${run_dir}/run.log" 2>&1 &

  echo $! > "${STATE_DIR}/bleeding.pid"
  LAST_TRAINED_SHA="$sha"
  save_state
}

load_state
log "watcher start branch=${BRANCH} remote=${REMOTE} poll=${POLL_INTERVAL_S}s last_trained=${LAST_TRAINED_SHA:-none}"

while true; do
  load_state

  if bleeding_trainer_running; then
    log "bleeding trainer active; waiting"
    sleep "$POLL_INTERVAL_S"
    continue
  fi

  if ! git fetch "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1; then
    log "git fetch failed"
    sleep "$POLL_INTERVAL_S"
    continue
  fi

  local_sha=$(git rev-parse HEAD)
  remote_sha=$(git rev-parse "${REMOTE}/${BRANCH}" 2>/dev/null || echo "")

  if [ -z "$remote_sha" ]; then
    log "no remote ref ${REMOTE}/${BRANCH}"
    sleep "$POLL_INTERVAL_S"
    continue
  fi

  if [ "$local_sha" != "$remote_sha" ]; then
    log "behind remote (${local_sha:0:8} -> ${remote_sha:0:8}); pulling"
    if ! git pull --ff-only "$REMOTE" "$BRANCH" >>"$LOG_FILE" 2>&1; then
      log "git pull failed — resolve manually"
      sleep "$POLL_INTERVAL_S"
      continue
    fi
    local_sha=$(git rev-parse HEAD)
    log "now at ${local_sha:0:8}"
  fi

  if [ "$local_sha" = "${LAST_TRAINED_SHA:-}" ]; then
    sleep "$POLL_INTERVAL_S"
    continue
  fi

  if [ -z "${LAST_TRAINED_SHA:-}" ]; then
    log "initializing state at ${local_sha:0:8} (no launch on first poll)"
    LAST_TRAINED_SHA="$local_sha"
    save_state
    sleep "$POLL_INTERVAL_S"
    continue
  fi

  if relevant_changes "${LAST_TRAINED_SHA}" "$local_sha"; then
    launch_bleeding_trainer "$local_sha" || true
  else
    log "commit ${local_sha:0:8} has no training-relevant changes; marking seen"
    LAST_TRAINED_SHA="$local_sha"
    save_state
  fi

  sleep "$POLL_INTERVAL_S"
done
