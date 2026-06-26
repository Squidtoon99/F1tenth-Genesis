#!/usr/bin/env bash
# Launch Mac solo (1v0) training with W&B logging, then chain 1v1 + self-play.
# Requires WANDB_API_KEY in .env (same project as the remote overnight trainer).
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
CHAIN="$ROOT/scripts/run_irl_training_chain.sh"

RUN_ID="${RUN_ID:-clean_1v0_v1}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-8}"
WANDB_GROUP="${WANDB_GROUP:-irl_mac_$(date +%Y-%m-%d)}"
SESSION="${TMUX_SESSION:-mac_train}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--1v0-only | --chain-only | --dry-run]

  (default)     Start 1v0 in tmux, then run the 1v1/self-play chain when it finishes.
  --1v0-only    Launch only the solo 1v0 stage (foreground).
  --chain-only  Run scripts/run_irl_training_chain.sh (waits for 1v0, then 1v1 + self-play).
  --dry-run     Print commands without executing.

Environment:
  WANDB_API_KEY   Required for online logging (set in .env).
  WANDB_GROUP     W&B group name (default: irl_mac_YYYY-MM-DD).
  RUN_ID          Solo run id (default: clean_1v0_v1).
  TOTAL_STEPS     Training steps per stage (default: 500000).
  NUM_ENVS        Parallel envs (default: 8).
  TMUX_SESSION    tmux session name for background 1v0 (default: mac_train).
EOF
}

MODE="full"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --1v0-only) MODE="1v0" ;;
    --chain-only) MODE="chain" ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY is not set. Create .env with your API key or run: wandb login" >&2
fi

run_dir="$ROOT/outputs/runs/$RUN_ID"
mkdir -p "$run_dir"

launch_1v0_cmd=(
  caffeinate -i -s
  "$VENV_PYTHON" -u "$TRAINER"
  --track IV_2026_SIM
  --num-envs "$NUM_ENVS"
  --total-steps "$TOTAL_STEPS"
  --precision 64
  --seed 42
  --run-id "$RUN_ID"
  --ckpt-interval 10000
  --wandb
  --wandb-mode online
  --wandb-group "$WANDB_GROUP"
  --hypothesis "Clean 1v0 solo baseline (Mac)"
)

if $DRY_RUN; then
  echo "WANDB_GROUP=$WANDB_GROUP"
  echo "1v0: ${launch_1v0_cmd[*]}"
  if [[ "$MODE" != "1v0" ]]; then
    echo "chain: WANDB_GROUP=$WANDB_GROUP $CHAIN"
  fi
  exit 0
fi

launch_1v0() {
  echo "Starting 1v0 (run_id=$RUN_ID, group=$WANDB_GROUP)"
  echo "Logs: $run_dir/run.log"
  "${launch_1v0_cmd[@]}" >>"$run_dir/run.log" 2>&1
}

case "$MODE" in
  1v0)
    launch_1v0
    ;;
  chain)
    exec env WANDB_GROUP="$WANDB_GROUP" "$CHAIN"
    ;;
  full)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "ERROR: tmux not found. Install tmux or use --1v0-only." >&2
      exit 1
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "ERROR: tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION" >&2
      exit 1
    fi
    SCRIPT="$ROOT/scripts/start_mac_training.sh"
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
      "bash -lc 'WANDB_GROUP=${WANDB_GROUP} exec \"${SCRIPT}\" --1v0-only; WANDB_GROUP=${WANDB_GROUP} exec \"${CHAIN}\"'"
    echo "Mac training launched in tmux session '$SESSION'."
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  1v0 log: tail -f $run_dir/run.log"
    echo "  Chain:   tail -f $ROOT/outputs/irl_training_chain.log"
    echo "  W&B:     https://wandb.ai (project f1tenth-genesis, group $WANDB_GROUP)"
    ;;
esac
