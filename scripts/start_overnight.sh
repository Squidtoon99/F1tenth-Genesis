#!/usr/bin/env bash
# Launch the overnight autonomous training orchestrator inside a detached tmux
# session so it survives the launching shell / SSH / agent session ending.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/runs/_orchestrator

SESSION="${TMUX_SESSION:-overnight}"
GROUP="${WANDB_GROUP:-overnight_$(date +%Y-%m-%d)}"
MAX_RUNS="${MAX_RUNS:-6}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found; install tmux or launch the orchestrator manually." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}" >&2
  echo "       Kill it first with: tmux kill-session -t ${SESSION}" >&2
  exit 1
fi

# The orchestrator already refuses to start a run while another trainer is up
# (--wait-for-trainer), but guard against a stray trainer from a prior session.
if pgrep -f "standalone_trainer.py" >/dev/null 2>&1; then
  echo "WARNING: a standalone_trainer.py is already running; the orchestrator will" >&2
  echo "         wait for it to exit before launching a new run." >&2
fi

# Build the command the tmux session runs. Conda + .env are sourced *inside* the
# session so the environment is correct regardless of how this script was invoked.
read -r -d '' INNER <<EOF || true
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}"
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate genesis
set -a
source .env
set +a
echo "Starting overnight orchestrator (group=${GROUP}, max_runs=${MAX_RUNS})"
exec python scripts/autonomous_train.py \
  --wait-for-trainer \
  --max-runs "${MAX_RUNS}" \
  --wandb-group "${GROUP}" \
  >> outputs/runs/_orchestrator/orchestrator.log 2>&1
EOF

tmux new-session -d -s "${SESSION}" -c "${ROOT}" "bash -lc '${INNER}'"

echo "Orchestrator launched in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Logs:    tail -f outputs/runs/_orchestrator/orchestrator.log"
echo "  Stop:    tmux kill-session -t ${SESSION}"
