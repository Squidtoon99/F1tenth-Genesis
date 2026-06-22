#!/usr/bin/env bash
# Launch the 30-minute training supervisor in a detached tmux session.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${SUPERVISOR_SESSION:-training_supervisor}"
INTERVAL_S="${SUPERVISOR_INTERVAL_S:-1800}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session '${SESSION}' already exists. Attach: tmux attach -t ${SESSION}" >&2
  exit 1
fi

mkdir -p outputs/runs/_supervisor

read -r -d '' INNER <<EOF || true
export SUPERVISOR_INTERVAL_S=${INTERVAL_S}
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}"
cd "${ROOT}"
exec bash scripts/training_supervisor.sh
EOF

tmux new-session -d -s "${SESSION}" -c "${ROOT}" "bash -lc '${INNER}'"

echo "Training supervisor launched in tmux session '${SESSION}' (${INTERVAL_S}s cycle)."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Logs:    tail -f outputs/runs/_supervisor/supervisor.log"
echo "  State:   cat outputs/runs/_supervisor/state.json"
echo "  Stop:    tmux kill-session -t ${SESSION}"
