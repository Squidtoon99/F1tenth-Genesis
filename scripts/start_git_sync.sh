#!/usr/bin/env bash
# Launch the git sync watcher in a detached tmux session.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${GIT_SYNC_SESSION:-git_sync}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-900}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session '${SESSION}' already exists. Attach: tmux attach -t ${SESSION}" >&2
  exit 1
fi

mkdir -p outputs/runs/_git_sync

read -r -d '' INNER <<EOF || true
export POLL_INTERVAL_S=${POLL_INTERVAL_S}
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}"
cd "${ROOT}"
exec bash scripts/git_sync_watcher.sh
EOF

tmux new-session -d -s "${SESSION}" -c "${ROOT}" "bash -lc '${INNER}'"

echo "Git sync watcher launched in tmux session '${SESSION}'."
echo "  Attach:  tmux attach -t ${SESSION}"
echo "  Logs:    tail -f outputs/runs/_git_sync/watcher.log"
echo "  Stop:    tmux kill-session -t ${SESSION}"
