#!/usr/bin/env bash
# Periodic training health check (intended for tmux loop every 15 minutes).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== CHECKIN $(date -Is) ==="

echo "--- trainer ---"
pgrep -af "[p]ython standalone_trainer.py" || echo "no trainer"

echo "--- orchestrator ---"
pgrep -af "[p]ython scripts/autonomous_train.py" || echo "no orchestrator"

echo "--- latest run log ---"
latest=$(find outputs/runs -name run.log -not -path '*/_orchestrator/*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$latest" ]; then
  echo "log: $latest"
  grep -a "buffer=" "$latest" | tail -1 || true
  grep -a "selfplay:" "$latest" | tail -1 || true
  echo "NaN_warnings=$(grep -ac 'NaN constraint forces' "$latest" 2>/dev/null || echo 0)"
else
  echo "no run.log found"
fi

echo "--- orchestrator tail ---"
tail -3 outputs/runs/_orchestrator/orchestrator.log 2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- GPU ---"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
fi

echo "=== END CHECKIN ==="
