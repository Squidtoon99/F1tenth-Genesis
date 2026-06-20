#!/usr/bin/env bash
# Launch the overnight autonomous training orchestrator (nohup, survives logout).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/overnight

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# shellcheck source=/dev/null
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate genesis

set -a
# shellcheck source=/dev/null
source .env
set +a

GROUP="${WANDB_GROUP:-overnight_$(date +%Y-%m-%d)}"
MAX_RUNS="${MAX_RUNS:-6}"

echo "Starting overnight orchestrator (group=${GROUP}, max_runs=${MAX_RUNS})"
nohup python scripts/autonomous_train.py \
  --wait-for-trainer \
  --max-runs "${MAX_RUNS}" \
  --wandb-group "${GROUP}" \
  >> outputs/overnight/orchestrator.log 2>&1 &

echo "Orchestrator PID: $!"
echo "Monitor: tail -f outputs/overnight/orchestrator.log"
