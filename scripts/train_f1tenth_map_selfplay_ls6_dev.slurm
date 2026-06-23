#!/bin/bash
# Lonestar6 development (CPU): self-play on reversed IRL f1tenth_map.
#
# Submit from repo root on ls6:
#   sbatch scripts/train_f1tenth_map_selfplay_ls6_dev.slurm

#SBATCH -J f1tenth_dev_sp
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -A ECS24012

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/F1tenth-Genesis}}"
VENV="${VENV:-${SCRATCH}/venvs/f1tenth-genesis}"
RUN_ID="${RUN_ID:-ls6_f1tenth_map_rev_selfplay_100k_v1}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
NUM_ENVS="${NUM_ENVS:-64}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-200000}"

mkdir -p "$ROOT/logs" "$ROOT/outputs/standalone/logs"

source "$ROOT/scripts/hpc_setup_ls6.sh"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

cd "$ROOT"

python -u standalone_trainer.py \
  --track f1tenth_map \
  --backend cpu \
  --device cpu \
  --precision 32 \
  --num-envs "$NUM_ENVS" \
  --total-steps "$TOTAL_STEPS" \
  --min-train-samples 2000 \
  --batch-size 256 \
  --alpha 0.2 \
  --log-interval 250 \
  --ckpt-interval 10000 \
  --buffer-capacity "$BUFFER_CAPACITY" \
  --self-play \
  --run-id "$RUN_ID" \
  --wandb-group "${WANDB_GROUP:-ls6_f1tenth_map_rev_$(date +%Y-%m-%d)}" \
  --hypothesis "LS6 dev CPU self-play, reversed f1tenth_map centerline, 100k" \
  2>&1 | tee -a "outputs/standalone/logs/${RUN_ID}.log"
