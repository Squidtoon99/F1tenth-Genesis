#!/bin/bash
# Lonestar6 CPU (normal partition): self-play on reversed IRL f1tenth_map.
#
# Submit from repo root on ls6:
#   sbatch scripts/train_f1tenth_map_selfplay_ls6_cpu.slurm
#
# Resume: reuse RUN_ID so standalone_trainer picks up existing checkpoints in run_dir.

#SBATCH -J f1tenth_cpu_sp
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A ECS24012

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/F1tenth-Genesis}}"
VENV="${VENV:-${SCRATCH}/venvs/f1tenth-genesis}"
RUN_ID="${RUN_ID:-ls6_f1tenth_map_rev_selfplay_300k_v1}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
NUM_ENVS="${NUM_ENVS:-128}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-500000}"

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
  --hypothesis "LS6 normal CPU self-play, reversed f1tenth_map, ${TOTAL_STEPS} steps" \
  2>&1 | tee -a "outputs/standalone/logs/${RUN_ID}.log"
