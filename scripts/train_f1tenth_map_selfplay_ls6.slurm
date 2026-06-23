#!/bin/bash
# Lonestar6: self-play on the IRL f1tenth_map track (standalone_trainer.py).
#
# Submit from repo root on ls6:
#   mkdir -p logs
#   sbatch -A ECS24012 scripts/train_f1tenth_map_selfplay_ls6.slurm
#
# Interactive smoke (dev queue, 2 h max):
#   idev -p gpu-a100-dev -N 1 -n 1 -t 2:00:00 -A $TACC_ALLOCATION
#   source $SCRATCH/venvs/f1tenth-genesis/bin/activate
#   cd $SCRATCH/F1tenth-Genesis
#   bash scripts/train_f1tenth_map_selfplay_ls6.slurm   # runs without sbatch if not in job
#
# Tune NUM_ENVS after: python scripts/bench_env.py --backend gpu --num-envs 256 512 --steps 200

#SBATCH -J f1tenth_map_sp
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A ECS24012

set -euo pipefail

# Under sbatch the script is copied to spool; SLURM_SUBMIT_DIR is the cwd at submit time.
ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH}/F1tenth-Genesis}}"
VENV="${VENV:-${SCRATCH}/venvs/f1tenth-genesis}"
RUN_ID="${RUN_ID:-ls6_f1tenth_map_selfplay_500k_v1}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-512}"
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
  --backend gpu \
  --device cuda \
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
  --wandb-group "${WANDB_GROUP:-ls6_f1tenth_map_$(date +%Y-%m-%d)}" \
  --hypothesis "LS6 A100 self-play on IRL f1tenth_map" \
  2>&1 | tee -a "outputs/standalone/logs/${RUN_ID}.log"
