#!/usr/bin/env bash
# Lonestar6 (ls6) environment for standalone GPU training.
# Source from interactive idev sessions or SLURM job scripts:
#   source scripts/hpc_setup_ls6.sh
#
# Notes:
# - LS6 A100 nodes have 3 GPUs; default to GPU 0 for single-process training.
# - Do NOT use scripts/hpc_setup.sh here — that script targets Vista MIG GPUs.
set -euo pipefail

module load gcc/11.2 2>/dev/null || true
module load python3/3.9.7 2>/dev/null || module load python3 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${SLURM_JOB_ID:-local}}"
export PYGLET_HEADLESS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
