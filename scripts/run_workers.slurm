#!/bin/bash
#SBATCH --job-name=genesis_workers
#SBATCH --partition=normal

#SBATCH --ntasks=4                 # 4 workers
#SBATCH --cpus-per-task=16         # 16 CPUs per worker
#SBATCH --mem=12G                  # 8 GB RAM per worker

#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs
module load miniconda/24.11.1
# Load shell + conda
source ~/.bashrc
conda activate genesis

# Go to your project
cd ~/work/F1tenth-Genesis

# Launch 4 workers in parallel (1 per task)
srun python worker.py
