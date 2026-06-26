module load cuda/12.4
module load apptainer/1.3.4
module load gnu14/14.2.0

export FI_PROVIDER=tcp
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep MIG | head -n 1 | sed -E 's/.*(MIG-[^)]+).*/\1/')