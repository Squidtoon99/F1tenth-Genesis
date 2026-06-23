#!/usr/bin/env bash
# One-time Lonestar6 Python env on $SCRATCH (TACC recommends SCRATCH over $WORK).
#
# Usage (from an idev gpu-a100-dev session):
#   cd $SCRATCH
#   git clone <repo-url> F1tenth-Genesis && cd F1tenth-Genesis
#   bash scripts/setup_ls6_env.sh
#
# Then copy .env with WANDB_API_KEY, or run `wandb login` inside the venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${SCRATCH:?Set SCRATCH first}/venvs/f1tenth-genesis}"

source "$ROOT/scripts/hpc_setup_ls6.sh"

if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install -U pip wheel
# Match LS6 PyTorch docs (CUDA 12.8 wheels). Adjust if site modules require a
# different CUDA — run `python -c "import torch; print(torch.cuda.is_available())"`.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r "$ROOT/requirements.txt"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo ""
echo "Env ready. Activate with:"
echo "  source $VENV/bin/activate"
echo "Smoke test:"
echo "  cd $ROOT && python scripts/physics_check.py"
echo "  python scripts/bench_env.py --backend gpu --num-envs 256 512 --steps 200"
