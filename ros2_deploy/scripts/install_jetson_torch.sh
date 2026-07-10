#!/usr/bin/env bash
# Install CUDA-enabled PyTorch for Jetson (jp6/cu126) into user site-packages
# for the system Python used by f1tenth_rl_agent ROS nodes.
set -euo pipefail

echo "=== Platform ==="
/bin/uname -m
cat /etc/nv_tegra_release
dpkg-query --show nvidia-l4t-core
apt-cache show nvidia-jetpack 2>/dev/null | grep -E "^Version:" | head -1 || true
/usr/local/cuda/bin/nvcc --version | tail -1
python3 --version
which python3

echo "=== Current torch ==="
python3 -m pip show torch 2>&1 || true
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1 || true

echo "=== Uninstall existing torch wheels ==="
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "=== Pin numpy (<2) for Jetson torch ABI ==="
python3 -m pip install --user "numpy>=1.26,<2.0"

echo "=== Install Jetson CUDA torch (jp6/cu126) ==="
python3 -m pip install --user torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

echo "=== Verify torch + GPU MLP + latency ==="
python3 <<'PY'
import time

import torch
import torch.nn as nn

print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))

m = nn.Sequential(nn.Linear(387, 256), nn.ReLU(), nn.Linear(256, 2)).cuda().eval()
x = torch.randn(1, 387, device="cuda")
with torch.no_grad():
    y = m(x)
print("mlp_out", tuple(y.shape), y.device)

for _ in range(20):
    m(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 500
for _ in range(N):
    m(x)
torch.cuda.synchronize()
ms = 1000.0 * (time.perf_counter() - t0) / N
print(f"actor_fwd_latency_ms={ms:.3f}")
PY

python3 -m pip show torch | grep -E "^(Name|Version|Location)"
