#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
DEPLOY="${DEPLOY:-$HOME/deploy}"
# CPU torch by default (Jetson GPU torch deferred; small MLP runs fine on CPU).
DEVICE="${DEVICE:-cpu}"
if ! python3 -c "import torch" 2>/dev/null; then
  echo "BLOCKED: torch not importable. Run: python3 -m pip install --index-url https://pypi.org/simple torch"
  exit 1
fi
if ! ros2 topic list 2>/dev/null | grep -qx /pf/pose/odom; then
  bash "$DEPLOY/run_localization.sh"
  sleep 4
  bash "$DEPLOY/set_initial_pose.sh"
  sleep 3
fi
export LIVE=1
export LIVE_SPEED_LIMIT="${LIVE_SPEED_LIMIT:-2.0}"
export LIVE_MIN_SPEED="${LIVE_MIN_SPEED:-1.0}"
export AGENT_PARAMS="$DEPLOY/agent_1v1_detector.yaml"
export CKPT="${CKPT:-$HOME/checkpoints/ckpt_500000.pt}"
TMP_AGENT="$(mktemp /tmp/agent_live_XXXXXX.yaml)"
sed "s/device: \"cpu\"/device: \"${DEVICE}\"/" "$AGENT_PARAMS" > "$TMP_AGENT"
export AGENT_PARAMS="$TMP_AGENT"
bash "$DEPLOY/run_rl_stack.sh"
echo "GO LIVE: deadman btn 5 = teleop override; release for RL at ${LIVE_SPEED_LIMIT} m/s"
