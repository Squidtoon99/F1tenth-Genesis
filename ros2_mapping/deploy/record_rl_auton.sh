#!/usr/bin/env bash
# Record RL obs/action/drive + teleop around manual -> auton handoff.
# See record_rl_auton.py for behavior.
set -euo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
exec python3 "${DEPLOY:-$HOME/deploy}/record_rl_auton.py" "$@"
