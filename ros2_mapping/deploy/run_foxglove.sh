#!/usr/bin/env bash
# Foxglove WebSocket bridge (replaces RViz/noVNC for visualization).
# In Foxglove Studio: Open connection -> Foxglove WebSocket -> ws://<car-ip>:8765
set -euo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "foxglove_bridge" 2>/dev/null || true
sleep 1

PORT="${PORT:-8765}"
setsid nohup ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
  port:="${PORT}" \
  > "$HOME/foxglove_bridge.log" 2>&1 < /dev/null &

IP=$(hostname -I | awk '{print $1}')
echo "foxglove_bridge pid $! port=${PORT} (log: ~/foxglove_bridge.log)"
echo "Foxglove Studio: ws://${IP}:${PORT}"
