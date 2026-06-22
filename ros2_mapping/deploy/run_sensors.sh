#!/usr/bin/env bash
# Start f1tenth_stack sensors + teleop + mux (no gap driver, no SLAM).
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "bringup_nogap_launch" 2>/dev/null || true
pkill -f "bringup_launch.py" 2>/dev/null || true
sleep 1

DEPLOY="$HOME/F1tenth-Genesis/ros2_mapping/deploy"
if [ ! -f "$DEPLOY/bringup_nogap_launch.py" ]; then
  DEPLOY="$HOME/deploy"
fi

setsid nohup ros2 launch "$DEPLOY/bringup_nogap_launch.py" \
  > "$HOME/sensors.log" 2>&1 < /dev/null &

echo "launched sensors pid $! (log: ~/sensors.log)"
