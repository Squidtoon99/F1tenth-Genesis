#!/usr/bin/env bash
# Restart slam_toolbox (online async) on the car in MAPPING mode.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "slam_toolbox online_async_launch" 2>/dev/null || true
pkill -f "async_slam_toolbox_node" 2>/dev/null || true
sleep 2

PARAMS="$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml"
setsid nohup ros2 launch slam_toolbox online_async_launch.py slam_params_file:="$PARAMS" \
  > "$HOME/slam.log" 2>&1 < /dev/null &

echo "launched slam_toolbox pid $!"
