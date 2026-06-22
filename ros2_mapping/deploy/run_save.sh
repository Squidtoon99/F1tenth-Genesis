#!/usr/bin/env bash
# Stop the reactive_explorer and save slam_toolbox's /map to ~/maps/f1tenth_map.{pgm,yaml}.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

# Stop the driver first so the car halts before we save.
pkill -f "f1tenth_mapping/lib/f1tenth_mapping/reactive_explorer" 2>/dev/null || true
sleep 1

mkdir -p "$HOME/maps"
OUT="${OUT:-$HOME/maps/f1tenth_map}"
echo "Saving /map -> ${OUT}.pgm / ${OUT}.yaml"
ros2 run nav2_map_server map_saver_cli -f "$OUT" \
  --ros-args -p map_subscribe_transient_local:=true -p save_map_timeout:=10000.0
echo "--- maps dir ---"
ls -la "$HOME/maps/"
