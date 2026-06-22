#!/usr/bin/env bash
# Relaunch the reactive_explorer on the car (faster speeds, follow left wall, live drive).
# Robust to flaky SSH: invoked as a single `bash ~/run_explorer.sh` call.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

# Stop any existing explorer first (single /drive publisher).
pkill -f "f1tenth_mapping/lib/f1tenth_mapping/reactive_explorer" 2>/dev/null || true
sleep 1

PARAMS="$HOME/f1tenth_ws/install/f1tenth_mapping/share/f1tenth_mapping/config/reactive_explorer_real.yaml"
FOLLOW_SIDE="${FOLLOW_SIDE:-left}"
MIN_SPEED="${MIN_SPEED:-1.0}"
CRUISE_SPEED="${CRUISE_SPEED:-1.8}"

setsid nohup ros2 run f1tenth_mapping reactive_explorer --ros-args \
  --params-file "$PARAMS" \
  -p follow_side:="$FOLLOW_SIDE" \
  -p dry_run:=false \
  -p min_speed_mps:="$MIN_SPEED" \
  -p cruise_speed_mps:="$CRUISE_SPEED" \
  -p map_save_path:="$HOME/maps/f1tenth_map" \
  > "$HOME/explorer.log" 2>&1 < /dev/null &

echo "launched reactive_explorer pid $! (follow=$FOLLOW_SIDE min=$MIN_SPEED cruise=$CRUISE_SPEED)"
