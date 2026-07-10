#!/usr/bin/env bash
# Particle-filter localization against the surveyed real map.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "iv2026_localize_launch" 2>/dev/null || true
pkill -f "particle_filter" 2>/dev/null || true
pkill -f "nav2_map_server map_server" 2>/dev/null || true
sleep 1

MAP_YAML="${MAP_YAML:-$HOME/maps/f1tenth_map.yaml}"

setsid nohup ros2 launch f1tenth_stack iv2026_localize_launch.py \
  map_yaml:="$MAP_YAML" \
  > "$HOME/localization.log" 2>&1 < /dev/null &

echo "launched PF localization pid $! map=$MAP_YAML (log: ~/localization.log)"
echo "Next: bash ~/deploy/set_initial_pose.sh  (PF was at wrong frame without this)"
