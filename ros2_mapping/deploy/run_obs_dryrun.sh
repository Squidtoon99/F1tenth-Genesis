#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
DEPLOY="${DEPLOY:-$HOME/deploy}"
PARAMS="${PARAMS:-$DEPLOY/rl_real_dryrun.yaml}"
TRACK="${TRACK:-$HOME/maps/f1tenth_map_centerline.csv}"
pkill -f "f1tenth_rl_vehicle vehicle_obs" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle opponent_detector" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle drive" 2>/dev/null || true
pkill -f "f1tenth_rl_agent policy_inference" 2>/dev/null || true
sleep 1
setsid nohup ros2 run f1tenth_rl_vehicle vehicle_obs --ros-args --params-file "$PARAMS" -p track_csv:="$TRACK" > "$HOME/vehicle_obs.log" 2>&1 < /dev/null &
setsid nohup ros2 run f1tenth_rl_vehicle opponent_detector --ros-args --params-file "$PARAMS" -p track_csv:="$TRACK" > "$HOME/opponent_detector.log" 2>&1 < /dev/null &
echo "GPU-free obs dry-run started (vehicle_obs + opponent_detector)"
