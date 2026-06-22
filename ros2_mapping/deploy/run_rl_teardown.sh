#!/usr/bin/env bash
# Stop RL deploy stack (keep sensors/localization if desired).
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "bringup_vehicle.launch.py" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle vehicle_obs" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle opponent_detector" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle drive" 2>/dev/null || true
pkill -f "f1tenth_rl_agent policy_inference" 2>/dev/null || true
sleep 1
echo "RL stack stopped"
