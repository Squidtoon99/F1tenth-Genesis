#!/usr/bin/env bash
# Launch the on-car 1v1 RL stack (vehicle_obs + opponent_detector + policy + drive).
# Usage:
#   DRY_RUN=1 ./run_rl_stack.sh   # speed_limit_mps=0 (no motion)
#   LIVE=1 ./run_rl_stack.sh      # speed_limit=2.0, min_speed=1.0
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

pkill -f "bringup_vehicle.launch.py" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle vehicle_obs" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle opponent_detector" 2>/dev/null || true
pkill -f "f1tenth_rl_vehicle drive" 2>/dev/null || true
pkill -f "f1tenth_rl_agent policy_inference" 2>/dev/null || true
sleep 1

DEPLOY="${DEPLOY:-$HOME/deploy}"
PARAMS="${PARAMS:-$DEPLOY/rl_real.yaml}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  PARAMS="${PARAMS_DRY:-$DEPLOY/rl_real_dryrun.yaml}"
  echo "DRY_RUN: using $PARAMS (speed_limit_mps=0)"
fi
if [ "${LIVE:-0}" = "1" ]; then
  PARAMS="${PARAMS:-$DEPLOY/rl_real.yaml}"
  echo "LIVE: speed_limit=${LIVE_SPEED_LIMIT:-2.0} min_speed=${LIVE_MIN_SPEED:-1.0}"
fi
CKPT="${CKPT:-$HOME/checkpoints/ckpt_500000.pt}"
TRACK="${TRACK:-$HOME/maps/f1tenth_map_centerline.csv}"
AGENT_PARAMS="${AGENT_PARAMS:-$DEPLOY/agent_1v1_detector.yaml}"
if [ ! -f "$AGENT_PARAMS" ]; then
  AGENT_PARAMS="$(ros2 pkg prefix f1tenth_rl_agent)/share/f1tenth_rl_agent/config/agent_1v1.yaml"
fi

setsid nohup ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \
  params_file:="$PARAMS" \
  agent_params_file:="$AGENT_PARAMS" \
  checkpoint_path:="$CKPT" \
  track_csv:="$TRACK" \
  enable_opponent:=true \
  > "$HOME/rl_stack.log" 2>&1 < /dev/null &

echo "launched RL stack pid $! params=$PARAMS (log: ~/rl_stack.log)"
