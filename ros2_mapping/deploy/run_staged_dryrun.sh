#!/usr/bin/env bash
# Full staged bringup: sensors -> localization -> RL dry-run stack.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

DEPLOY="${DEPLOY:-$HOME/deploy}"
bash "$DEPLOY/run_teardown.sh" 2>/dev/null || true
bash "$DEPLOY/run_rl_teardown.sh" 2>/dev/null || true
sleep 2

bash "$DEPLOY/run_sensors.sh"
sleep 5
bash "$DEPLOY/run_localization.sh"
sleep 5
DRY_RUN=1 bash "$DEPLOY/run_rl_stack.sh"
sleep 8
bash "$DEPLOY/verify_rl_obs.sh"
