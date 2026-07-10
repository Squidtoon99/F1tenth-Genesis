#!/usr/bin/env bash
# Tear down the mapping stack we started on the car and restore the slam config.
echo "stopping our mapping nodes..."
pkill -f "f1tenth_mapping/lib/f1tenth_mapping/reactive_explorer" 2>/dev/null || true
pkill -f "bringup_nogap_launch" 2>/dev/null || true
pkill -f "slam_toolbox online_async_launch" 2>/dev/null || true
pkill -f "async_slam_toolbox_node" 2>/dev/null || true
pkill -f "urg_node_driver" 2>/dev/null || true
pkill -f "vesc_driver_node" 2>/dev/null || true
pkill -f "vesc_to_odom_node" 2>/dev/null || true
pkill -f "ackermann_to_vesc_node" 2>/dev/null || true
pkill -f "ackermann_mux" 2>/dev/null || true
pkill -f "joy_teleop" 2>/dev/null || true
pkill -f "joy_node" 2>/dev/null || true
pkill -f "static_baselink_to_laser" 2>/dev/null || true
sleep 2

# Restore the slam params file to the state captured before we edited it (mode swap).
CFG="$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml"
if [ -f /tmp/slam_cfg.bak ]; then
  cp /tmp/slam_cfg.bak "$CFG" && echo "restored slam config from /tmp/slam_cfg.bak"
fi

echo "--- remaining ros processes ---"
ps -eo pid,args | grep -E "[s]lam_toolbox|[u]rg_node|[v]esc|[a]ckermann_mux|[r]eactive_explorer|[j]oy_" | sed -E "s#--ros-args.*##; s#--params-file.*##" || echo "none"
