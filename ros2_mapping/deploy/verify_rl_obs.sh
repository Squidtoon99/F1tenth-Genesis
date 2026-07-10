#!/usr/bin/env bash
# Verify RL topics + obs_norm load after stack is up. Retries on flaky SSH/ros.
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

wait_topic() {
  local topic="$1"
  local tries="${2:-20}"
  for ((i=1; i<=tries; i++)); do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "=== waiting for core topics ==="
for t in /scan /odom /pf/pose/odom /rl/observation /rl/action /drive; do
  if wait_topic "$t" 30; then
    echo "OK $t"
  else
    echo "MISSING $t"
  fi
done

echo "=== hz checks (3s) ==="
timeout 5 ros2 topic hz /rl/observation 2>&1 | tail -3 || true
timeout 5 ros2 topic hz /rl/action 2>&1 | tail -3 || true

echo "=== observation sample ==="
OBS=$(timeout 3 ros2 topic echo /rl/observation --once 2>/dev/null)
echo "$OBS" | head -8
python3 - <<'PY'
import os
text = os.environ.get("OBS", "")
vals = [float(l.strip()[2:]) for l in text.splitlines() if l.strip().startswith("- ")]
if len(vals) >= 387:
    print(f"obs dim={len(vals)} ey={vals[10]:.3f} contact={vals[11]:.0f} opp_block={vals[380:387]}")
else:
    print(f"obs dim={len(vals)} (expect 387)")
PY

echo "=== policy log (obs_norm) ==="
grep -E "Loaded policy|Loaded obs_norm|Failed" "$HOME/rl_stack.log" 2>/dev/null | tail -5 || true

echo "=== drive sample ==="
timeout 3 ros2 topic echo /drive --once 2>/dev/null | head -12 || true

echo "=== pf pose sample ==="
timeout 3 ros2 topic echo /pf/pose/odom --once 2>/dev/null | head -20 || true
