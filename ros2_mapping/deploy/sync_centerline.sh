#!/usr/bin/env bash
# Install map_server-aligned centerline on the car (~/maps/f1tenth_map_centerline.csv).
set -euo pipefail

MAPS="${1:-$HOME/maps}"
mkdir -p "$MAPS"
BACKUP="$MAPS/f1tenth_map_centerline_legacy.csv"

SRC=""
for candidate in \
  "$HOME/F1tenth-Genesis/ros2_deploy/assets/f1tenth_map_centerline.csv" \
  "$HOME/deploy/f1tenth_map_centerline.csv" \
  "$HOME/F1tenth-Genesis/ros2_mapping/output/real_map/clean/f1tenth_map_centerline.csv"; do
  if [ -f "$candidate" ]; then
    SRC="$candidate"
    break
  fi
done

if [ -z "$SRC" ]; then
  echo "ERROR: aligned centerline CSV not found"
  exit 1
fi

if [ -f "$MAPS/f1tenth_map_centerline.csv" ]; then
  cp -n "$MAPS/f1tenth_map_centerline.csv" "$BACKUP" 2>/dev/null || true
fi
cp "$SRC" "$MAPS/f1tenth_map_centerline.csv"
echo "installed $MAPS/f1tenth_map_centerline.csv from $SRC"

VERIFY="$HOME/F1tenth-Genesis/ros2_mapping/deploy/verify_centerline_align.py"
[ -f "$VERIFY" ] || VERIFY="$HOME/deploy/verify_centerline_align.py"
if [ -f "$VERIFY" ]; then
  python3 "$VERIFY" "$MAPS"
fi
