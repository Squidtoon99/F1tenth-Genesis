#!/usr/bin/env bash
# Flip legacy centerline CSV into map_server coordinates and verify on the PGM.
set -euo pipefail

MAPS="${1:-$HOME/maps}"
MAP_YAML="$MAPS/f1tenth_map.yaml"
IN_CSV="$MAPS/f1tenth_map_centerline.csv"
OUT_CSV="$MAPS/f1tenth_map_centerline_aligned.csv"
BACKUP="$MAPS/f1tenth_map_centerline_legacy.csv"

SCRIPT="$HOME/F1tenth-Genesis/ros2_mapping/postprocess/align_centerline_to_map.py"
if [ ! -f "$SCRIPT" ]; then
  SCRIPT="$HOME/deploy/align_centerline_to_map.py"
fi

if [ ! -f "$IN_CSV" ]; then
  echo "missing $IN_CSV"
  exit 1
fi

cp -n "$IN_CSV" "$BACKUP" 2>/dev/null || true
python3 "$SCRIPT" --map-yaml "$MAP_YAML" --in-csv "$IN_CSV" --out-csv "$OUT_CSV"
mv "$OUT_CSV" "$IN_CSV"
python3 "$HOME/deploy/verify_centerline_align.py" "$MAPS"
echo "aligned centerline installed at $IN_CSV (backup: $BACKUP)"
