#!/usr/bin/env bash

# Usage:
#   ./tunnel.sh <remote_host>
#
# Behavior:
#   - Reverse forward remote:21812 -> local:21812
#   - Forward local:6379 -> remote:6379

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <remote_host>"
  exit 1
fi

REMOTE_HOST="$1"

# Reverb on 21812 -> Compute Node
# Redis on 6379 -> Local Machine 

ssh -N -R 21812:localhost:21812 -L 6379:localhost:6379 "$REMOTE_HOST"