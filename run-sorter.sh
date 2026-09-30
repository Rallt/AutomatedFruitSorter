#!/usr/bin/env bash
set -euo pipefail

# Start the dashboard with a camera IP or full stream URL.
# Usage: ./run-sorter.sh 10.11.221.17

CAMERA_INPUT="${1:-${FRUIT_SORTER_CAMERA_URL:-}}"
if [[ -z "$CAMERA_INPUT" ]]; then
  echo "Usage: $0 <camera-ip-or-stream-url>" >&2
  exit 2
fi

if [[ "$CAMERA_INPUT" == *"://"* ]]; then
  export FRUIT_SORTER_CAMERA_URL="$CAMERA_INPUT"
else
  export FRUIT_SORTER_CAMERA_URL="http://${CAMERA_INPUT}:8080/stream"
fi

echo "Starting fruit sorter with camera: $FRUIT_SORTER_CAMERA_URL"
exec uv run uvicorn API.SERVE:app --host 0.0.0.0 --port 8000
