#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$REPO_ROOT/firmware/teensy/pq_firmware"

if ! command -v pio >/dev/null 2>&1; then
  echo "PlatformIO (pio) is not installed."
  echo "Install it first, then rerun this script."
  exit 1
fi

pio run -d "$FIRMWARE_DIR" -t upload --upload-port "$PORT"

echo "Teensy firmware flash succeeded."
