#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$REPO_ROOT/firmware/esp32p4/pq_firmware"

if ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF idf.py is not on PATH."
  echo "Run: . \$IDF_PATH/export.sh"
  exit 1
fi

idf.py -C "$FIRMWARE_DIR" set-target esp32p4
idf.py -C "$FIRMWARE_DIR" build

echo "ESP32-P4 firmware compile succeeded."
