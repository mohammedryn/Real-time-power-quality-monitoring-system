#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$REPO_ROOT/firmware/teensy/pq_firmware"

if ! command -v pio >/dev/null 2>&1; then
  echo "PlatformIO (pio) is not installed."
  echo "Install it first, then rerun this script."
  exit 1
fi

# Build firmware for Teensy 4.1 using platformio.ini in firmware directory.
pio run -d "$FIRMWARE_DIR"

echo "Firmware compile succeeded for PlatformIO env: teensy41"
