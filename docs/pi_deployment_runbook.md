# Raspberry Pi Deployment Runbook

## Target
- Device: Raspberry Pi 5 (8 GB)
- Runtime entrypoint: python -m src.ui.app
- Service unit: src/system/service/pq-monitor.service

## 1. System Preparation
1. Install Raspberry Pi OS (64-bit) and update packages.
2. Enable serial USB access and verify the active MCU appears as /dev/ttyACM0 (or note actual port).
3. Install Python 3.10+ and venv support.

## 2. Application Setup
1. Copy repository to /opt/pq-monitor.
2. Create environment and install dependencies:

```bash
cd /opt/pq-monitor
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Validate Python package imports:

```bash
.venv/bin/python scripts/smoke_test.py
```

## 3. Production Model Placement
Place the canonical teammate-provided TFLite artifact under `artifacts/models/pqm_multilabel_model.tflite`. Override the service model path only if you intentionally deploy a different `.tflite` artifact.

## 4. Install Kiosk Service
Run installer:

```bash
cd /opt/pq-monitor
sudo ./src/system/kiosk_setup.sh \
  --repo /opt/pq-monitor \
  --user pi \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite
```

## ESP32-P4 Pre-Live Validation

```bash
PQ_RAW_MODE=1 ./scripts/compile_esp32p4_firmware.sh
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python scripts/probe_esp32p4_raw.py --port /dev/ttyACM0 --frames 5
PQ_RAW_MODE=0 ./scripts/compile_esp32p4_firmware.sh
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite --max-frames 10
```

Proceed to the UI only after the CLI reports scored frames and zero CRC failures.

## 5. Verify Service Health
```bash
sudo systemctl status pq-monitor.service --no-pager -n 50
journalctl -u pq-monitor.service -n 100 --no-pager
```

Expected:
- service state is active (running)
- dashboard opens automatically on boot
- logs written to /var/log/pq-monitor

## 6. Log Rotation Validation
```bash
sudo logrotate -d /etc/logrotate.d/pq-monitor
```

## 7. Recovery Procedure
If service fails:
1. Check serial port name and update /etc/default/pq-monitor.
2. Re-run systemd reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart pq-monitor.service
```

3. Check pyqtgraph/pyside dependency issues in journal logs.

## 8. Hardware-Dependent Validation Commands
These require connected hardware. Use the ESP32-P4 path first; keep the Teensy path as a fallback/reference.

ESP32-P4 firmware build:
```bash
PQ_RAW_MODE=1 ./scripts/compile_esp32p4_firmware.sh
PQ_RAW_MODE=0 ./scripts/compile_esp32p4_firmware.sh
```

ESP32-P4 raw probe:
```bash
python scripts/probe_esp32p4_raw.py --port /dev/ttyACM0 --frames 5
```

Legacy Teensy firmware build:
```bash
./scripts/compile_teensy_firmware.sh
```

HIL parity capture:
```bash
.venv/bin/python scripts/hil_compare_raw_feature.py \
  --port /dev/ttyACM0 \
  --frames 50 \
  --pairing anchor \
  --skip-prompts
```

Timing capture (PQ_DEBUG_TIMING firmware build):
```bash
.venv/bin/python scripts/capture_teensy_timing.py \
  --port /dev/ttyACM0 \
  --seconds 30
```

## 9. Success Criteria
1. Boot-to-dashboard within 30 seconds.
2. UI remains responsive with sustained acquisition.
3. Service auto-recovers after unplug/replug of the active MCU.
4. Logs persist and remain bounded by rotation policy.
