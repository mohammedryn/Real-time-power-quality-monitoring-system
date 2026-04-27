# Real-Time Power Quality Monitoring

Non-ML integration repository for a Raspberry Pi kiosk that receives Teensy power-quality frames, runs live or replay inference, and displays a touch dashboard.

This repository preserves the fixed acquisition and feature contract used by the firmware and host runtime:

- Sampling rate: `fs = 5000 Hz`
- Samples per frame: `N = 500`
- Feature vector length: `282`
- Serial frame compatibility is defined in `src/io/frame_protocol.py`

## Environment Setup

```bash
cd /home/rayan/Real-Time-Power-Quality-Monitoring
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py
```

## Test Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## Firmware Compile Command

```bash
./scripts/compile_teensy_firmware.sh
```

## Live Run Command

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode feature
```

Optional model and scaler artifacts can be supplied when available:

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode feature \
  --model artifacts/models/pq_model.joblib \
  --scaler artifacts/scalers/pq_scaler.joblib
```

## Replay Fallback Command

Use this when live hardware is unavailable.

```bash
.venv/bin/python -m src.infer.offline_replay \
  --input artifacts/protocol_test_frames.bin \
  --config configs/default.yaml
```

## Kiosk Deployment Command

Run on the target Raspberry Pi after copying the repository to `/opt/pq-monitor` and creating the virtual environment there.

```bash
cd /opt/pq-monitor
sudo ./src/system/kiosk_setup.sh \
  --repo /opt/pq-monitor \
  --user pi \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode feature
```

With artifact paths:

```bash
cd /opt/pq-monitor
sudo ./src/system/kiosk_setup.sh \
  --repo /opt/pq-monitor \
  --user pi \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode feature \
  --model artifacts/models/pq_model.joblib \
  --scaler artifacts/scalers/pq_scaler.joblib
```

Service checks:

```bash
sudo systemctl status pq-monitor.service --no-pager -n 50
journalctl -u pq-monitor.service -n 100 --no-pager
sudo logrotate -d /etc/logrotate.d/pq-monitor
```

## Hardware Evidence Commands

Confirm the serial device:

```bash
ls -l /dev/ttyACM0
```

Capture host-vs-firmware feature parity:

```bash
.venv/bin/python scripts/hil_compare_raw_feature.py \
  --port /dev/ttyACM0 \
  --frames 50 \
  --pairing anchor \
  --skip-prompts
```

Capture firmware timing output from a `PQ_DEBUG_TIMING` build:

```bash
.venv/bin/python scripts/capture_teensy_timing.py \
  --port /dev/ttyACM0 \
  --seconds 30
```

Run the Pi kiosk soak:

```bash
sudo systemctl restart pq-monitor.service
sudo systemctl status pq-monitor.service --no-pager -n 50
journalctl -u pq-monitor.service -f
```
