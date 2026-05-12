# Demo Runbook

## Objective
Execute an end-to-end demonstration from frame ingestion to live classification output with fallback options.

## Pre-Demo Checklist
1. Verify dependencies and environment:

```bash
.venv/bin/python scripts/smoke_test.py
```

2. Run automated tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

3. Verify firmware build:

```bash
PQ_RAW_MODE=1 ./scripts/compile_esp32p4_firmware.sh
```

4. Confirm serial device path and permissions.

5. Validate raw frame transport before the demo:

```bash
python scripts/probe_esp32p4_raw.py --port /dev/ttyACM0 --frames 5
python -m src.infer.live_infer \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode raw \
  --max-frames 10
PQ_RAW_MODE=0 ./scripts/compile_esp32p4_firmware.sh
```

## Demo Sequence
1. Start UI (live mode):

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite
```

2. Explain panels:
- waveforms (voltage/current)
- harmonic spectrum (orders 1..13)
- class probabilities and top-1
- metrics cards (RMS, THD, PF/DPF, frequency)
- event timeline and system health

3. Trigger known disturbance and observe event entry.

## Fallback Paths
If live hardware is unavailable:
1. Run replay mode:

```bash
.venv/bin/python -m src.infer.offline_replay \
  --input artifacts/protocol_test_frames.bin \
  --config configs/default.yaml
```

2. Present session log output from artifacts/live_sessions.

## Troubleshooting During Demo
- No serial frames: verify /dev/ttyACM0 and receiver mode.
- UI open but no predictions: verify the TFLite model path or run without overriding the default artifact path.
- High latency: inspect health panel and runtime queues.

## Required Demo Artifacts
- session log JSONL
- latest pytest output summary
- firmware build success log
- parity/timing artifacts if hardware session performed
- ESP32-P4 raw probe output when hardware session performed
