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
./scripts/compile_teensy_firmware.sh
```

4. Confirm serial device path and permissions.

## Demo Sequence
1. Start UI (live mode):

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode feature
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
- UI open but no predictions: verify model/scaler paths or run without model for uniform fallback.
- High latency: inspect health panel and runtime queues.

## Required Demo Artifacts
- session log JSONL
- latest pytest output summary
- firmware build success log
- parity/timing artifacts if hardware session performed
