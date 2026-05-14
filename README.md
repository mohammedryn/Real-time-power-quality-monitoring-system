# Real-Time Power Quality Monitoring System

A hardware-to-AI pipeline for real-time detection and classification of power quality disturbances. The system acquires dual-channel waveforms from a Teensy 4.1 microcontroller, extracts a 298-element phase-aware feature vector on-device, streams model-ready frames over USB, runs multi-label TFLite inference on a Raspberry Pi 5, and renders a live touch dashboard.

---

## Architecture

```
┌─────────────────────────────┐      USB / Serial       ┌────────────────────────────────────┐
│       Teensy 4.1            │ ─────────────────────── │       Raspberry Pi 5 (Host)         │
│                             │                         │                                    │
│  • Dual ADC @ 5 kHz         │   Model-ready frame     │  Acquisition loop                  │
│  • Zero-crossing trigger    │   (5204 bytes, CRC32)   │    └─ SerialFrameReceiver          │
│  • Goertzel + DWT DSP       │ ─────────────────────>  │  DSP / inference loop              │
│  • 298-feature extraction   │                         │    └─ FeatureExtractor             │
│  • CRC-framed USB stream    │                         │    └─ TFLitePredictor              │
└─────────────────────────────┘                         │  UI render loop (30 FPS)           │
                                                        │    └─ PySide6 / pyqtgraph          │
                                                        └────────────────────────────────────┘
```

The three loops communicate through bounded, thread-safe queues with a `drop_oldest` backpressure policy to keep the UI responsive under load.

---

## Signal Acquisition Contract

| Parameter              | Value                   |
|------------------------|-------------------------|
| Sampling rate          | 5000 Hz                 |
| Samples per frame      | 500                     |
| Mains frequency        | 50 Hz                   |
| Harmonic orders        | 1 – 13                  |
| ADC channels           | Voltage (A0), Current (A10) |
| Frame trigger          | Rising zero-crossing on voltage channel |
| Raw frame size         | 2012 bytes              |
| Model-ready frame size | 5204 bytes              |
| Frame magic            | `0xDEADBEEF` (4 B, big-endian) |
| CRC                    | CRC32 (little-endian), compatible with `binascii.crc32` |

---

## Feature Vector (298 elements)

| Slice       | Block               | Description                                         |
|-------------|---------------------|-----------------------------------------------------|
| `[0:12]`    | `time_v`            | 12 time-domain statistics for voltage               |
| `[12:24]`   | `time_i`            | 12 time-domain statistics for current               |
| `[24:28]`   | `power_metrics`     | Apparent power, active power, reactive power, PF    |
| `[28:56]`   | `mag_feats`         | 13 V magnitudes + 13 I magnitudes + THD-V + THD-I  |
| `[56:108]`  | `phase_self`        | sin / cos of all 13 V and I harmonic phases         |
| `[108:134]` | `phase_cross`       | sin / cos of per-harmonic V-I phase difference      |
| `[134:182]` | `phase_rel`         | sin / cos of phases relative to fundamental         |
| `[182:208]` | `power_harm`        | Interleaved active / reactive power per harmonic    |
| `[208:214]` | `circ_stats`        | Circular mean and std of V, I, cross phases         |
| `[214:256]` | `dwt_v`             | 36 standard + 6 transient-booster wavelet stats, V  |
| `[256:298]` | `dwt_i`             | 36 standard + 6 transient-booster wavelet stats, I  |

The model-ready frame splits the vector into three branches:

```
X_wave  = v_norm[500] + i_norm[500]     — normalized waveforms
X_mag   = feat[28:56]                   — 28 magnitude features
X_phase = feat[0:28] + feat[56:298]     — 270 phase-aware features
```

---

## ML Model

| Property           | Value                                              |
|--------------------|----------------------------------------------------|
| Architecture       | Phase-aware hybrid (3-branch multi-input network)  |
| Output semantics   | Multi-label (independent sigmoid per class)        |
| Classes (7)        | Normal, Sag, Swell, Interruption, HarmonicDistortion, Transient, Flicker |
| Production format  | TFLite (`.tflite`)                                 |
| Artifact path      | `artifacts/models/pqm_multilabel_model.tflite`     |
| Inference runtime  | `tflite_runtime`, `ai-edge-litert`, or `tensorflow.lite` (auto-detected) |

Per-class detection thresholds are defined in `configs/default.yaml` under `ml_inference.thresholds`.

---

## Repository Structure

```
firmware/
  teensy/pq_firmware/          Active firmware — Teensy 4.1 (PlatformIO)
  esp32p4/pq_firmware/         ESP32-P4 firmware port (set aside)

src/
  io/                          Serial receiver, frame protocol parser
  dsp/                         Feature extraction (features.py, wavelet_features.py)
  runtime/                     3-loop pipeline, bounded queues, metrics, TFLite predictor
  infer/                       Live inference and offline replay entrypoints
  ui/                          PySide6 touch dashboard (dashboard, events views)
  system/                      Kiosk setup script, systemd service, log rotation

configs/
  default.yaml                 All tuneable parameters (classes, calibration, thresholds)

artifacts/
  models/                      pqm_multilabel_model.tflite  (production)
                                pqm_multilabel_model.keras   (training checkpoint)
  live_sessions/               Per-session JSONL inference logs
  perf/                        Runtime and soak test reports

tests/                         63 unit and integration tests
scripts/                       Hardware validation, HIL comparison, demo helpers
docs/                          Deployment runbook, assembly guide, alignment matrix
legacy/                        Archived single-signal model experiments (reference only)
```

---

## Hardware Requirements

**Sensing node**
- Teensy 4.1
- Isolated analog front-end: AMC1301 differential isolation amplifier, TLV9001 op-amp
- Voltage divider: 2.2 MΩ / 560 Ω

**Compute and display**
- Raspberry Pi 5 (8 GB)
- Official Raspberry Pi touch display
- Active cooling (required for sustained operation)
- Stable 5 V supply rated for Pi 5 peak load

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd Real-Time-Power-Quality-Monitoring
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Verify installation
.venv/bin/python scripts/smoke_test.py
```

Python 3.10 or later is required.

---

## Running

### Live inference (production)

```bash
.venv/bin/python -m src.ui.app \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite \
  --model artifacts/models/pqm_multilabel_model.tflite
```

### Offline replay (no hardware required)

```bash
.venv/bin/python -m src.infer.offline_replay \
  --input artifacts/protocol_test_frames.bin \
  --config configs/default.yaml
```

### CLI inference (headless)

```bash
.venv/bin/python -m src.infer.live_infer \
  --port /dev/ttyACM0 \
  --receiver-mode tflite \
  --max-frames 200
```

---

## Firmware

### Teensy 4.1 (active)

```bash
# Compile
./scripts/compile_teensy_firmware.sh

# Flash
./scripts/flash_teensy_firmware.sh /dev/ttyACM0

# Raw ADC mode (for training data capture and HIL parity validation)
PQ_RAW_MODE=1 pio run -d firmware/teensy/pq_firmware -e teensy41_raw \
  -t upload --upload-port /dev/ttyACM0
```

Build flags:

| Flag                       | Effect                                          |
|----------------------------|-------------------------------------------------|
| `PQ_RAW_MODE=1`            | Emit 2012-byte raw ADC frame instead of model-ready frame |
| `PQ_DEBUG_TIMING=1`        | Print per-frame DSP and total latency over Serial |
| `PQ_FREE_RUN_FALLBACK=1`   | Free-run sampling if zero-crossing times out    |

### ESP32-P4 (set aside)

The ESP32-P4 port is fully implemented and available under `firmware/esp32p4/`. It emits the same model-ready frame format and can be substituted for the Teensy node without host-side changes.

```bash
./scripts/compile_esp32p4_firmware.sh
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
```

---

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

63 tests across protocol, feature extraction, runtime buffers, TFLite predictor, e2e pipeline, kiosk startup, and receiver resync.

---

## Kiosk Deployment (Raspberry Pi)

Copy the repository to `/opt/pq-monitor` on the target Pi, then run the installer:

```bash
cd /opt/pq-monitor
sudo ./src/system/kiosk_setup.sh \
  --repo /opt/pq-monitor \
  --user pi \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite \
  --model artifacts/models/pqm_multilabel_model.tflite
```

The installer installs and enables `pq-monitor.service`, which starts the dashboard on boot in fullscreen kiosk mode and auto-restarts on crash or serial disconnect.

**Service management**

```bash
sudo systemctl status pq-monitor.service --no-pager -n 50
journalctl -u pq-monitor.service -n 100 --no-pager
sudo logrotate -d /etc/logrotate.d/pq-monitor
```

Full deployment steps, pre-flight checklist, and thermal validation procedure are in [docs/pi_deployment_runbook.md](docs/pi_deployment_runbook.md).

---

## Hardware Validation

Capture host-vs-firmware feature parity (requires connected Teensy):

```bash
.venv/bin/python scripts/hil_compare_raw_feature.py \
  --port /dev/ttyACM0 \
  --frames 50 \
  --pairing anchor \
  --skip-prompts
```

Capture firmware timing from a `PQ_DEBUG_TIMING` build:

```bash
.venv/bin/python scripts/capture_teensy_timing.py \
  --port /dev/ttyACM0 \
  --seconds 30
```

---

## Configuration

All runtime parameters are in `configs/default.yaml`:

| Section            | Key fields                                            |
|--------------------|-------------------------------------------------------|
| `signal`           | `fs_hz`, `samples_per_frame`, `harmonic_orders`       |
| `calibration`      | ADC midpoints, counts-to-volts and counts-to-amps     |
| `features`         | Feature block toggles, DWT wavelet and level          |
| `ml_inference`     | Model path, receiver mode, per-class thresholds       |
| `runtime`          | Queue size, UI FPS target, inference Hz target        |
| `paths`            | Artifact and session log directories                  |

---

## Performance Targets

| Metric                       | Target          |
|------------------------------|-----------------|
| UI render rate               | >= 25 FPS       |
| Inference update rate        | >= 8 Hz         |
| End-to-end frame latency     | < 200 ms        |
| Startup time                 | <= 30 seconds   |
| Sustained operation          | 30+ minutes without thermal shutdown or UI freeze |

---

## Safety

This system interfaces with mains-voltage wiring. High-voltage (HV) and low-voltage (LV) compartments must be physically separated. Do not energize the sensing node without qualified supervision. Full wiring separation rules, insulation requirements, and pre-power-on checklists are in [docs/handheld_assembly.md](docs/handheld_assembly.md).
