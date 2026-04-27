# Teensy DSP Migration Tracker

## Purpose
This document tracks the migration of feature extraction DSP from Pi host-side execution to Teensy 4.1 firmware, including rationale, implementation status, and validation evidence.

## Snapshot
- Date: 2026-04-27
- Migration status: In progress, software path complete, pending hardware evidence capture
- Primary goal: Teensy computes full 282-feature vector, Pi consumes feature vector directly for scaler + model inference

## Architecture Delta
### Before
- Teensy sent raw ADC window frame (2012 bytes): [MAGIC][seq][n=500][v_raw][i_raw][CRC32]
- Pi performed preprocess + feature extraction

### After
- Teensy sends feature frame (1140 bytes): [MAGIC][seq][n=282][features float32][CRC32]
- Pi receives features and can directly feed scaler + model
- Raw mode still available for fallback, parity, and capture workflows

### Why this change
- Lower host CPU load for real-time UI + inference
- Lower USB payload size and parser overhead
- Clear hardware/software boundary: firmware handles deterministic DSP, host handles ML/runtime orchestration

## Implemented Changes and Rationale

### 1) Firmware: DSP on Teensy
- Added full DSP feature extractor and supporting math/wavelet modules:
  - [../firmware/teensy/pq_firmware/src/dsp.h](../firmware/teensy/pq_firmware/src/dsp.h)
  - [../firmware/teensy/pq_firmware/src/dsp.cpp](../firmware/teensy/pq_firmware/src/dsp.cpp)
  - [../firmware/teensy/pq_firmware/src/goertzel.h](../firmware/teensy/pq_firmware/src/goertzel.h)
  - [../firmware/teensy/pq_firmware/src/dwt.h](../firmware/teensy/pq_firmware/src/dwt.h)
- Integrated feature frame transmission and mode flags in:
  - [../firmware/teensy/pq_firmware/src/main.cpp](../firmware/teensy/pq_firmware/src/main.cpp)
- Added compile-time controls:
  - `PQ_RAW_MODE` for raw frame fallback
  - `PQ_DEBUG_TIMING` for timing telemetry

Why this way:
- `PQ_RAW_MODE` enables A/B parity validation and safe rollback.
- `PQ_DEBUG_TIMING` allows direct measurement of DSP and total frame timing budget on hardware.
- Goertzel avoids full FFT overhead for fixed harmonic bins.
- Firmware-side DWT and stats keep host runtime lightweight.

### 2) Protocol: mixed raw + feature frame support
- Extended protocol constants, feature dataclass, and parser support:
  - [../src/io/frame_protocol.py](../src/io/frame_protocol.py)
- Added variable-length iterator with header peek (`n` dispatch) and resync behavior for unknown frame types.

Why this way:
- Supports mixed streams during migration and debugging.
- Preserves backward compatibility with existing raw frame tooling.
- Improves robustness in noisy streams by explicit resynchronization behavior.

### 3) Receiver: feature mode support in runtime and CLI
- Added receiver mode selection (`raw` or `feature`) and frame-size aware parsing:
  - [../src/io/serial_receiver.py](../src/io/serial_receiver.py)
- Added feature stream recording path (`--mode feature`) in CLI.

Why this way:
- Enables direct feature pipeline capture for inference and diagnostics.
- Keeps existing raw and snapshot workflows available.

### 4) Live feature path switched to direct FeatureFrame consumption
- Updated live feature demo script to default to feature receiver mode:
  - [../scripts/live_features_demo.py](../scripts/live_features_demo.py)
- Added explicit `--receiver-mode` switch (`feature` default, `raw` fallback).

Why this way:
- Uses the intended low-latency production path by default.
- Retains host-DSP fallback for troubleshooting.

### 5) DWT boundary alignment
- Set explicit PyWavelets mode to periodization in:
  - [../src/dsp/wavelet_features.py](../src/dsp/wavelet_features.py)

Why this way:
- Removes ambiguity in boundary handling.
- Improves parity consistency between Python and firmware DWT implementations.

### 6) Canonical feature index and tolerances
- Added central feature index map and parity tolerance grouping:
  - [../src/dsp/feature_index.py](../src/dsp/feature_index.py)

Why this way:
- Prevents silent feature order drift across firmware, Python, tests, and future model code.
- Makes parity validation reproducible and auditable.

### 7) Calibration alignment
- Firmware calibration constants aligned to config baseline in:
  - [../firmware/teensy/pq_firmware/src/dsp.h](../firmware/teensy/pq_firmware/src/dsp.h)
- Config baseline remains:
  - [../configs/default.yaml](../configs/default.yaml)
- Parity tests now load calibration/signal constants from config:
  - [../tests/test_teensy_dsp_parity.py](../tests/test_teensy_dsp_parity.py)

Why this way:
- Single source of truth avoids host/firmware scaling drift.
- Keeps physical-unit feature interpretation consistent across environments.

### 8) Tests and stress coverage
- Added protocol feature-frame tests:
  - [../tests/test_feature_frame_protocol.py](../tests/test_feature_frame_protocol.py)
- Added mixed-stream corruption and stress tests:
  - [../tests/test_feature_frame_stress.py](../tests/test_feature_frame_stress.py)
- Added parity harness (Python reference of firmware logic):
  - [../tests/test_teensy_dsp_parity.py](../tests/test_teensy_dsp_parity.py)

Why this way:
- Protocol correctness and stream resilience are critical for unattended runtime.
- Parity tests catch numerical and ordering regressions early.

### 9) Hardware-in-loop comparison workflow
- Added parity logging script for raw-mode vs feature-mode capture and slice-wise comparison:
  - [../scripts/hil_compare_raw_feature.py](../scripts/hil_compare_raw_feature.py)
- Script outputs:
  - raw_mode_features.npy
  - feature_mode_vectors.npy
  - summary.json
  - slice_metrics.csv
  - pairwise_slice_metrics.csv
  - distribution_slice_metrics.csv
  - pairing_matches.csv
  - baseline_raw_drift_slice_metrics.csv (when baseline drift capture enabled)

Additional methodology upgrades:
- Optional anchor-based cross-run pairing (`--pairing anchor`) using stable physics anchors (V_RMS, I_RMS, THD_V, THD_I) to reduce non-contemporaneous pairing noise.
- Optional baseline raw drift run (`--baseline-raw-drift`) to estimate natural source/load variation floor.
- Separate reporting for:
  - pairwise errors (paired-frame deltas)
  - distribution-level drift (population statistics independent of pairing)

Why this way:
- Provides objective hardware evidence that MCU-transmitted features align with host recomputation.
- Avoids over-interpreting frame-by-frame deltas from non-contemporaneous captures.
- Produces artifacts suitable for review and regression tracking.

### 10) Hardware timing capture workflow
- Added timing capture utility for `PQ_DEBUG_TIMING` serial output:
  - [../scripts/capture_teensy_timing.py](../scripts/capture_teensy_timing.py)
- Script outputs:
  - timing_raw.log
  - timing_samples.csv
  - summary.json

Why this way:
- Standardizes DSP and end-to-end timing evidence collection.
- Produces directly reviewable latency metrics (mean, p95, max).

## Verification Log (latest)
- Full test suite: pass (59 passed)
- New migration test subset: pass (41 passed)
- Firmware compile: pass (`pio run` via compile script)
- Raw artifact protocol validator: pass
- HIL script import robustness: pass (direct script execution no longer requires manual `PYTHONPATH`)

Hardware blocker evidence (current environment):
- HIL parity command attempted:
  - `.venv/bin/python scripts/hil_compare_raw_feature.py --port /dev/ttyACM0 --frames 2 --timeout 0.2 --skip-prompts --no-baseline-raw-drift`
  - Result: `SerialException`, `/dev/ttyACM0` not present
- Device listing confirmed no Teensy serial node:
  - `ls /dev/tty*` (no `/dev/ttyACM0`)

Note:
- Known non-fatal warnings remain for precision-loss in wavelet skew/kurtosis on near-identical coefficients.

## Tracking Checklist
- [x] Teensy feature extraction modules integrated
- [x] Feature frame protocol implemented
- [x] Receiver runtime feature mode implemented
- [x] Receiver CLI feature mode implemented
- [x] Live feature script default switched to feature path
- [x] DWT boundary mode aligned to periodization
- [x] Canonical feature index map added
- [x] Calibration constants aligned across firmware/config
- [x] Protocol and stream stress tests added
- [x] Parity harness tests added
- [x] HIL parity logging script added
- [ ] Capture and archive hardware timing logs from `PQ_DEBUG_TIMING` runs (blocked: hardware unavailable in this environment)
- [ ] Execute full HIL parity run on stable source and archive artifacts (blocked: hardware unavailable in this environment)

## Standard HIL Procedure
1. Flash firmware with `PQ_RAW_MODE=1`.
2. Run [../scripts/hil_compare_raw_feature.py](../scripts/hil_compare_raw_feature.py) and capture raw-mode feature recomputation.
3. Flash firmware with default feature mode (`PQ_RAW_MODE=0`).
4. Continue script capture for direct feature frames.
5. Review output artifacts under `artifacts/hil_parity` and track regressions over time.

## How to maintain this tracker
For each new migration change, add:
- What changed
- Which files were touched
- Why this design was chosen
- Validation command(s) and outcome
- Any known residual risk
