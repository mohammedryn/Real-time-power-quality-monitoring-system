# Implementation Tasks (Chunked, End-to-End)

This plan is designed so any AI coding agent can execute the whole project with minimal ambiguity. Complete chunks in order. Do not move to the next chunk unless acceptance criteria pass.

## Global Rules
1. Preserve canonical class order: Normal, Sag, Swell, Interruption, Harmonic Distortion, Transient, Flicker.
2. Lock sampling assumptions: fs=5000 Hz, N=500 samples per frame.
3. Preserve serial frame protocol exactly as defined in PRD.
4. Add tests for every critical interface.
5. Save all outputs under artifacts with run IDs.
6. Preserve handheld target: Raspberry Pi 5 (8GB) + Pi display with touch-first UI.
7. Preserve real-time targets: responsive UI and non-blocking acquisition/inference.

## Chunk 0: Project Bootstrap and Cleanup
### Tasks
1. Create src-based structure from PRD.
2. Add requirements.txt or pyproject.toml with pinned versions.
3. Add configs/default.yaml including class map, fs, N, feature toggles, paths.
4. Normalize naming and path conventions.
5. Fix existing script path/import issues and keep legacy scripts in legacy/ if needed.

### Acceptance Criteria
1. Fresh environment install completes without dependency errors.
2. Python package imports resolve from project root.
3. Config file loads successfully in a smoke test.

### Deliverables
1. Standardized directory layout.
2. Dependency file.
3. Config baseline.

## Chunk 1: Protocol and Firmware Implementation
### Tasks
1. Create firmware file at firmware/teensy/pq_firmware/pq_firmware.ino.
2. Implement synchronized ADC reads for voltage and current channels.
3. Implement rising zero-crossing synchronized frame capture.
4. Implement 2012-byte frame with magic, seq, N, buffers, CRC32.
5. Add compile-time constants and comments for pin mappings.
6. Add host-side protocol validator script for recorded bytes.

### Acceptance Criteria
1. Firmware compiles for Teensy 4.1.
2. Host validator confirms frame length and CRC pass for 100+ frames.
3. Sequence numbers increment monotonically modulo uint16.

### Deliverables
1. pq_firmware.ino
2. src/io/frame_protocol.py
3. src/io/serial_receiver.py
4. tests/test_frame_protocol.py

## Chunk 2: Host IO and Preprocessing
### Tasks
1. Implement serial port open/read/reconnect with timeout handling.
2. Implement magic resync and partial-read recovery.
3. Convert ADC counts to physical units with config-based calibration constants.
4. Implement DC offset removal and normalization outputs.
5. Add frame logger writing raw+processed snapshots.

### Acceptance Criteria
1. Receiver runs continuously without crashes for 10 minutes.
2. Corrupt frame injection test drops bad frames without halting.
3. Preprocess outputs expected shapes and finite values.

### Deliverables
1. src/io/serial_receiver.py
2. src/dsp/preprocess.py
3. tests/test_receiver_resync.py
4. tests/test_preprocess.py

## Chunk 3: Report-Aligned Feature Extraction (282)
### Tasks
1. Implement time-domain feature extractor (24 total across V/I).
2. Implement harmonic magnitude+phase extraction for h=1..13 using N=500 bins.
3. Implement THD calculations for V and I.
4. Implement phase-aware feature block with sin/cos, wrapped deltas, V-I cross, coupling.
5. Implement circular statistics using scipy circmean/circstd.
6. Implement DWT features exactly sized to 72 across V/I.
7. Build final vector assembler and schema validator enforcing length 282.

### Acceptance Criteria
1. Unit test confirms feature length is exactly 282 for random valid frame.
2. Circular statistics tests pass edge case angles near +-pi.
3. Harmonic bin test confirms 50Hz->bin5 and 13th->bin65 at fs=5000,N=500.

### Deliverables
1. src/dsp/features.py
2. src/dsp/wavelet_features.py
3. tests/test_feature_shape.py
4. tests/test_circular_stats.py
5. tests/test_fft_bins.py

## Chunk 4: Synthetic Dataset Generator (Report-Compliant)
### Tasks
1. Implement balanced synthetic generator for 7 classes, 38,500 total samples.
2. Add physically motivated parameters from report per class.
3. Implement Harmonic Distortion subtypes with Von Mises phase distributions.
4. Implement current signal generation tied to class physics.
5. Implement noise augmentation options (SNR, jitter, phase offset, amplitude jitter).
6. Persist dataset metadata and deterministic split files.

### Acceptance Criteria
1. Dataset size and per-class counts exactly match target.
2. Split reproducibility validated with fixed random seed.
3. Metadata includes class id/name mapping and generation parameters.

### Deliverables
1. src/data/synthetic_generator.py
2. src/data/splits.py
3. artifacts/datasets/synth_v1/metadata.json
4. tests/test_dataset_balance.py

## Chunk 5: Model Implementations M1-M4
### Tasks
1. Implement M1 baseline magnitude MLP.
2. Implement M2 waveform-only model.
3. Implement M3 waveform+magnitude model.
4. Implement M4 full phase-aware hybrid model per report.
5. Add factory method to instantiate by variant id.
6. Add model summary export per variant.

### Acceptance Criteria
1. Each model builds and runs single forward pass with correct input tensors.
2. Input dimension checks fail fast with readable errors.
3. Parameter counts and branch shapes logged.

### Deliverables
1. src/models/m1_baseline.py
2. src/models/m2_waveform.py
3. src/models/m3_waveform_mag.py
4. src/models/m4_phase_aware.py
5. src/models/factory.py
6. tests/test_model_shapes.py

## Chunk 6: Training Pipeline and Artifacting
### Tasks
1. Implement unified training script with config-driven variant selection.
2. Add scaler fitting/saving for tabular branches.
3. Add callbacks: checkpoint, early stop, LR reduce, tensorboard optional.
4. Save run manifest with hashes, config snapshot, metrics.
5. Save training curves and confusion matrices automatically.

### Acceptance Criteria
1. End-to-end training run completes for at least one variant.
2. best_model and scaler files are saved and reloadable.
3. Training plot and confusion matrix files are generated.

### Deliverables
1. src/train/train.py
2. src/eval/evaluate.py
3. artifacts/runs/<run_id>/...
4. tests/test_train_smoke.py

## Chunk 7: Ablation Study Automation
### Tasks
1. Implement ablation runner for M1-M4 with shared splits.
2. Add optional modality subsets: V-only, I-only, V+I where relevant.
3. Aggregate metrics into a single table and plot.
4. Export markdown and csv report for ablation results.

### Acceptance Criteria
1. All variants train/evaluate under one command.
2. Aggregated results file exists and is human-readable.
3. M3 vs M4 comparison computed and highlighted.

### Deliverables
1. src/eval/ablation.py
2. artifacts/ablation/ablation_results.csv
3. artifacts/ablation/ablation_report.md

## Chunk 8: Real Data Capture and Domain Adaptation
### Tasks
1. Implement capture utility to label windows from live hardware sessions.
2. Store raw frame + processed waveform + label metadata.
3. Implement adaptation training script freezing backbone, tuning final layers.
4. Evaluate zero-shot vs adapted on held-out real set.
5. Produce comparison table and confusion matrices.

### Acceptance Criteria
1. Capture tool records labeled data without protocol loss.
2. Adaptation script runs and saves adapted checkpoint.
3. Before/after adaptation metrics artifact generated.

### Deliverables
1. src/data/real_capture.py
2. src/adapt/domain_adapt.py
3. artifacts/real_eval/before_after.csv
4. artifacts/real_eval/confusion_before.png
5. artifacts/real_eval/confusion_after.png

## Chunk 9: Live Inference Integration
### Tasks
1. Implement live inference pipeline from serial receiver to model output.
2. Add optional fallback mode using recorded frames.
3. Display class probabilities and top-1 prediction with timestamp.
4. Add rolling log writer for session analysis.

### Acceptance Criteria
1. Live script runs continuously and emits predictions per frame.
2. Latency target under 200 ms/frame on laptop CPU for inference path.
3. Session logs saved and reloadable.

### Deliverables
1. src/infer/live_infer.py
2. src/infer/offline_replay.py
3. artifacts/live_sessions/<session_id>.jsonl

## Chunk 9A: Handheld UI (Pi Display, Production-Grade)
### Tasks
1. Implement Pi-optimized touch UI using a high-performance plotting stack (PySide6 + pyqtgraph recommended).
2. Create dashboard view with real-time voltage/current waveform plots.
3. Add harmonic spectrum panel (orders 1-13 minimum).
4. Add class probability panel and top-1 class card.
5. Add metrics cards: RMS-V, RMS-I, THD-V, THD-I, PF/DPF where available, frequency estimate.
6. Add event timeline panel with event type, confidence, and timestamp.
7. Add system health panel: serial status, CRC fail count, dropped frames, inference latency, CPU/GPU temp.
8. Ensure responsive layout for Pi display resolution and touch interactions.

### Acceptance Criteria
1. UI renders smoothly on Pi 5 without blocking acquisition/inference.
2. Dashboard updates live values continuously with stable refresh.
3. All required parameters are visible and readable on the handheld display.

### Deliverables
1. src/ui/app.py
2. src/ui/views/dashboard.py
3. src/ui/views/events.py
4. src/ui/widgets/plots.py
5. assets/ui_theme/ (styles, icons, fonts)

## Chunk 9B: Real-Time Runtime Architecture and Performance Tuning
### Tasks
1. Implement 3-loop runtime architecture: acquisition loop, DSP/inference loop, UI render loop.
2. Add bounded queues/ring buffers between loops.
3. Add backpressure policy (drop oldest, keep latest) to prevent UI freeze.
4. Add latency instrumentation per stage and end-to-end.
5. Add frame pacing for UI target FPS.
6. Profile and optimize hot paths on Pi 5.

### Acceptance Criteria
1. System remains responsive during continuous run.
2. UI FPS and inference cadence meet configured targets under normal load.
3. No unbounded memory growth during 30-minute run.

### Deliverables
1. src/runtime/pipeline.py
2. src/runtime/buffers.py
3. src/runtime/metrics.py
4. artifacts/perf/pi_runtime_profile.md

## Chunk 9C: Device Packaging and Kiosk Deployment
### Tasks
1. Add kiosk startup for fullscreen app on boot.
2. Add systemd service for auto-start and restart-on-failure.
3. Add serial auto-reconnect and robust error-state UI overlays.
4. Add local log rotation and persistent storage policy.
5. Add thermal and power validation checklist for handheld operation.
6. Add handheld assembly guide with enclosure and wiring separation notes.

### Acceptance Criteria
1. Device boots directly into UI without manual shell commands.
2. App recovers automatically after crash or serial disconnect.
3. Operational logs persist across reboot and are bounded in size.

### Deliverables
1. src/system/kiosk_setup.sh
2. src/system/service/pq-monitor.service
3. docs/handheld_assembly.md
4. docs/pi_deployment_runbook.md

## Chunk 10: Verification, QA, and Demo Packaging
### Tasks
1. Create test matrix covering protocol, features, models, training, inference.
2. Add end-to-end integration test using synthetic frame stream.
3. Add runbook for demo day sequence and fallback procedure.
4. Update README with exact commands for full setup and run.
5. Ensure artifacts required by report are generated.
6. Add Pi handheld soak test (>=30 minutes) with performance and stability summary.

### Acceptance Criteria
1. Test suite passes locally.
2. End-to-end dry run completes from data generation to inference.
3. Demo runbook validated by a second person or clean environment.
4. Pi handheld run meets stability and responsiveness criteria.

### Deliverables
1. tests/test_e2e_pipeline.py
2. README.md updates
3. docs/demo_runbook.md
4. artifacts/final_summary.md
5. artifacts/perf/pi_soak_test_report.md

## Chunk 11: Alignment Closure Against Report
### Tasks
1. Build a section-by-section mapping report: report requirement -> code path -> artifact.
2. Mark each requirement as Implemented, Partial, or Deferred.
3. Resolve all Partial items that are in-scope.
4. Freeze version tag for submission.

### Acceptance Criteria
1. Every key report claim has code and artifact evidence.
2. No unresolved in-scope requirement remains.
3. Final handoff packet is complete for review.

### Deliverables
1. docs/report_alignment_matrix.md
2. release/submission_checklist.md
3. release/version_manifest.json

## Suggested Execution Commands (for AI agent)
1. Setup environment and install deps.
2. Run protocol tests.
3. Run synthetic dataset generation.
4. Run feature extractor tests.
5. Run M1-M4 ablation training.
6. Run evaluation and artifact export.
7. Run live/offline inference smoke tests.
8. Run report alignment generator.

## Final Handoff Checklist
1. All chunks complete with acceptance criteria passed.
2. Required artifacts exist and open correctly.
3. Config defaults point to valid paths.
4. README and runbook are executable as written.
5. Submission package includes alignment matrix and final metrics.
6. Handheld Pi build boots to UI and displays live PQ graphs and all required parameters.
