# Product Requirements Document (PRD)

## 1. Product Name
Phase-Aware Real-Time Power Quality Monitoring System

## 2. Purpose
Build an end-to-end, low-cost, hardware-enabled system that measures 230V/50Hz power waveforms, extracts phase-aware harmonic features, and classifies power quality disturbances in real time.

This implementation must match the technical intent of the project report in [PQ_Monitor_Complete_Report.md](PQ_Monitor_Complete_Report.md), while converting it into working code, firmware, training artifacts, and validation outputs.

## 3. Problem Statement
Current low-cost systems and most standard tools rely mostly on harmonic magnitudes and ignore harmonic phase structure. This loses discriminative information and limits disturbance/source classification quality.

The product must prove that phase-aware features improve classification performance, and that the system works from acquisition to inference, not only on synthetic data but also with real captured data.

## 4. Goals
1. Deliver working hardware-to-AI pipeline from Teensy frame capture to classified output.
2. Implement the report-aligned DSP stack with N=500, fs=5000, and 282-feature vector.
3. Train and evaluate model variants M1-M4 with reproducible experiments.
4. Close synthetic-to-real gap with domain adaptation workflow.
5. Produce demo-ready outputs and reproducible scripts for evaluation.
6. Deploy as a handheld Raspberry Pi 5 (8GB) device with real-time touch UI on Pi display.

## 5. Non-Goals
1. IEC certification or regulatory approval.
2. Three-phase production deployment.
3. Cloud backend or web app dashboard unless needed for demo support.

## 6. Users
1. Project team members (development and testing).
2. Evaluators/reviewers (academic assessment).
3. Future contributors using AI coding agents.

## 7. Scope
### In Scope
1. Teensy firmware for synchronized dual ADC capture and CRC-framed USB streaming.
2. Python receiver and preprocessing for binary frame protocol.
3. Report-aligned synthetic dataset generation (38,500 samples, 7 classes).
4. Feature extraction pipeline: time-domain + FFT harmonic + phase-aware + DWT = 282 features.
5. Model training for M1-M4 and evaluation scripts.
6. Domain adaptation on real data (100-200 samples per class target).
7. Reproducible artifacts: models, scalers, metrics, plots, confusion matrices.
8. Integration demo script for live inference.
9. Raspberry Pi 5 handheld runtime: acquisition, DSP/inference, and UI pipeline.
10. Production-grade touch UI showing live waveforms, harmonics, class probabilities, THD, RMS, PF, event timeline, and system health.
11. Kiosk-mode boot and service management for appliance-like operation.

### Out of Scope
1. Full industrial hardening and certification package.
2. Hardware BOM procurement automation.

## 8. Functional Requirements
### FR-1 Firmware Acquisition
1. Configure Teensy 4.1 dual ADC synchronized reads on A0 and A10.
2. Sampling rate must be 5 kHz with timer ISR period 200 us.
3. Window collection must use rising zero-crossing trigger on voltage channel.
4. Each frame must include 500 voltage samples and 500 current samples.

### FR-2 Serial Frame Protocol
1. Frame format must be exactly 2012 bytes.
2. Fields must include magic 0xDEADBEEF, seq uint16, N uint16, data arrays, CRC32.
3. CRC32 must match Python binascii.crc32 verification.

### FR-3 Host Receiver
1. Must robustly resync on magic bytes.
2. Must verify n_check equals 500.
3. Must verify CRC before accepting a frame.
4. Must expose accepted frames as numpy int16 arrays.

### FR-4 Calibration and Preprocessing
1. Convert ADC counts to physical units using configurable constants.
2. Remove DC offsets per frame.
3. Provide normalized waveform pair for model branches requiring normalized signals.

### FR-5 Feature Extraction (282 features)
1. Time-domain stats: 24 total across V and I.
2. FFT magnitudes h=1..13 for V and I: 26.
3. THD values for V and I: 2.
4. Phase-aware features exactly per report, including circular statistics.
5. DWT feature extraction for V and I: 72.
6. Final vector shape must be exactly 282.

### FR-6 Dataset Generation
1. Generate 7 classes: Normal, Sag, Swell, Interruption, Harmonic Distortion, Transient, Flicker.
2. Total synthetic dataset must be 38,500 balanced samples.
3. Harmonic Distortion class must use load-specific phase distributions (Von Mises).
4. Train/val/test split must be reproducible and documented.

### FR-7 Models and Ablation
1. Implement M1, M2, M3, M4 variants as described in report.
2. Run controlled ablation with identical splits and logging.
3. Save best checkpoints and metrics per model.

### FR-8 Domain Adaptation
1. Support fine-tuning with real captured labeled windows.
2. Freeze backbone and train selected final layers at lower LR.
3. Report before/after adaptation metrics.

### FR-9 Inference and Demo
1. Real-time or near-real-time inference script consuming live or recorded frames.
2. Output predicted class + confidence.
3. Persist logs with timestamp and selected derived values.

### FR-10 Handheld Device Runtime (Raspberry Pi 5)
1. Target runtime device: Raspberry Pi 5 (8GB) with official Pi display.
2. Application must start automatically on boot in kiosk/fullscreen mode.
3. Runtime must use a multi-loop architecture:
4. Acquisition loop for serial frames.
5. DSP/inference loop for feature extraction and model execution.
6. UI render loop with independent frame rate.
7. Inter-loop communication must use thread-safe queues/ring buffers.

### FR-11 Real-Time UI and Monitoring Parameters
1. UI must render smooth live voltage and current waveforms.
2. UI must render harmonic spectrum (orders 1-13 minimum).
3. UI must show per-class probability bars and top-1 class with confidence.
4. UI must show at least these scalar parameters in real time: RMS-V, RMS-I, THD-V, THD-I, PF/DPF where available, frequency estimate, timestamp.
5. UI must include event timeline/log panel with severity and event type.
6. UI must include health panel: serial status, CRC fail count, dropped frames, inference latency, device temperature.
7. UI must be touch-friendly for handheld use (large controls, page tabs, clear contrast).

### FR-12 Device Operations and Reliability
1. App must auto-reconnect serial stream after transient disconnect.
2. App must degrade gracefully under load (drop oldest frames, keep latest display responsive).
3. System must persist session logs locally and support replay mode.
4. Optional watchdog service should restart app on crash.

## 9. Non-Functional Requirements
1. Reproducibility: fixed seeds, deterministic split files, versioned config.
2. Maintainability: modular files by concern (firmware, io, features, models, eval).
3. Performance: per-frame processing latency under 200 ms on laptop CPU.
4. Reliability: dropped/corrupt serial frames handled without crash.
5. Safety documentation: preserve high-voltage warnings and handling checklists.
6. Pi performance target: UI >= 25 FPS, inference update >= 8 Hz under normal load.
7. Pi thermal target: sustained operation without thermal shutdown (active cooling required).
8. Startup target: application ready <= 30 seconds after boot.

## 10. Data and File Requirements
1. Keep raw captured frames and processed datasets separate.
2. Save scalers and model weights with explicit version names.
3. Save confusion matrices and training curves for each model variant.
4. Add a run manifest (json) for each experiment.

## 11. Proposed Repository Structure
1. firmware/teensy/pq_firmware/pq_firmware.ino
2. src/io/frame_protocol.py
3. src/io/serial_receiver.py
4. src/dsp/preprocess.py
5. src/dsp/features.py
6. src/dsp/wavelet_features.py
7. src/data/synthetic_generator.py
8. src/data/splits.py
9. src/models/m1_baseline.py
10. src/models/m2_waveform.py
11. src/models/m3_waveform_mag.py
12. src/models/m4_phase_aware.py
13. src/train/train.py
14. src/eval/evaluate.py
15. src/eval/ablation.py
16. src/adapt/domain_adapt.py
17. src/infer/live_infer.py
18. configs/default.yaml
19. scripts/run_full_pipeline.sh
20. artifacts/ (ignored large outputs)
21. src/ui/app.py
22. src/ui/views/dashboard.py
23. src/ui/views/trends.py
24. src/ui/views/events.py
25. src/ui/widgets/plots.py
26. src/system/kiosk_setup.sh
27. src/system/service/pq-monitor.service
28. docs/handheld_assembly.md

## 12. Dependencies
1. Python >= 3.10
2. numpy, scipy, pywavelets, scikit-learn, pyserial, tensorflow, matplotlib, seaborn, joblib, pyyaml
3. Arduino/Teensy toolchain with ADC and FastCRC libraries

## 13. Success Metrics
1. All FR requirements implemented and tested.
2. Feature vector shape validated at 282 in unit/integration tests.
3. M4 outperforms M3 on synthetic test split by measurable margin.
4. Domain adaptation improves real-data performance over zero-shot baseline.
5. End-to-end live pipeline runs without protocol errors for at least 100 consecutive frames.
6. Handheld app runs for 30 minutes continuously without crash or UI freeze.
7. Live dashboard shows all required PQ parameters with timestamped updates.

## 14. Risks and Mitigations
1. Synthetic-real mismatch risk.
Mitigation: domain adaptation, noise modeling, calibration checks.
2. Timing drift or frame corruption risk.
Mitigation: CRC, sequence checks, drop/retry logic, serial buffer tuning.
3. Class definition mismatch risk.
Mitigation: lock canonical class map in config and tests.
4. Scope creep risk.
Mitigation: strict chunked execution using tasks.md.
5. Handheld thermal/power instability risk.
Mitigation: active cooling, official PSU or validated battery power path, runtime telemetry.
6. UI stutter under DSP/inference load.
Mitigation: decoupled render loop, bounded queues, backpressure strategy.

## 15. Definition of Done
1. End-to-end demo works from Teensy capture to class output.
2. Reproducible training for M1-M4 completed and documented.
3. All required artifacts generated and stored with manifests.
4. Domain adaptation script and results included.
5. README-level run instructions updated and verified by clean run.
6. Raspberry Pi handheld build boots directly into monitoring UI and shows required live graphs and metrics.
7. Handheld deployment runbook completed, including thermal, power, and safety checks.

## 16. AI Agent Execution Notes
1. Follow tasks.md chunk order; do not skip integration tests.
2. Every chunk must end with acceptance checks and artifact generation.
3. Keep report alignment explicit in commit messages and logs.
4. Do not silently change class labels, sample rates, frame format, or feature counts.
