# Report Alignment Matrix

## Status Legend
- Implemented: code and artifacts available
- Partial: software implemented, hardware evidence pending
- ML-owned: owned by ML teammate per project split

| Requirement Area | Report/PRD Intent | Code Path | Artifact/Doc Evidence | Status |
|---|---|---|---|---|
| Firmware acquisition and framing | Teensy synchronized capture and CRC transport | firmware/teensy/pq_firmware/src/main.cpp, src/io/frame_protocol.py | docs/teensy_dsp_migration_tracker.md | Partial |
| Host receiver robustness | Magic resync, CRC checks, reconnect | src/io/serial_receiver.py, src/io/frame_protocol.py | tests/test_frame_protocol.py, tests/test_receiver_resync.py | Implemented |
| Preprocess and 282 features | Report-aligned feature stack | src/dsp/preprocess.py, src/dsp/features.py, src/dsp/wavelet_features.py | tests/test_preprocess.py, tests/test_feature_shape.py, tests/test_teensy_dsp_parity.py | Implemented |
| Synthetic dataset generation | 38,500 balanced synthetic samples | ML teammate scope | docs/model_prd.md | ML-owned |
| M1-M4 models and training | model variants, ablation, training artifacts | ML teammate scope | docs/model_prd.md | ML-owned |
| Domain adaptation | zero-shot vs adapted real-data eval | ML teammate scope | docs/model_prd.md | ML-owned |
| Live inference integration | live and replay inference paths with logging | src/infer/live_infer.py, src/infer/offline_replay.py, src/runtime/pipeline.py | tests/test_e2e_pipeline.py | Implemented |
| Runtime architecture | 3-loop pipeline with bounded queues and backpressure | src/runtime/pipeline.py, src/runtime/buffers.py, src/runtime/metrics.py | tests/test_runtime_buffers.py, tests/test_runtime_metrics.py | Implemented |
| Handheld UI | touch-friendly waveform, harmonics, probabilities, metrics, events, health | src/ui/app.py, src/ui/views/dashboard.py, src/ui/views/events.py, src/ui/widgets/plots.py, assets/ui_theme/style.qss | artifacts/perf/pi_runtime_profile.md | Partial |
| Device deployment | kiosk boot, service restart, log policy | src/system/kiosk_setup.sh, src/system/service/pq-monitor.service, src/system/service/pq-monitor.logrotate | docs/pi_deployment_runbook.md, docs/handheld_assembly.md | Implemented |
| Demo packaging | runbook and fallback | docs/demo_runbook.md | docs/demo_runbook.md | Implemented |
| Final QA and soak | full run + 30 min Pi soak | test and soak execution on target hardware | artifacts/perf/pi_soak_test_report.md | Partial |

## Open Hardware-Dependent Items
1. Capture and archive PQ_DEBUG_TIMING logs from live hardware stream.
2. Execute full HIL parity run with stable source/load and archive artifacts.
3. Execute and append measured Pi soak-test metrics.
