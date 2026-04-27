# Submission Checklist

## A. Core Software
- [x] Runtime architecture implemented (acquisition/inference/result loops)
- [x] Bounded queue and backpressure policy implemented
- [x] Live inference runner implemented
- [x] Offline replay runner implemented
- [x] Session JSONL logging implemented
- [x] Touch UI implementation added
- [x] Kiosk systemd service and installer script added

## B. Validation
- [x] Full pytest suite passes in development environment (59 passed)
- [x] Runtime buffers/metrics/e2e tests added
- [x] Firmware compile script validated
- [ ] Full HIL parity artifact capture on connected Teensy
- [ ] PQ_DEBUG_TIMING hardware timing capture
- [ ] 30-minute Pi soak test with measured telemetry

## C. Documentation
- [x] Pi deployment runbook
- [x] Handheld assembly guide
- [x] Demo runbook
- [x] Report alignment matrix
- [x] Final summary artifact
- [x] Version manifest

## D. ML Integration Boundary
- [x] ML-owned scope left untouched (dataset/model/training/ablation/domain adaptation)
- [x] Artifact-based model/scaler loading hooks provided for integration

## E. Final Handoff Gate
Submission-ready after section B hardware items are completed and their artifact files are updated with measured values.
