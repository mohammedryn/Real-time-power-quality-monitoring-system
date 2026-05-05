# PQ Repository Audit Report

## 1. Executive Summary

The repository is not operating against a single canonical contract. The PRD, `README.md`, `tasks.md`, and several docs still define a `2012`-byte raw frame and a `282`-element feature vector, while the active firmware DSP, runtime feature indexing, config, and parity tests have moved to a `5204`-byte model-ready frame carrying a reconstructed `298`-element feature space. That split is not cosmetic: it breaks live compatibility between the current firmware and the kiosk/UI defaults, causes the runtime to decode legacy `282`-feature frames with `298`-layout indices, and leaves the repository partially migrated with PRD-critical ML modules in `src/` still missing.

What is healthy:
- The raw-frame transport contract itself is implemented cleanly when `PQ_RAW_MODE=1`: `0xDEADBEEF`, `2012` bytes, `CRC32`, and Python-side `binascii.crc32` alignment are consistent between [firmware/teensy/pq_firmware/src/main.cpp:133](firmware/teensy/pq_firmware/src/main.cpp#L133) and [src/io/frame_protocol.py:133](src/io/frame_protocol.py#L133).
- The ISR path keeps blocking work out of the `200 us` sampling interrupt. Acquisition is done in `sampleISR()`, and DSP/USB transmission is deferred to `loop()` in [firmware/teensy/pq_firmware/src/main.cpp:95](firmware/teensy/pq_firmware/src/main.cpp#L95) and [253](firmware/teensy/pq_firmware/src/main.cpp#L253).
- The Python runtime has bounded queues and explicit drop policies in [src/runtime/buffers.py:22](src/runtime/buffers.py#L22).

What is blocking:
- The default firmware no longer emits the frame type the default UI/kiosk expects.
- The runtime claims to support both `282` and `298` feature vectors, but the metric extraction and harmonic indexing logic are hardcoded to the `298` layout.
- The `src/models`, `src/train`, `src/data`, `src/eval`, and `src/adapt` trees required by the PRD are effectively unimplemented stubs.
- The remaining legacy ML scripts under `jafed_model/` do not provide an independent held-out test workflow; `eval.py` replays the validation split recreated from the same dataset.

Audit execution notes:
- Static code inspection covered `README.md`, `prd.md`, `tasks.md`, `study_roadmap.md`, `firmware/`, `src/`, `tests/`, and `configs/`.
- `pytest` collection could not run the DSP/runtime tests in the current environment because `pywt` is not installed. Protocol-only subsets that do not import the DSP stack did run and passed (`36 passed`).

## 2. Critical Vulnerabilities / Blocking Bugs

1. **Spec split-brain: the repository simultaneously defines incompatible frame and feature contracts.**
   - The PRD and top-level docs still require `2012`-byte frames and `282` features: [README.md:5-9](README.md#L5), [prd.md:18](prd.md#L18), [prd.md:60-81](prd.md#L60), [tasks.md:37](tasks.md#L37), [tasks.md:71-82](tasks.md#L71), [study_roadmap.md:147-149](study_roadmap.md#L147), [study_roadmap.md:212-226](study_roadmap.md#L212).
   - The active firmware/runtime/config/tests have moved to `298` features and a `5204`-byte model-ready frame: [firmware/teensy/pq_firmware/src/dsp.h:4-20](firmware/teensy/pq_firmware/src/dsp.h#L4), [firmware/teensy/pq_firmware/src/main.cpp:39-51](firmware/teensy/pq_firmware/src/main.cpp#L39), [src/dsp/features.py:84-101](src/dsp/features.py#L84), [src/dsp/feature_index.py:3-17](src/dsp/feature_index.py#L3), [configs/default.yaml:36-48](configs/default.yaml#L36), [tests/test_feature_shape.py:6-24](tests/test_feature_shape.py#L6), [tests/test_model_ready_protocol.py:37-40](tests/test_model_ready_protocol.py#L37).
   - This violates the PRD’s “Do not silently change ... frame format, or feature counts” rule in [prd.md:215](prd.md#L215).

2. **The default firmware cannot talk to the default UI/kiosk runtime.**
   - Current firmware only has two transmission modes:
     - raw `2012`-byte mode behind `PQ_RAW_MODE=1`: [firmware/teensy/pq_firmware/src/main.cpp:53-57](firmware/teensy/pq_firmware/src/main.cpp#L53), [133-162](firmware/teensy/pq_firmware/src/main.cpp#L133)
     - default model-ready `5204`-byte mode when `PQ_RAW_MODE=0`: [firmware/teensy/pq_firmware/src/main.cpp:164-242](firmware/teensy/pq_firmware/src/main.cpp#L164), [253-259](firmware/teensy/pq_firmware/src/main.cpp#L253)
   - There is no feature-frame sender in the firmware for the `n=282` / `1140`-byte path.
   - The UI and kiosk tooling still default to `feature` mode and do not accept `model4`:
     - [src/ui/app.py:74-79](src/ui/app.py#L74)
     - [src/system/kiosk_setup.sh:14](src/system/kiosk_setup.sh#L14), [42](src/system/kiosk_setup.sh#L42), [98-103](src/system/kiosk_setup.sh#L98)
     - [README.md:40-49](README.md#L40), [75-87](README.md#L75)
   - Result: the documented kiosk path is pointed at a frame type the shipped firmware does not emit.

3. **Feature-mode runtime decoding is wrong because the metric/harmonic indices are hardcoded to the `298` layout but the code claims they are valid for `282` too.**
   - Runtime indices are defined for the `298` vector in [src/runtime/pipeline.py:23-30](src/runtime/pipeline.py#L23) and used in [src/runtime/pipeline.py:542-596](src/runtime/pipeline.py#L542).
   - The code explicitly claims those indices are “valid for both `298-element` and `282-element` vectors” in [src/runtime/pipeline.py:543-544](src/runtime/pipeline.py#L543).
   - That claim is false. The documented legacy `282` layout places:
     - THD at `50:52`, not `54:56`
     - current harmonics at `37:50`, not `41:54`
     - cross-phase sin/cos at `104:130`, not `108:134`
     - DWT at `210:282`, not `214:298`
     - see [docs/model_prd.md:49-69](docs/model_prd.md#L49)
   - Because `src/ui/app.py` defaults to `feature`, the dashboard’s RMS/THD/PF/harmonics are decoded from the wrong offsets in the default path.

4. **A 3-input Keras model will break in feature mode because the predictor silently falls back to a single-input call.**
   - Multi-input detection is set in [src/runtime/pipeline.py:171-176](src/runtime/pipeline.py#L171).
   - `predict_proba()` only uses the 3-input path when both `v_norm` and `i_norm` are present: [src/runtime/pipeline.py:194-196](src/runtime/pipeline.py#L194).
   - Otherwise it falls through to the single-input branch and calls `self._model.predict(x, verbose=0)` on a single tensor: [src/runtime/pipeline.py:197-224](src/runtime/pipeline.py#L197).
   - The repository’s model-4 training/inference code expects 3 inputs with shapes `(500,2)`, `(28,)`, `(270,)`: [jafed_model/model_4/model_4/model.py:18](jafed_model/model_4/model_4/model.py#L18), [37](jafed_model/model_4/model_4/model.py#L37), [50](jafed_model/model_4/model_4/model.py#L50).
   - Since `src/ui/app.py` exposes only `feature` and `raw` modes, the UI cannot safely use the current multi-input `.keras` artifact path.

5. **The PRD-required ML/data/train stack in `src/` is missing, so FR-6 through FR-8 are not implemented in the active codebase.**
   - `src/models/__init__.py`, `src/train/__init__.py`, `src/data/__init__.py`, `src/eval/__init__.py`, and `src/adapt/__init__.py` are stubs only: [src/models/__init__.py:1](src/models/__init__.py#L1), [src/train/__init__.py:1](src/train/__init__.py#L1), [src/data/__init__.py:1](src/data/__init__.py#L1), [src/eval/__init__.py:1](src/eval/__init__.py#L1), [src/adapt/__init__.py:1](src/adapt/__init__.py#L1).
   - The PRD expects concrete modules such as `src/data/splits.py`, `src/train/train.py`, `src/eval/evaluate.py`, and `src/adapt/domain_adapt.py`: [prd.md:152-156](prd.md#L152).
   - That is a direct implementation gap against the repository’s stated scope.

## 3. Architecture & Performance Anomalies

1. **Model-ready mode cannot drive the waveform plots because the runtime drops physical waveform arrays before snapshot construction.**
   - `ModelReadyFrame` is converted into a `FrameContext` with only `v_norm`/`i_norm`: [src/runtime/pipeline.py:450-465](src/runtime/pipeline.py#L450).
   - `InferenceSnapshot` only serializes `v_phys`/`i_phys`, not normalized waveforms: [src/runtime/pipeline.py:651-652](src/runtime/pipeline.py#L651).
   - The dashboard plots are updated only when `snapshot.v_phys` and `snapshot.i_phys` are present: [src/ui/views/dashboard.py:97-99](src/ui/views/dashboard.py#L97).
   - Net effect: even after adding `model4` UI support, the current dashboard would render no live waveforms for the default firmware mode.

2. **The runtime package has an avoidable import-time dependency explosion that breaks unrelated tests and utilities.**
   - `src/runtime/__init__.py` imports `.pipeline` unconditionally: [src/runtime/__init__.py:3-5](src/runtime/__init__.py#L3).
   - `.pipeline` imports `src.dsp.features`, which imports `pywt`: [src/runtime/pipeline.py:13](src/runtime/pipeline.py#L13), [src/dsp/features.py:3](src/dsp/features.py#L3), [src/dsp/wavelet_features.py:2](src/dsp/wavelet_features.py#L2).
   - Observed during audit: `tests/test_runtime_buffers.py` could not even collect because importing `src.runtime.buffers` first imports `src.runtime.__init__`, which then requires `pywt`.
   - This is not a protocol failure, but it is a packaging/testability bug that makes small runtime components harder to validate in isolation.

3. **Frequency is not estimated from the incoming waveform; the UI displays the configured nominal value.**
   - The pipeline caches `mains_frequency_hz` from config in [src/runtime/pipeline.py:297-299](src/runtime/pipeline.py#L297).
   - Snapshot metrics then report that constant directly as `frequency_hz`: [src/runtime/pipeline.py:561-569](src/runtime/pipeline.py#L561).
   - FR-11 requires a live frequency estimate, not a config echo.

4. **UI harmonic rendering recreates plot objects every update instead of updating them in place.**
   - [src/ui/widgets/plots.py:75-82](src/ui/widgets/plots.py#L75) removes and recreates both `BarGraphItem`s on every update.
   - At the target Pi kiosk rates, this is extra scene-graph churn with no obvious benefit. It is a moderate but real rendering inefficiency.

5. **The repository’s own migration documentation no longer matches the implementation.**
   - `docs/teensy_dsp_migration_tracker.md` still states that the target is a `282`-feature direct feature frame: [docs/teensy_dsp_migration_tracker.md:7-19](docs/teensy_dsp_migration_tracker.md#L7).
   - The same document says DWT was aligned to `periodization`: [docs/teensy_dsp_migration_tracker.md:74-80](docs/teensy_dsp_migration_tracker.md#L74), but the active Python and firmware both use symmetric boundaries: [src/dsp/wavelet_features.py:43-52](src/dsp/wavelet_features.py#L43), [firmware/teensy/pq_firmware/src/dwt.h:6-20](firmware/teensy/pq_firmware/src/dwt.h#L6).
   - That drift makes the migration tracker unsuitable as a validation artifact.

6. **There is no executable firmware timing evidence checked into the active repo, so the model-ready path’s end-to-end timing budget is asserted but not substantiated.**
   - Timing hooks exist in [firmware/teensy/pq_firmware/src/main.cpp:194-237](firmware/teensy/pq_firmware/src/main.cpp#L194) and capture tooling exists in [scripts/capture_teensy_timing.py:24-94](scripts/capture_teensy_timing.py#L24).
   - But the repo does not contain the promised hardware timing artifacts under active `artifacts/` paths, and the alignment matrix still marks hardware-dependent validation as partial: [docs/report_alignment_matrix.md:21-25](docs/report_alignment_matrix.md#L21).

## 4. Data Science & ML Issues

1. **The active `src/` tree does not contain the PRD’s reproducible dataset splitting or training code, so train/val/test leakage cannot be audited there because the implementation is missing.**
   - Missing modules: `src/data/splits.py`, `src/train/train.py`, `src/eval/evaluate.py`, `src/adapt/domain_adapt.py` per [prd.md:152-156](prd.md#L152).
   - Present files are stubs only: [src/data/__init__.py:1](src/data/__init__.py#L1), [src/train/__init__.py:1](src/train/__init__.py#L1), [src/eval/__init__.py:1](src/eval/__init__.py#L1), [src/adapt/__init__.py:1](src/adapt/__init__.py#L1).

2. **Legacy model evaluation reuses the validation split instead of an independent test set.**
   - `jafed_model/model_4/model_4/train.py` performs a single `train_test_split(..., test_size=0.1, random_state=42)` and uses the held-out chunk as validation: [jafed_model/model_4/model_4/train.py:61-74](jafed_model/model_4/model_4/train.py#L61).
   - `jafed_model/model_4/model_4/eval.py` recreates that same validation split from the full dataset using the same `random_state=42` and evaluates on it: [jafed_model/model_4/model_4/eval.py:29-37](jafed_model/model_4/model_4/eval.py#L29).
   - This is not a true test workflow. It reports performance on the same split family used for model selection.

3. **The legacy multi-label dataset generator no longer matches the PRD’s dataset definition.**
   - The PRD specifies 7 disturbance classes and a `38,500`-sample balanced synthetic dataset: [prd.md:84-87](prd.md#L84).
   - `jafed_model/model_4/model_4/data_gen.py` generates `32` multi-label combinations and allocates `samples_per_combo * 32` examples, defaulting to a `128000`-sample dataset: [jafed_model/model_4/model_4/data_gen.py:139-167](jafed_model/model_4/model_4/data_gen.py#L139), [172-217](jafed_model/model_4/model_4/data_gen.py#L172).
   - That may be a valid experiment, but it is not the PRD dataset contract.

4. **Multi-label training removes stratification without replacing it with a multi-label-safe split strategy.**
   - `jafed_model/model_4/model_4/train.py` explicitly removes stratification because standard `train_test_split` does not support the label matrix: [jafed_model/model_4/model_4/train.py:61-70](jafed_model/model_4/model_4/train.py#L61).
   - There is no replacement iterative stratification or per-label distribution audit, so validation composition can drift materially across label combinations.

5. **Class naming is not canonical across the repository.**
   - PRD/tasks use `Harmonic Distortion`: [prd.md:84](prd.md#L84), [tasks.md:6](tasks.md#L6).
   - Config and some docs use `HarmonicDistortion`: [configs/default.yaml:12-20](configs/default.yaml#L12), [docs/model_prd.md:129](docs/model_prd.md#L129).
   - Some legacy model scripts still use `Harmonics`: [jafed_model/model_4/model_4/inference.py:11-19](jafed_model/model_4/model_4/inference.py#L11), [jafed_model/model_4/model_4/fault_combo.txt:5](jafed_model/model_4/model_4/fault_combo.txt#L5).
   - This is a real metadata consistency risk for threshold maps, plots, session logs, and downstream evaluation scripts.

## 5. Actionable Fixes

- [ ] **Pick one canonical protocol/feature contract and enforce it everywhere.**
  - If the project is now `298` features + `model4` frames, update `README.md`, `prd.md`, `tasks.md`, `study_roadmap.md`, deployment runbooks, and alignment docs to that contract.
  - If the canonical contract must remain `282`/`2012`, revert the runtime/config/tests to that layout and remove the newer offsets.

- [ ] **Remove the dead `feature` deployment path or reintroduce an actual firmware feature-frame sender.**
  - Current firmware has only raw and model-ready transmission.
  - Either add a `sendFeatureFrame()` branch in `firmware/teensy/pq_firmware/src/main.cpp` and document when it is used, or remove `feature` from `src/ui/app.py`, `src/system/kiosk_setup.sh`, `README.md`, and the runbooks.

- [ ] **Make `RuntimePipeline` schema-aware instead of pretending one set of indices fits both feature layouts.**
  - Add explicit validators for `282` and `298` vectors.
  - Move metric extraction offsets into a contract object keyed by frame mode.
  - Refuse to decode `FeatureFrame` with `298` indices.

- [ ] **Fix multi-input model handling at startup.**
  - When `_is_multi_input` is true, reject `feature` mode with a clear error before the live run starts.
  - Alternatively, support only `model4`/raw sources for 3-input models and document that requirement.

- [ ] **Add `model4` to the UI receiver choices and render waveforms from available data.**
  - Update `src/ui/app.py` to accept `model4`.
  - In `src/runtime/pipeline.py`, include either `v_norm/i_norm` in `InferenceSnapshot` or reconstruct a display waveform payload for the dashboard.
  - Update `src/ui/views/dashboard.py` so the waveform panel can plot normalized waves when physical-unit waves are unavailable.

- [ ] **Implement a real frequency estimator or remove the misleading live metric.**
  - Estimate frequency from the incoming frame (zero-crossing, phase slope, or harmonic phase progression).
  - Do not label a config constant as a live measurement.

- [ ] **Decouple `src/runtime/__init__.py` from the DSP-heavy pipeline import.**
  - Re-export only lightweight symbols there, or lazily import `pipeline`.
  - This will let `buffers`/`metrics` tests run without pulling in `pywt`.

- [ ] **Complete or explicitly de-scope the missing `src/` ML modules.**
  - Add the PRD-required `src/data/splits.py`, `src/train/train.py`, `src/eval/evaluate.py`, `src/eval/ablation.py`, and `src/adapt/domain_adapt.py`.
  - If the repo is intentionally “non-ML integration only,” rewrite the PRD/README/release docs to say that plainly and stop presenting those modules as implemented.

- [ ] **Create a true train/val/test workflow for the legacy `jafed_model` artifacts.**
  - Persist split indices once.
  - Evaluate only on a held-out test split never reused for model selection.
  - For multi-label data, use iterative stratification or an equivalent label-distribution-aware splitter.

- [ ] **Normalize canonical class names across config, docs, runtime, and legacy model scripts.**
  - Pick one spelling for class 4 and use it everywhere.
  - Regenerate threshold maps, confusion-matrix labels, and fault-combination metadata after renaming.

- [ ] **Archive actual timing and HIL parity artifacts for the active firmware mode.**
  - Capture `PQ_DEBUG_TIMING` results for the `model4` path.
  - Store parity/timing outputs under versioned artifact directories referenced from `docs/report_alignment_matrix.md`.
