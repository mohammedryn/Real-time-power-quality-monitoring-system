# PQ Monitor Repository — Comprehensive Code Audit Report

**Generated:** 2026-05-04  
**Auditor:** Claude Sonnet 4.6 (AI Principal Engineer)  
**Repository:** Real-Time Power Quality Monitoring System  
**Branch audited:** `main` (commit `6a6a8de`)

---

## 1. Executive Summary

The repository implements a real-time power quality monitoring system: Teensy 4.1 firmware acquires synchronized dual-channel ADC data at 5 kHz, streams binary frames over USB-serial, a Python host runs DSP and ML inference, and a PySide6/pyqtgraph touch UI renders the dashboard on a Raspberry Pi 5.

**The runtime core — protocol framing, CRC, thread-safe ring buffers, metrics telemetry, and serial reconnect logic — is production-quality.** The firmware DSP (Goertzel + C++ DWT with phase correction) matches the Python reference within tested tolerances. The UI architecture cleanly decouples acquisition, inference, and render loops with correct backpressure.

**However, the codebase is in a partially-migrated state.** The feature vector has grown from 282 to 298 elements (inserting 4 power-metric features and 12 DWT transient-booster features), but the migration is **not atomic**: the PRD, README, the serial feature-frame protocol constant (`N_FEATURES = 282`), the offline-replay JSONL validator, the UI CLI argument parser, and the integration test fixture all still reference the old 282-element contract. This creates a class of **silent data-corruption bugs** that produce wrong THD and harmonic values on the live dashboard whenever the default receiver mode is used. Additionally, **5 of 11 `tasks.md` implementation chunks are entirely absent** — all ML training, dataset generation, model definitions, ablation, and domain adaptation infrastructure is missing stubs only.

**Overall health rating: AMBER.** Hardware acquisition pipeline and runtime plumbing are production-quality. ML pipeline and feature-contract consistency are NOT.

---

## 2. Critical Vulnerabilities / Blocking Bugs

### BUG-001 ⚠️ CRITICAL — Feature Vector Size: PRD says 282, code produces 298

**Impact:** PRD acceptance criteria fail against actual code; the feature-frame serial protocol cannot carry current pipeline output; integration test exercises wrong-length vectors.

| Location | Stated value |
|---|---|
| `prd.md:39, 82` | "282-feature vector" / "exactly **282**" |
| `README.md:9` | `Feature vector length: 282` |
| `src/io/frame_protocol.py:19` | `N_FEATURES = 282  # legacy feature frame` |
| `src/dsp/features.py:88–198` | Produces **298 elements**, self-checks with `ValueError` at line 193 |
| `configs/default.yaml:41` | `expected_feature_length: 298` |
| `firmware/teensy/pq_firmware/src/dsp.h:20` | `static constexpr int N_FEATURES = 298;` |
| `src/dsp/feature_index.py:60` | `TOTAL_FEATURES = 298` |

The system grew the vector by inserting a 4-element power-metrics block at `[24:28]` and expanding DWT from 72→84 features (6 transient-booster features per channel). The firmware, Python DSP, config, and parity tests are **consistently updated to 298**. But the serial feature-frame protocol (`N_FEATURES = 282`) was never updated, causing three concrete failures:

1. **`src/io/frame_protocol.py:pack_feature_frame()` (line 179)** enforces `len(feat) != N_FEATURES` (282). Any attempt to pack a 298-element vector via the `feature` serial mode raises `ValueError` — the current pipeline output cannot be transmitted in this mode.

2. **`tests/test_e2e_pipeline.py:57`** constructs `feat_row = rng.standard_normal(N_FEATURES).astype(np.float32)` — a 282-element vector — and feeds it to a pipeline configured for 298. All downstream metric indices are wrong during this test (see BUG-002), meaning the test is exercising a broken data path silently.

3. **PRD §13.2** states the feature vector must be exactly 282. This fails against the actual implementation.

**Required fixes:**
```python
# prd.md + README.md — update all "282" to "298" for feature vector count

# tests/test_e2e_pipeline.py:57
# BEFORE: feat_row = rng.standard_normal(N_FEATURES).astype(np.float32)
# AFTER:
from src.dsp.feature_index import TOTAL_FEATURES
feat_row = rng.standard_normal(TOTAL_FEATURES).astype(np.float32)
```

---

### BUG-002 ⚠️ CRITICAL — Wrong Metric Indices for Legacy 282-Element FeatureFrame Path

**Impact:** Silent data corruption — live UI dashboard displays wrong THD and harmonic values whenever the `feature` receiver mode is active (which is the **only** mode the UI CLI currently supports — see BUG-003).

**Location:** `src/runtime/pipeline.py:23–29` (constants), `src/runtime/pipeline.py:543–596` (`_build_snapshot`)

```python
# pipeline.py — index constants calibrated for 298-vector ONLY
_IDX_THD_V  = 54
_IDX_THD_I  = 55
_IDX_HARM_V = slice(28, 41)   # 13 voltage harmonic magnitudes
_IDX_HARM_I = slice(41, 54)   # 13 current harmonic magnitudes
```

The comment at line 543–544 claims:
```python
# Indices are valid for both 298-element (model4) and 282-element
# (legacy) vectors — all accesses are within 0..281.
```

**This claim is FALSE.** The 298-vector layout inserts `power_metrics[24:28]` that does not exist in the 282-vector. The layouts diverge starting at index 24:

| Feature | Correct index in **298-vector** | Correct index in **282-vector** | Code uses |
|---|---|---|---|
| THD-V | 54 | 50 | **54 ← WRONG for 282** |
| THD-I | 55 | 51 | **55 ← WRONG for 282** |
| HARM_V[0..12] | `[28:41]` | `[24:37]` | **`[28:41]` ← WRONG for 282** |
| HARM_I[0..12] | `[41:54]` | `[37:50]` | **`[41:54]` ← WRONG for 282** |

Note: `RMS_V` (index 2) and `RMS_I` (index 14) are correct for both layouts and are unaffected.

When the UI runs in `--receiver-mode feature` (the default, and only supported mode — see BUG-003), `FeatureFrame` objects with 282 elements enter `_frame_to_context()` and are passed directly to `_build_snapshot()`. The extracted `thd_v`, `thd_i`, and harmonic bar values are read from wrong positions, producing **silent garbage on the live dashboard**.

**Required fix:**
```python
# src/runtime/pipeline.py — layout-aware index selection in _build_snapshot()
_LEGACY_N       = 282
_LEGACY_THD_V   = 50
_LEGACY_THD_I   = 51
_LEGACY_HARM_V  = slice(24, 37)
_LEGACY_HARM_I  = slice(37, 50)

# In _build_snapshot(), replace hardcoded indices:
is_legacy = (len(features) == _LEGACY_N)
thd_v_idx = _LEGACY_THD_V  if is_legacy else _IDX_THD_V
thd_i_idx = _LEGACY_THD_I  if is_legacy else _IDX_THD_I
harm_v_sl = _LEGACY_HARM_V if is_legacy else _IDX_HARM_V
harm_i_sl = _LEGACY_HARM_I if is_legacy else _IDX_HARM_I
```

---

### BUG-003 ⚠️ CRITICAL — `app.py` UI CLI Cannot Select `model4` Receiver Mode

**Impact:** The production-optimal firmware path (`model4` — 5204-byte frame with full 298 features + normalized waveforms) is unreachable from the kiosk UI. The UI is locked to legacy `feature` mode, which directly triggers BUG-002.

**Location:** `src/ui/app.py:77–83`

```python
parser.add_argument(
    "--receiver-mode",
    choices=["feature", "raw"],   # "model4" is ABSENT
    default="feature",
    help="feature: use MCU 282-feature frames, raw: host DSP fallback",
)
```

The firmware **by default** emits `model4` frames (see `firmware/.../main.cpp:164` — the `#else` branch runs by default). `configs/default.yaml:48` specifies `ml_inference.receiver_mode: model4`. The config value is loaded but **never applied** — `app.py` uses `args.receiver_mode` only.

Contrast with `src/infer/live_infer.py:24–28`, which correctly includes all three modes with `model4` as default. The UI is the only entry point that cannot access the correct mode.

**Required fix:**
```python
# src/ui/app.py
parser.add_argument(
    "--receiver-mode",
    choices=["model4", "feature", "raw"],
    default="model4",   # match firmware default and configs/default.yaml
    help="model4: firmware model-ready frames (default), feature: legacy 282-element, raw: host DSP",
)
```

---

### BUG-004 ⚠️ CRITICAL — `dsp_manual.py` Executes Global Side-Effect Code on Import

**Impact:** Any import of `src.dsp.dsp_manual` triggers a blocking `plt.show()` call, console output, and signal generation — fatal on a headless Raspberry Pi.

**Location:** `src/dsp/dsp_manual.py:168–254`

```python
# These lines execute unconditionally at module import time:
active_classes = [1, 4, 5]
v_buf, i_buf, v_base, i_base = generate_mixed_signal(active_classes)
feature_vector, v_cleaned, i_cleaned = extract_complete_feature_vector(v_buf, i_buf)
print("EXTRACTED 282-ELEMENT FEATURE ARRAY: ...")   # stdout side effect
plt.show()   # BLOCKING — kills headless Pi session
```

This module also uses the **old 282-element layout** with a structurally incompatible feature scheme (different time-domain stats, different phase features, no transient DWT boosters, DWT without `mode='symmetric'`), making its output incompatible with the production 298-element pipeline even if the import did not block.

Nothing currently imports this module, but it lives inside the importable `src/dsp/` package.

**Required fix:** Wrap all execution code in `if __name__ == "__main__":`. Relocate file to `legacy/dsp_manual.py` with a deprecation header.

---

### BUG-005 ⚠️ CRITICAL — JSONL Offline Replay Rejects Valid 298-Element Features

**Impact:** Replay of any JSONL session log generated by the current Python pipeline (298-element features) fails with `ValueError`.

**Location:** `src/infer/offline_replay.py:93–97`

```python
if has_features:
    features = np.asarray(payload["features"], dtype=np.float32).reshape(-1)
    if features.size != N_FEATURES:   # N_FEATURES = 282
        raise ValueError(f"Invalid features length {features.size} ...; expected {N_FEATURES}")
```

The `.npy` loader in the same file at lines 41–42 correctly accepts both widths:
```python
valid_widths = (N_FEATURES, TOTAL_FEATURES)   # (282, 298)
if array.shape[1] not in valid_widths:
    raise ValueError(...)
```

The JSONL loader is inconsistent with the `.npy` loader in the same file.

**Required fix:**
```python
# src/infer/offline_replay.py:_validate_replay_record()
valid_lengths = (N_FEATURES, TOTAL_FEATURES)   # accept both 282 and 298
if features.size not in valid_lengths:
    raise ValueError(
        f"Invalid features length {features.size}; expected one of {valid_lengths}"
    )
```

---

### BUG-006 ⚠️ CRITICAL — ML Pipeline: Chunks 4–8 of `tasks.md` Not Implemented

**Impact:** PRD Definition of Done items 2–4 cannot be satisfied; the model cannot be retrained, ablated, or domain-adapted from this codebase.

The following files specified in PRD §11 and `tasks.md` are entirely absent (only empty `__init__.py` stubs exist in their directories):

| PRD Requirement | Required module | Status |
|---|---|---|
| FR-6: Dataset generation | `src/data/synthetic_generator.py` | **MISSING** |
| FR-6: Train/val/test splits | `src/data/splits.py` | **MISSING** |
| FR-7: M1 baseline MLP | `src/models/m1_baseline.py` | **MISSING** |
| FR-7: M2 waveform model | `src/models/m2_waveform.py` | **MISSING** |
| FR-7: M3 waveform+magnitude | `src/models/m3_waveform_mag.py` | **MISSING** |
| FR-7: M4 phase-aware hybrid | `src/models/m4_phase_aware.py` | **MISSING** |
| FR-7: Model factory | `src/models/factory.py` | **MISSING** |
| Chunk 6: Training pipeline | `src/train/train.py` | **MISSING** |
| Chunk 6: Evaluation | `src/eval/evaluate.py` | **MISSING** |
| Chunk 7: Ablation runner | `src/eval/ablation.py` | **MISSING** |
| FR-8: Domain adaptation | `src/adapt/domain_adapt.py` | **MISSING** |
| FR-8: Real data capture | `src/data/real_capture.py` | **MISSING** |

A trained model artifact (`artifacts/models/pqm_multilabel_model.keras`) exists, so inference works against the pre-trained model. But the model cannot be retrained, the M4 vs M3 ablation (PRD §13.3) cannot be reproduced, and domain adaptation (FR-8) is non-functional.

---

## 3. Architecture & Performance Anomalies

### ANOM-001 — `ArtifactPredictor` Multi-Input Detection Uses Deprecated Keras API

**Severity:** HIGH  
**Location:** `src/runtime/pipeline.py:172–177`

```python
try:
    if isinstance(loaded.input, list) and len(loaded.input) == 3:
        self._is_multi_input = True
except Exception:
    pass
```

In TensorFlow 2.x / Keras 3.x, accessing `model.input` on a multi-input model raises `AttributeError` ("Use `model.inputs` instead"). The bare `except Exception: pass` silently suppresses this, leaving `_is_multi_input = False`. The model then falls to the single-input code path and attempts to reshape a flat 298-element vector as input to a three-branch architecture — causing a shape mismatch exception on the first real inference call.

**Required fix:**
```python
try:
    inputs = getattr(loaded, "inputs", None) or [loaded.input]
    if isinstance(inputs, list) and len(inputs) == 3:
        self._is_multi_input = True
except Exception:
    pass
```

---

### ANOM-002 — Multi-Label Probabilities Incorrectly Normalized in Single-Input Path

**Severity:** HIGH  
**Location:** `src/runtime/pipeline.py:219–224`

```python
# Single-input path:
probs = np.maximum(probs, 0.0)
denom = float(np.sum(probs))
return probs / denom   # softmax normalization applied to sigmoid outputs!
```

For multi-label models (independent sigmoid per class), summing and normalizing converts multi-label independent probabilities into a forced single-label distribution. Two simultaneous faults at 0.7 confidence each become 0.5 each — potentially falling below the per-class detection thresholds defined in `configs/default.yaml` (`Swell: 0.35`, `Transient: 0.35`, etc.).

The three-input path (`_predict_multi_input`) correctly skips this normalization and returns raw sigmoid outputs. The single-input and three-input paths produce structurally different inference behavior for the same underlying model.

**Required fix:** Add `_is_multi_label` detection (from config) to `ArtifactPredictor`. Skip normalization when `multi_label=True`.

---

### ANOM-003 — `SessionLogger.write()` Holds Lock During Disk Flush

**Severity:** MEDIUM  
**Location:** `src/runtime/pipeline.py:100–114`

```python
def write(self, snapshot: InferenceSnapshot) -> None:
    record = {...}
    with self._lock:
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()   # disk I/O inside the lock
```

JSON serialization and SD-card flush both happen while `_lock` is held. On a Raspberry Pi with a microSD card, a synchronous flush can take 10–50ms. Since `write()` is called from `_inference_loop` on every scored frame (up to 10 Hz target per `configs/default.yaml:62`), a slow flush can consume the entire 200ms per-frame inference budget.

**Required fix:** Move JSON serialization outside the lock:
```python
def write(self, snapshot: InferenceSnapshot) -> None:
    serialized = json.dumps({...}) + "\n"   # CPU-only — outside lock
    with self._lock:
        self._fp.write(serialized)
        self._fp.flush()
```

---

### ANOM-004 — `BoundedQueue.get(timeout=None)` Can Deadlock

**Severity:** MEDIUM  
**Location:** `src/runtime/buffers.py:57–62`

```python
def get(self, timeout: Optional[float] = None) -> Optional[T]:
    with self._cond:
        if timeout is None:
            while not self._data:
                self._cond.wait()   # no wakeup if producer dies
```

If `get(timeout=None)` is called with an empty queue after `stop()` is called (no active producer), the calling thread waits indefinitely. The `_inference_loop` always uses `get(timeout=0.2)`, which is safe today. But the public API exposes a deadlock footgun for any future caller or test that uses the blocking form.

---

### ANOM-005 — Class Name Inconsistency: PRD vs Config

**Severity:** MEDIUM  
**Location:** `prd.md:84–86` vs `configs/default.yaml:14–20`

| Source | Class name |
|---|---|
| `prd.md` | `Harmonic Distortion` (with space) |
| `configs/default.yaml` | `HarmonicDistortion` (no space, CamelCase) |

All runtime code reads class names from config, so the operational impact is contained. However, any code that hardcodes `"Harmonic Distortion"` (from reading the PRD, from test fixtures, or from UI label comparison) will silently fail to match the config value. One canonical form must be chosen and enforced across PRD, config, and all code.

---

### ANOM-006 — Firmware `circ_std` Fallback Differs From Python NaN→0.0 Clamp

**Severity:** LOW  
**Locations:** `firmware/teensy/pq_firmware/src/dsp.cpp:216` vs `src/dsp/features.py:171`

```c
// Firmware — near-zero R → ≈ 7.43
if (R < 1e-12f) return sqrtf(-2.0f * logf(1e-12f));
```
```python
# Python — NaN clamped to 0.0
circ_stats = [0.0 if np.isnan(x) else float(x) for x in circ_stats]
```

For near-zero or constant input signals where the resultant vector length `R → 0`, the firmware emits `≈ 7.43` while Python clamps `NaN → 0.0`. The parity test `test_near_zero_signal_no_nan` only checks for NaN presence, not value agreement. This creates an undetected firmware/Python parity gap at `circ_stats[1]`, `circ_stats[3]`, `circ_stats[5]` for degenerate signals.

---

### ANOM-007 — Firmware DWT Static Locals Are Not Re-Entrant (Informational)

**Severity:** LOW (no impact on single-threaded bare-metal Teensy)  
**Locations:** `firmware/.../dwt.h:101–102`, `firmware/.../dsp.cpp:145`

```c
static float approx[6][260];     // one shared copy — not re-entrant
static float detail_buf[5][260];
static float dwt_out[DWT_TOTAL_COEFFS];
```

On bare-metal single-core Teensy (non-preemptive ISR model), `dwt_db4_level5()` is never called from ISR context, so this is safe. If ever ported to a multi-core environment or called re-entrantly, data corruption would occur silently without any compiler or runtime warning.

---

## 4. Data Science & ML Issues

### ML-001 — No Data Leakage Prevention Infrastructure

**Severity:** HIGH

`src/data/splits.py` does not exist. There is no implementation to create or validate reproducible train/val/test splits, no fixed-seed documentation, and no mechanism to detect that synthetic samples (which share generative parameters by class) might inadvertently leak feature patterns across splits. PRD §13.1 (reproducibility via fixed seeds) and `tasks.md` Chunk 4 acceptance criteria cannot be verified.

---

### ML-002 — No Model Definitions — Ablation Cannot Be Run

**Severity:** HIGH

M1–M4 model variants are entirely unimplemented. A trained model artifact exists at `artifacts/models/pqm_multilabel_model.keras`, but:
- Its architecture cannot be verified from source code.
- The M4 vs M3 comparison required by PRD §13.3 cannot be reproduced.
- The ablation study (Chunk 7, `tasks.md:154–169`) cannot be executed.

The PRD Definition of Done explicitly requires "Reproducible training for M1–M4 completed and documented" (§15.2). This is not satisfiable from the current codebase.

---

### ML-003 — `ArtifactPredictor` Normalizes Multi-Label Outputs as Single-Label

**Severity:** MEDIUM  
*(Duplicated here for ML-context visibility — see ANOM-002 for full detail.)*

The single-input path in `ArtifactPredictor.predict_proba()` (`src/runtime/pipeline.py:219–224`) applies softmax-style normalization (`probs / sum(probs)`) to sigmoid multi-label outputs. This suppresses simultaneous fault detection. The three-input path does not normalize. The same model produces structurally different results depending on which code path is taken.

---

### ML-004 — Legacy `dsp_manual.py` Feature Scheme Is Incompatible With Production

**Severity:** LOW (legacy script, not imported by any production code)  
**Location:** `src/dsp/dsp_manual.py:89–102`

`extract_complete_feature_vector()` assembles a 282-element vector with:
- Different time-domain features (includes `energy` and `total_variation`; lacks `min`, `max`)
- Different phase feature scheme (amplitude-weighted coupling terms not in production)
- DWT without `mode='symmetric'` (may use `'periodization'` in older pywt)
- THD returned as percentage (×100) vs production ratio

If this script were ever used to generate training data or compare against production features, the mismatch would be undetected by any existing test.

---

## 5. Actionable Fixes (Prioritised Checklist)

### Priority 1 — Fix Before Any Live Hardware Run

- [ ] **`src/ui/app.py:77`** — Add `"model4"` to `--receiver-mode` choices; set default to `"model4"` to match firmware default and config.  
  *Fixes BUG-003. Unblocks the correct firmware path from the kiosk UI.*

- [ ] **`src/runtime/pipeline.py:543–596`** — Add layout-aware index selection in `_build_snapshot()`: for 282-element frames use `THD_V=50`, `THD_I=51`, `HARM_V=[24:37]`, `HARM_I=[37:50]`.  
  *Fixes BUG-002. Eliminates silent THD and harmonic corruption in the live dashboard.*

- [ ] **`src/runtime/pipeline.py:172–177`** — Replace `loaded.input` with `getattr(loaded, "inputs", None)` for multi-input model detection.  
  *Fixes ANOM-001. Prevents inference crash on first frame with a multi-input Keras model.*

- [ ] **`src/dsp/dsp_manual.py`** — Wrap all execution code in `if __name__ == "__main__":`. Move file to `legacy/dsp_manual.py`.  
  *Fixes BUG-004. Prevents blocking `plt.show()` on any import.*

### Priority 2 — Fix Before Any Replay or Test Workflow

- [ ] **`src/infer/offline_replay.py:94`** — Accept both `N_FEATURES` (282) and `TOTAL_FEATURES` (298) in JSONL feature validation, matching the `.npy` loader in the same file.  
  *Fixes BUG-005.*

- [ ] **`tests/test_e2e_pipeline.py:57`** — Change `rng.standard_normal(N_FEATURES)` → `rng.standard_normal(TOTAL_FEATURES)` to exercise the current 298-element pipeline.  
  *Fixes BUG-001 test impact.*

- [ ] **`prd.md` and `README.md`** — Update all instances of `"282"` → `"298"` for the feature vector count. Update FR-5 layout description.  
  *Fixes BUG-001 documentation impact.*

### Priority 3 — Architecture Hardening

- [ ] **`src/runtime/pipeline.py:219–224`** — Skip softmax normalization for multi-label models in the single-input path. Detect `multi_label` flag from config.  
  *Fixes ANOM-002 + ML-003. Restores correct simultaneous fault detection.*

- [ ] **`src/runtime/pipeline.py:100–114`** — Move JSON serialization outside `_lock` in `SessionLogger.write()`. Hold lock only for the raw write and flush.  
  *Fixes ANOM-003. Prevents inference thread stalls on SD-card I/O.*

- [ ] **`configs/default.yaml` + `prd.md`** — Align `"Harmonic Distortion"` / `"HarmonicDistortion"` to one canonical string everywhere.  
  *Fixes ANOM-005.*

### Priority 4 — Implement Missing ML Pipeline (tasks.md Chunks 4–8)

- [ ] `src/data/synthetic_generator.py` — 38,500 balanced samples, 7 classes, Von Mises phase distributions for HarmonicDistortion class.
- [ ] `src/data/splits.py` — Reproducible stratified train/val/test splits with fixed seed and leakage validation.
- [ ] `src/models/m1_baseline.py` through `m4_phase_aware.py` — All four model variants per PRD §7.
- [ ] `src/train/train.py` — Unified training with checkpoint, early stop, scaler saving, and run manifest.
- [ ] `src/eval/evaluate.py` + `src/eval/ablation.py` — Controlled M1–M4 ablation with shared splits.
- [ ] `src/adapt/domain_adapt.py` — Backbone freeze + final-layer fine-tuning workflow.
- [ ] `src/data/real_capture.py` — Labeled real-hardware window capture utility.

---

## Appendix: Key File Cross-Reference

| Bug / Anomaly | File | Line(s) |
|---|---|---|
| Feature vector 298 self-check | `src/dsp/features.py` | 193–198 |
| Legacy N_FEATURES=282 wire constant | `src/io/frame_protocol.py` | 19 |
| Wrong THD/harmonic index constants | `src/runtime/pipeline.py` | 23–29 |
| False "works for both layouts" claim | `src/runtime/pipeline.py` | 543–544 |
| UI argparser missing model4 | `src/ui/app.py` | 77–83 |
| JSONL 282-only validation | `src/infer/offline_replay.py` | 94–97 |
| .npy accepts 282+298 (reference) | `src/infer/offline_replay.py` | 41–42 |
| dsp_manual global exec code | `src/dsp/dsp_manual.py` | 168–254 |
| Multi-input detection bug | `src/runtime/pipeline.py` | 172–177 |
| SessionLogger lock + flush | `src/runtime/pipeline.py` | 100–114 |
| Multi-label normalization bug | `src/runtime/pipeline.py` | 219–224 |
| circ_std fallback — firmware | `firmware/teensy/pq_firmware/src/dsp.cpp` | 216 |
| circ_std fallback — Python | `src/dsp/features.py` | 171 |
| Goertzel phase correction (correct) | `firmware/teensy/pq_firmware/src/goertzel.h` | 40–43 |
| DWT static buffers (informational) | `firmware/teensy/pq_firmware/src/dwt.h` | 101–102 |
| Config receiver_mode=model4 (ignored by UI) | `configs/default.yaml` | 48 |
| E2E test uses 282-element vector | `tests/test_e2e_pipeline.py` | 57 |
