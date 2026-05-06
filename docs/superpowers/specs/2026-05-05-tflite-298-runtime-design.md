# TFLite 298 Runtime Design

**Date:** 2026-05-05

**Status:** Proposed and user-approved for planning

**Goal**

Make the repository use the teammate-provided `.tflite` model as the only production inference artifact, lock `298` as the canonical feature-vector length, and remove `model4` as the active runtime term while keeping the existing packed three-input live path under neutral naming.

## 1. Canonical Contract

The active system contract is:

- Deployment inference artifact: `artifacts/models/pqm_multilabel_model.tflite`
- Canonical feature-vector length: `298`
- Canonical live inference path: packed three-input inference frame
- Canonical model inputs:
  - `X_wave`: shape `(500, 2)`
  - `X_mag`: shape `(28,)`
  - `X_phase`: shape `(270,)`

The old `.keras` artifact is not part of the production contract.

The old `282`-feature contract is not part of the active production contract.

The teammate claim that the final production model uses `292` features is not accepted as canonical because the inspected teammate inference script and artifact-aligned slicing both match the `298`-feature layout.

## 2. Evidence for the Contract

The selected contract is grounded in the current repository and teammate-provided artifacts:

- The teammate inference script at `artifacts/models/inference (1).py` slices:
  - `X_mag = X_full[28:56]`
  - `X_phase = concat(X_full[0:28], X_full[56:214], X_full[214:])`
- Those slices correspond to a `298`-element feature vector.
- The packed live frame structure already maps to the same three-input form used by the artifact:
  - waveform branch
  - magnitude branch
  - phase branch
- The repository DSP, feature index, firmware packing, and tests already align much more strongly with `298` than with `282`.

## 3. Naming Direction

`model4` should no longer be used as the active runtime, UI, config, deployment, or documentation term.

The transport and runtime naming should be changed to neutral terminology such as:

- `tflite`
- `inference`
- `packed_inference`

The exact final label should be chosen once during implementation and used consistently across:

- host runtime
- UI CLI
- serial receiver modes
- config
- deployment scripts
- docs
- tests

## 4. Runtime Architecture

### 4.1 Inference Backend

The runtime should become TFLite-only for production inference.

That means:

- remove the `.keras` and general mixed-backend production path
- remove runtime ambiguity about whether `.keras`, `.h5`, `joblib`, or `.tflite` is primary
- load the `.tflite` artifact at startup
- validate the artifact interface before live scoring begins

### 4.2 Feature-to-Input Mapping

The canonical `298`-feature vector remains the internal source vector for derived inference inputs.

The mapping is:

- `X_mag = features[28:56]`
- `X_phase = concat(features[0:28], features[56:214], features[214:298])`
- `X_wave = stack(v_norm, i_norm)` with final shape `(500, 2)`

This mapping must be treated as the active source of truth unless the `.tflite` artifact itself proves otherwise during runtime validation.

### 4.3 Live Frame Path

The current packed three-input live frame structure can remain if it already matches the TFLite contract.

However:

- the code should stop presenting it as `model4`
- the runtime should treat it as the canonical TFLite/inference frame path

### 4.4 Legacy Compatibility Boundary

Legacy `282` feature-frame support, if retained, is compatibility-only.

It may remain for:

- debug tooling
- replay support
- protocol archaeology

It should not remain:

- the default UI path
- the canonical live path
- a first-class equal contract beside `298`

## 5. Validation Rules

At startup, the runtime should validate the `.tflite` model interface.

Validation should include:

- input tensor count
- output tensor count
- input tensor rank and shape
- compatibility between prepared runtime tensors and model tensor expectations

The runtime must fail fast with a clear error if:

- the artifact cannot be loaded
- the input count is wrong
- the tensor shapes do not match the runtime contract
- the configured runtime mode does not supply the required waveform and feature branches

## 6. Testing Direction

The tests should be reorganized around the new canonical contract.

### 6.1 Canonical Success Path

The default passing path should verify:

- `298`-feature extraction
- packed three-input live frame handling
- TFLite predictor startup validation
- correct `X_wave`, `X_mag`, and `X_phase` construction
- correct runtime/UI defaults

### 6.2 Compatibility Tests

If `282` support is retained, those tests should be clearly labeled as:

- legacy
- compatibility-only
- non-canonical

They should not define the active contract.

## 7. Documentation Direction

All active docs should converge on one consistent statement:

- production model artifact is `.tflite`
- active feature-vector length is `298`
- active live path is the packed three-input inference path
- `model4` is not the preferred active term

The main files expected to change during implementation are:

- `README.md`
- `prd.md`
- `tasks.md`
- `docs/model_prd.md`
- `docs/teensy_dsp_migration_tracker.md`
- `docs/report_alignment_matrix.md`
- deployment and runbook docs

## 8. Problems This Migration Intentionally Solves

This design is intended to eliminate these active contradictions:

- `.keras` versus `.tflite` ambiguity
- `282` versus `298` ambiguity
- stray `292` claims without artifact backing
- `model4` naming drift
- wrong UI/runtime defaults
- documentation describing obsolete contracts as current

## 9. Non-Goals

This design does not try to:

- make `282` and `298` equal first-class active contracts
- preserve `.keras` as a production inference backend
- preserve `model4` as the active user-facing runtime term
- keep outdated docs unchanged for historical convenience

## 10. Implementation Output Expected Later

The follow-up implementation plan should cover:

- TFLite-only predictor design
- runtime mode and naming migration
- UI/deployment default updates
- schema and startup validation
- replay/test/doc updates
- compatibility boundaries for legacy `282` support
