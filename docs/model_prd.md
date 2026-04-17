# Model Implementation PRD
## Phase-Aware Power Quality Monitor — ML Subsystem

**For:** ML / Deep Learning Teammate (Japeth)
**Owner:** Mohammed Rayan
**Date:** April 2026
**Status:** Active — Start Here

---

## 0. Quick Start — What You Need to Know in 60 Seconds

You are responsible for **Chunks 4, 5, 6, and 7** from `tasks.md`. The rest of the pipeline is already built and working. Your job is to:

1. Write the synthetic dataset generator (`src/data/synthetic_generator.py`)
2. Implement four model variants M1–M4 (`src/models/`)
3. Write the training + evaluation pipeline (`src/train/train.py`, `src/eval/evaluate.py`)
4. Run the ablation study and produce artifacts (`src/eval/ablation.py`)

**You do not touch:** firmware, serial receiver, preprocessing, or feature extraction. Those are done and tested.

**Do not repeat the legacy models.** The files in `legacy/Single_signal_LM/` are from an earlier iteration. They use the wrong sampling rate (6000 Hz vs 5000 Hz), wrong window duration assumptions, wrong class set (Notch vs Flicker), and arbitrary signal amplitudes. They cannot be adapted into the current spec. Use the signal injection logic as reference only — see Section 9.

**The canonical config is `configs/default.yaml`.** Never hardcode `fs`, `N`, class names, or paths. Always load from config.

---

## 1. System Context — What Already Exists

### 1.1 Existing Pipeline (Do Not Modify)

```
Teensy 4.1 Hardware
    ↓  USB / CRC-validated 2012-byte frames
src/io/frame_protocol.py        → pack_frame(), parse_frame(), validate_recorded_stream()
src/io/serial_receiver.py       → SerialFrameReceiver (reconnect, CRC, magic resync)
    ↓  ParsedFrame(v_raw, i_raw: np.int16[500])
src/dsp/preprocess.py           → preprocess_frame(v_raw, i_raw, cfg)
    ↓  dict: v_phys, i_phys, v_norm, i_norm  (all np.float64[500])
src/dsp/features.py             → extract_features(v_phys, i_phys)
src/dsp/wavelet_features.py     → extract_dwt_features(signal)
    ↓  np.float32[282]  ← THIS IS YOUR MODEL'S INPUT
```

### 1.2 What `extract_features()` Gives You

The function `src/dsp/features.py::extract_features(v_phys, i_phys)` takes the physical-unit waveforms (Volts and Amps, DC-removed, 500 samples each) and returns a **single `np.float32` array of length exactly 282**. This is the feature vector every model in this project operates on (in whole or in part).

**The exact layout of the 282-element vector — memorize this:**

| Slice | Indices | Count | Description |
|---|---|---|---|
| `X[:, 0:12]` | 0–11 | 12 | Time-domain stats: voltage (mean, std, rms, peak, crest\_factor, form\_factor, skew, kurtosis, ptp, zero\_crossings, min, max) |
| `X[:, 12:24]` | 12–23 | 12 | Same 12 stats for current channel |
| `X[:, 24:37]` | 24–36 | 13 | FFT harmonic magnitudes h=1..13, voltage channel |
| `X[:, 37:50]` | 37–49 | 13 | FFT harmonic magnitudes h=1..13, current channel |
| `X[:, 50:52]` | 50–51 | 2 | THD\_V, THD\_I |
| `X[:, 52:78]` | 52–77 | 26 | Absolute phase sin/cos encoding voltage: `[sin(φ_v1)...sin(φ_v13), cos(φ_v1)...cos(φ_v13)]` |
| `X[:, 78:104]` | 78–103 | 26 | Absolute phase sin/cos encoding current: same pattern |
| `X[:, 104:117]` | 104–116 | 13 | V-I cross-channel phase sin: `[sin(φ_v_h - φ_i_h)]` for h=1..13 |
| `X[:, 117:130]` | 117–129 | 13 | V-I cross-channel phase cos: `[cos(φ_v_h - φ_i_h)]` for h=1..13 |
| `X[:, 130:142]` | 130–141 | 12 | Relative phase sin (V): `[sin(φ_v_h - φ_v_1)]` for h=2..13 |
| `X[:, 142:154]` | 142–153 | 12 | Relative phase cos (V): `[cos(φ_v_h - φ_v_1)]` for h=2..13 |
| `X[:, 154:166]` | 154–165 | 12 | Relative phase sin (I): `[sin(φ_i_h - φ_i_1)]` for h=2..13 |
| `X[:, 166:178]` | 166–177 | 12 | Relative phase cos (I): `[cos(φ_i_h - φ_i_1)]` for h=2..13 |
| `X[:, 178:204]` | 178–203 | 26 | Per-harmonic active+reactive power: `[P_h, Q_h]` for h=1..13 |
| `X[:, 204:210]` | 204–209 | 6 | Circular stats: circmean\_V, circstd\_V, circmean\_I, circstd\_I, circmean\_cross, circstd\_cross |
| `X[:, 210:246]` | 210–245 | 36 | DWT subband features, voltage: 6 subbands × 6 stats (mean,std,skew,kurt,energy,entropy) |
| `X[:, 246:282]` | 246–281 | 36 | DWT subband features, current: same 6×6 pattern |

**Total: 282. Verified passing in tests.**

### 1.3 Model Input Slices (Critical — Use These Exactly)

The M4 model has three branches. Each branch receives a different slice of the data:

```python
# Branch 1: Raw normalized waveform — from preprocess_frame(), NOT from features.py
X_wave  = np.stack([v_norm, i_norm], axis=-1)   # shape: (500, 2)

# Branch 2: Magnitude features — FFT magnitudes + THD only
X_mag   = X_full[:, 24:52]                       # shape: (28,)
# Indices 24:52 = [13 V mags] + [13 I mags] + [THD_V, THD_I] = 28 features

# Branch 3: Phase + DWT + time-domain features
X_phase = np.concatenate([
    X_full[:, 0:24],     # time-domain stats (24)
    X_full[:, 52:210],   # all phase blocks (158): abs_phase(52) + cross(26) + rel(48) + powers(26) + circ(6)
    X_full[:, 210:282],  # DWT features (72)
], axis=1)               # shape: (254,)
```

Verify: `28 + 254 = 282`. The time-domain stats (24) appear in both X_mag (indirectly via M1/M3's combined input) and X_phase (explicitly). They are reused across branches by design.

For M1 (magnitude MLP) and M3 (CNN-LSTM + magnitude), the magnitude branch input is: `np.concatenate([X_full[:, 0:24], X_full[:, 24:52]], axis=1)` = time-domain (24) + FFT+THD (28) = 52 features.

---

## 2. Canonical Constants — Never Hardcode These

All constants must be loaded from `configs/default.yaml`. For your scripts:

```python
import yaml
from pathlib import Path

def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()
FS          = cfg["signal"]["fs_hz"]               # 5000
N           = cfg["signal"]["samples_per_frame"]   # 500
F0          = cfg["signal"]["mains_frequency_hz"]  # 50
CLASS_NAMES = cfg["classes"]["names"]              # list of 7
N_CLASSES   = len(CLASS_NAMES)                     # 7
ARTIFACTS   = Path(cfg["paths"]["artifacts_root"]) # artifacts/
SYNTH_DIR   = Path(cfg["paths"]["synthetic_dataset"])  # artifacts/datasets/synth_v1
```

**Class ID mapping (canonical — never change the order):**

| ID | Name |
|---|---|
| 0 | Normal |
| 1 | Sag |
| 2 | Swell |
| 3 | Interruption |
| 4 | HarmonicDistortion |
| 5 | Transient |
| 6 | Flicker |

---

## 3. Chunk 4 — Synthetic Dataset Generator

**File:** `src/data/synthetic_generator.py`
**File:** `src/data/splits.py`
**Test:** `tests/test_dataset_balance.py`
**Output:** `artifacts/datasets/synth_v1/`

### 3.1 Dataset Dimensions

| Split | Samples/class | Total |
|---|---|---|
| Train | 4,000 | 28,000 |
| Validation | 750 | 5,250 |
| Test | 750 | 5,250 |
| **Total** | **5,500** | **38,500** |

All classes strictly balanced. Fixed random seed = **42** for full reproducibility.

### 3.2 Signal Parameters

```python
FS = 5000          # Hz — must match hardware
N  = 500           # samples — 100 ms window (5 cycles at 50Hz)
t  = np.linspace(0, N/FS, N, endpoint=False)  # 0 to 0.0999... seconds, 500 points
```

Every synthetic sample generates two arrays: `v_sig` (Volts, physical) and `i_sig` (Amps, physical). These are passed directly to `extract_features(v_sig, i_sig)` — **no ADC conversion, no midpoint subtraction**. The signals are already in physical units.

### 3.3 Per-Class Signal Models

Implement a function `generate_sample(class_id: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` that returns `(v_sig, i_sig)` as float64 arrays of shape `(500,)`.

#### Class 0 — Normal

```python
V1   = rng.uniform(200, 245) * np.sqrt(2)      # peak voltage (282–346 V_peak)
phi1 = rng.uniform(0, 2 * np.pi)               # random absolute phase
thd_bg = rng.uniform(0.01, 0.04)               # background THD 1–4%

v_sig = V1 * np.sin(2*np.pi*50*t + phi1)

# Add small background harmonics (random mix, total THD ~ thd_bg)
for h in [3, 5, 7]:
    amp = (thd_bg / np.sqrt(3)) * V1 * rng.uniform(0.5, 1.5)
    v_sig += amp * np.sin(2*np.pi*h*50*t + rng.uniform(0, 2*np.pi))

# Current: resistive-inductive load
dpf     = rng.uniform(0.85, 1.0)
I1      = rng.uniform(1.0, 15.0)
phi_i   = phi1 - np.arccos(dpf)                # lagging by PF angle
i_sig   = I1 * np.sin(2*np.pi*50*t + phi_i)

# AWGN
sigma_v = rng.uniform(0.002, 0.010) * V1
sigma_i = rng.uniform(0.002, 0.010) * I1
v_sig  += rng.normal(0, sigma_v, N)
i_sig  += rng.normal(0, sigma_i, N)
```

#### Class 1 — Sag

```python
V1    = rng.uniform(200, 245) * np.sqrt(2)
phi1  = rng.uniform(0, 2*np.pi)
v_sig = V1 * np.sin(2*np.pi*50*t + phi1)

depth    = rng.uniform(0.1, 0.9)               # 10%–90% reduction
duration = rng.uniform(0.5, 30.0) / 50.0      # 0.5–30 cycles in seconds
t_start  = rng.uniform(0, max(0, (N/FS) - duration))
mask     = (t >= t_start) & (t <= t_start + duration)
v_sig[mask] *= (1.0 - depth)

I1    = rng.uniform(1.0, 15.0)
i_sig = I1 * np.sin(2*np.pi*50*t + phi1)
i_sig[mask] *= (1.0 - depth)                   # current follows voltage

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

#### Class 2 — Swell

```python
V1    = rng.uniform(200, 245) * np.sqrt(2)
phi1  = rng.uniform(0, 2*np.pi)
v_sig = V1 * np.sin(2*np.pi*50*t + phi1)

# Beta(1.5, 5.0) biases toward small swells (realistic distribution)
depth    = float(rng.beta(1.5, 5.0)) * 0.7    # 0–70% above nominal, mostly 5–25%
duration = rng.uniform(0.5, 30.0) / 50.0
t_start  = rng.uniform(0, max(0, (N/FS) - duration))
mask     = (t >= t_start) & (t <= t_start + duration)
v_sig[mask] *= (1.0 + depth)

I1    = rng.uniform(1.0, 15.0)
i_sig = I1 * np.sin(2*np.pi*50*t + phi1)
i_sig[mask] *= (1.0 + depth)

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

#### Class 3 — Interruption

```python
V1      = rng.uniform(200, 245) * np.sqrt(2)
phi1    = rng.uniform(0, 2*np.pi)
v_sig   = V1 * np.sin(2*np.pi*50*t + phi1)

residual = rng.uniform(0.0, 0.1)              # 0–10% residual voltage
duration = rng.uniform(1.0, 5.0) / 50.0      # 1–5 cycles
t_start  = rng.uniform(0, max(0, (N/FS) - duration))
mask     = (t >= t_start) & (t <= t_start + duration)
v_sig[mask] *= residual

I1    = rng.uniform(1.0, 15.0)
i_sig = I1 * np.sin(2*np.pi*50*t + phi1)
i_sig[mask] *= residual

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

#### Class 4 — HarmonicDistortion (Von Mises — most important class)

This class has three load subtypes with physically distinct harmonic phase signatures. Randomly select one subtype per sample with equal probability. This is the core scientific contribution of the project — the phase distributions are NOT uniform random.

```python
from scipy.stats import vonmises

V1   = rng.uniform(200, 245) * np.sqrt(2)
phi1 = rng.uniform(0, 2*np.pi)
v_sig = V1 * np.sin(2*np.pi*50*t + phi1)
I1   = rng.uniform(2.0, 15.0)
i_sig = I1 * np.sin(2*np.pi*50*t + phi1)

subtype = rng.integers(0, 3)   # 0=SMPS, 1=VFD, 2=Transformer

if subtype == 0:   # SMPS / single-phase bridge rectifier
    # 3rd harmonic concentrated near +π/4 (peak-conduction near voltage peak)
    phi3 = vonmises.rvs(kappa=3.0, loc=np.pi/4,  random_state=rng)
    phi5 = vonmises.rvs(kappa=2.5, loc=-np.pi/5, random_state=rng)
    phi7 = vonmises.rvs(kappa=2.0, loc=np.pi/7,  random_state=rng)
    r3 = rng.uniform(0.30, 0.90)
    r5 = rng.uniform(0.10, 0.40)
    r7 = rng.uniform(0.05, 0.20)
    # Inject into voltage (small) and current (large — SMPS draws distorted current)
    v_sig += (r3 * 0.05 * V1) * np.sin(2*np.pi*150*t + phi3)
    v_sig += (r5 * 0.02 * V1) * np.sin(2*np.pi*250*t + phi5)
    i_sig += (r3 * I1)        * np.sin(2*np.pi*150*t + phi3)
    i_sig += (r5 * I1)        * np.sin(2*np.pi*250*t + phi5)
    i_sig += (r7 * I1 * 0.5)  * np.sin(2*np.pi*350*t + phi7)

elif subtype == 1:  # 6-pulse VFD rectifier
    # 5th and 7th harmonics locked to firing angle, triplen harmonics suppressed
    phi5 = vonmises.rvs(kappa=4.0, loc=-2*np.pi/5, random_state=rng)
    phi7 = vonmises.rvs(kappa=4.0, loc=2*np.pi/7,  random_state=rng)
    r5 = rng.uniform(0.15, 0.25)
    r7 = rng.uniform(0.08, 0.15)
    v_sig += (r5 * 0.03 * V1) * np.sin(2*np.pi*250*t + phi5)
    v_sig += (r7 * 0.02 * V1) * np.sin(2*np.pi*350*t + phi7)
    i_sig += (r5 * I1)        * np.sin(2*np.pi*250*t + phi5)
    i_sig += (r7 * I1)        * np.sin(2*np.pi*350*t + phi7)

else:               # Transformer saturation
    # 3rd harmonic near +π/2 (in quadrature — saturation at voltage zero-crossing flux peak)
    phi3 = vonmises.rvs(kappa=3.5, loc=np.pi/2, random_state=rng)
    r3 = rng.uniform(0.05, 0.20)   # lower THD than SMPS
    v_sig += (r3 * V1) * np.sin(2*np.pi*150*t + phi3)
    i_sig += (r3 * I1 * 0.8) * np.sin(2*np.pi*150*t + phi3)

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

#### Class 5 — Transient

```python
V1    = rng.uniform(200, 245) * np.sqrt(2)
phi1  = rng.uniform(0, 2*np.pi)
v_sig = V1 * np.sin(2*np.pi*50*t + phi1)

A_t     = rng.uniform(0.1, 1.5) * V1
tau     = rng.uniform(0.1, 2.0) / 1000.0      # decay time 0.1–2 ms
f_t     = rng.uniform(500, 2000)               # oscillation 500–2000 Hz
t_start = rng.uniform(0.0, (N/FS) - 0.005)     # inject within 100ms frame, avoid right-edge truncation
polarity = rng.choice([-1, 1])

t_rel   = t - t_start
transient = np.where(
    t_rel >= 0,
    polarity * A_t * np.exp(-t_rel / tau) * np.sin(2*np.pi*f_t*t_rel),
    0.0
)
v_sig += transient

I1    = rng.uniform(1.0, 15.0)
i_sig = I1 * np.sin(2*np.pi*50*t + phi1)      # current mostly unaffected for voltage transient

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

#### Class 6 — Flicker

```python
V1    = rng.uniform(200, 245) * np.sqrt(2)
phi1  = rng.uniform(0, 2*np.pi)

m    = rng.uniform(0.05, 0.20)    # modulation depth 5–20%
f_f  = rng.uniform(1.0, 25.0)    # flicker frequency 1–25 Hz (human perception range)

# Amplitude modulation of the fundamental
v_sig = V1 * (1 + m * np.sin(2*np.pi*f_f*t)) * np.sin(2*np.pi*50*t + phi1)

I1    = rng.uniform(1.0, 15.0)
dpf   = rng.uniform(0.85, 1.0)
# Current flickers in proportion to voltage
i_sig = I1 * (1 + m * np.sin(2*np.pi*f_f*t)) * np.sin(2*np.pi*50*t + phi1 - np.arccos(dpf))

v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)
```

**Note on Flicker:** The frame is 100 ms (5 cycles). Flicker modulation at 1–25 Hz is only partially observable in one frame, especially near 1–5 Hz, so this remains a known limitation. The classifier may still extract weak cues via crest factor, peak-to-peak, and DWT energy asymmetry, but do not claim full IEC-style flickermeter behavior from single-frame inference.

### 3.4 Additional Train-Time Augmentation

Base class generators above already include light AWGN. On top of that, apply additional augmentation **at training time only** (not to validation/test) to improve robustness. For each training sample during dataset loading, randomly apply one of:

```python
snr_db = rng.choice([20, 30, 40, 50, np.inf])
if snr_db < np.inf:
    signal_power = np.mean(v_sig**2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    v_sig += rng.normal(0, np.sqrt(noise_power), N)
    # Same for i_sig proportionally

# Amplitude jitter ±5%
amp_scale = rng.uniform(0.95, 1.05)
v_sig *= amp_scale

# Phase offset jitter ±10° (affects all harmonics equally via time shift)
phase_jitter = rng.uniform(-np.pi/18, np.pi/18)   # ±10 degrees
t_jitter = phase_jitter / (2 * np.pi * 50)
v_sig = np.interp(t + t_jitter, t, v_sig, period=(N/FS))
```

Apply augmentation to training split only. Validation and test splits use clean signals (no augmentation) for reproducible metric tracking.

### 3.5 Feature Extraction and Storage

For each generated `(v_sig, i_sig)` pair:

```python
from src.dsp.features import extract_features
from src.dsp.preprocess import remove_dc_offset

v_dc, i_dc = remove_dc_offset(v_sig, i_sig)   # remove mean (already ~0 for generated signals)
feat_vec    = extract_features(v_dc, i_dc)     # returns float32[282]
```

Store the dataset as numpy arrays:

```python
# artifacts/datasets/synth_v1/
X_wave_train.npy    # shape: (28000, 500, 2)  — stacked [v_norm, i_norm]
X_feat_train.npy    # shape: (28000, 282)     — feature vectors
y_train.npy         # shape: (28000,)          — integer class labels 0–6

# Same pattern for val and test
X_wave_val.npy, X_feat_val.npy, y_val.npy
X_wave_test.npy, X_feat_test.npy, y_test.npy

# Metadata
metadata.json       # generation params, seed, counts per class, file hashes
```

For waveform storage, normalize before saving:
```python
v_norm = v_dc / (np.max(np.abs(v_dc)) + 1e-8)
i_norm = i_dc / (np.max(np.abs(i_dc)) + 1e-8)
X_wave[idx] = np.stack([v_norm, i_norm], axis=-1)   # shape (500, 2)
```

### 3.6 Scaler Fitting

Fit scalers **only on the training split** — never on validation or test:

```python
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

scaler_mag      = StandardScaler()  # for 28-feature branch (24:52)
scaler_phase    = StandardScaler()  # for 254-feature branch
scaler_mag_full = StandardScaler()  # for 52-feature M1/M3 branch (0:52)

X_mag_train   = X_feat_train[:, 24:52]     # shape (28000, 28)
X_phase_train = np.concatenate([
    X_feat_train[:, 0:24],
    X_feat_train[:, 52:210],
    X_feat_train[:, 210:282],
], axis=1)                                   # shape (28000, 254)
X_mag_full_train = X_feat_train[:, 0:52]     # shape (28000, 52)

scaler_mag.fit(X_mag_train)
scaler_phase.fit(X_phase_train)
scaler_mag_full.fit(X_mag_full_train)

scalers_dir = Path("artifacts/scalers")
scalers_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler_mag,   scalers_dir / "scaler_mag_v1.pkl")
joblib.dump(scaler_phase, scalers_dir / "scaler_phase_v1.pkl")
joblib.dump(scaler_mag_full, scalers_dir / "scaler_mag_full_v1.pkl")
```

Then apply transforms to all splits:
```python
X_mag_train_s   = scaler_mag.transform(X_mag_train)
X_phase_train_s = scaler_phase.transform(X_phase_train)
X_mag_full_train_s = scaler_mag_full.transform(X_mag_full_train)
# Repeat for val and test splits using transform (NOT fit_transform)
```

### 3.7 Metadata File Format

`artifacts/datasets/synth_v1/metadata.json`:

```json
{
  "version": "synth_v1",
  "seed": 42,
  "fs_hz": 5000,
  "samples_per_frame": 500,
  "n_classes": 7,
  "class_names": ["Normal","Sag","Swell","Interruption","HarmonicDistortion","Transient","Flicker"],
  "samples_per_class_total": 5500,
  "split": {
    "train": 4000,
    "val": 750,
    "test": 750
  },
  "total_samples": 38500,
  "counts_per_class_train": {"0":4000,"1":4000,"2":4000,"3":4000,"4":4000,"5":4000,"6":4000},
  "feature_length": 282,
  "generated_at": "<ISO timestamp>"
}
```

### 3.8 Chunk 4 Acceptance Criteria

- [ ] `artifacts/datasets/synth_v1/X_feat_train.npy` exists, shape `(28000, 282)`
- [ ] `artifacts/datasets/synth_v1/X_wave_train.npy` exists, shape `(28000, 500, 2)`
- [ ] Per-class counts in train are exactly 4000 each
- [ ] Re-running generator with same seed produces byte-identical arrays
- [ ] `artifacts/scalers/scaler_mag_v1.pkl`, `scaler_phase_v1.pkl`, and `scaler_mag_full_v1.pkl` exist and load
- [ ] `tests/test_dataset_balance.py` passes: checks counts, shapes, no NaN/Inf in features
- [ ] `metadata.json` exists and is valid JSON with correct counts

---

## 4. Chunk 5 — Model Architectures M1–M4

**Files:** `src/models/m1_baseline.py`, `m2_waveform.py`, `m3_waveform_mag.py`, `m4_phase_aware.py`, `factory.py`
**Test:** `tests/test_model_shapes.py`

All models output a 7-class softmax probability vector. All use `categorical_crossentropy` loss. All are implemented in TensorFlow/Keras functional API.

### 4.1 M1 — Magnitude-Only MLP (Baseline)

**Purpose:** Lower-bound baseline. Conventional magnitude-centric approach. No raw waveform, no phase, no DWT.

**Input:** `X_mag_scaled` — shape `(batch, 52)` — time-domain stats (24) + FFT mags (26) + THD (2)

```python
# NOTE: M1 uses the full 52-feature magnitude+time-domain block, not just 28.
# This is the "magnitude-only" baseline as described: time-domain stats + FFT mags + THD.
# Slice it as: X_full[:, 0:52]

from tensorflow.keras import layers, Model, Input

def build_m1(n_mag_feats=52, n_classes=7):
    x_in  = Input(shape=(n_mag_feats,), name="magnitude_input")
    x     = layers.Dense(128, activation="relu")(x_in)
    x     = layers.Dropout(0.3)(x)
    x     = layers.Dense(64, activation="relu")(x)
    x     = layers.Dropout(0.3)(x)
    out   = layers.Dense(n_classes, activation="softmax", name="class_output")(x)
    return Model(inputs=x_in, outputs=out, name="M1_MagnitudeMLP")
```

**Parameters:** ~16,000. Trains in minutes on CPU.

### 4.2 M2 — Raw Waveform CNN-LSTM Only

**Purpose:** Evaluates what the raw waveform alone can achieve. No engineered features.

**Input:** `X_wave` — shape `(batch, 500, 2)` — normalized [v_norm, i_norm]

```python
def build_m2(n_samples=500, n_channels=2, n_classes=7):
    x_in = Input(shape=(n_samples, n_channels), name="waveform_input")

    x = layers.Conv1D(32, 7, padding="same", use_bias=False)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)                 # → (125, 32)

    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)                 # → (31, 64)

    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(2)(x)                 # → (15, 128)

    x   = layers.LSTM(64, return_sequences=False)(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_output")(x)

    return Model(inputs=x_in, outputs=out, name="M2_RawWaveformCNNLSTM")
```

**Parameters:** ~130,000.

### 4.3 M3 — CNN-LSTM + Magnitude (Phase-Free Upper Bound)

**Purpose:** Best achievable without phase features. The comparison M3 → M4 isolates the phase contribution.

**Inputs:**
- `X_wave` — shape `(batch, 500, 2)`
- `X_mag_scaled` — shape `(batch, 52)` — time-domain + FFT + THD (full 52-feature block, scaled)

```python
def build_m3(n_samples=500, n_channels=2, n_mag_feats=52, n_classes=7):
    # Branch 1: CNN-LSTM (identical to M2)
    waveform_in = Input(shape=(n_samples, n_channels), name="waveform_input")
    x = layers.Conv1D(32, 7, padding="same", use_bias=False)(waveform_in)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(2)(x)
    branch1 = layers.Dropout(0.3)(layers.LSTM(64)(x))

    # Branch 2: Magnitude MLP
    mag_in   = Input(shape=(n_mag_feats,), name="magnitude_input")
    y        = layers.Dense(64, activation="relu")(mag_in)
    y        = layers.Dropout(0.2)(y)
    branch2  = layers.Dense(32, activation="relu")(y)

    # Fusion
    fused = layers.Concatenate()([branch1, branch2])
    out   = layers.Dense(128, activation="relu")(fused)
    out   = layers.Dropout(0.4)(out)
    out   = layers.Dense(64, activation="relu")(out)
    out   = layers.Dense(n_classes, activation="softmax", name="class_output")(out)

    return Model(inputs=[waveform_in, mag_in], outputs=out, name="M3_CNNLSTMplusMag")
```

**Parameters:** ~180,000.

### 4.4 M4 — Full Phase-Aware Hybrid (The Proposed Model)

**Purpose:** Full model with all three branches. This is the main contribution. The accuracy delta M4 − M3 is the project's key result.

**Inputs:**
- `X_wave` — shape `(batch, 500, 2)` — normalized waveform
- `X_mag_scaled` — shape `(batch, 28)` — FFT magnitudes + THD only (indices 24:52 of feature vector)
- `X_phase_scaled` — shape `(batch, 254)` — phase + DWT + time-domain (concatenation of 0:24, 52:210, 210:282)

```python
def build_m4(n_samples=500, n_channels=2, n_mag_feats=28, n_phase_dwt_feats=254, n_classes=7):
    # ── Branch 1: Raw Waveform CNN-LSTM ──────────────────────────────────────
    waveform_in = Input(shape=(n_samples, n_channels), name="waveform_input")
    x = layers.Conv1D(32, 7, padding="same", use_bias=False)(waveform_in)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)                  # → (125, 32)
    x = layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(4)(x)                  # → (31, 64)
    x = layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(2)(x)                  # → (15, 128)
    x = layers.LSTM(64, return_sequences=False)(x)
    branch1 = layers.Dropout(0.3)(x)            # output: (64,)

    # ── Branch 2: Magnitude-Only MLP ─────────────────────────────────────────
    mag_in  = Input(shape=(n_mag_feats,), name="magnitude_input")
    y       = layers.Dense(64, activation="relu")(mag_in)
    y       = layers.Dropout(0.2)(y)
    branch2 = layers.Dense(32, activation="relu")(y)   # output: (32,)

    # ── Branch 3: Phase + DWT + Time-Domain MLP ───────────────────────────────
    phase_in = Input(shape=(n_phase_dwt_feats,), name="phase_dwt_input")
    z        = layers.Dense(128, activation="relu")(phase_in)
    z        = layers.Dropout(0.3)(z)
    branch3  = layers.Dense(64, activation="relu")(z)  # output: (64,)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused = layers.Concatenate()([branch1, branch2, branch3])  # (160,)
    out   = layers.Dense(128, activation="relu")(fused)
    out   = layers.Dropout(0.4)(out)
    out   = layers.Dense(64, activation="relu")(out)
    out   = layers.Dense(n_classes, activation="softmax", name="class_output")(out)

    return Model(
        inputs=[waveform_in, mag_in, phase_in],
        outputs=out,
        name="M4_PhaseAwareHybrid"
    )
```

**Parameters:** ~280,000. Trains in under 1 hour on a laptop CPU for 38,500 samples.

### 4.5 Model Factory

```python
# src/models/factory.py
from src.models.m1_baseline     import build_m1
from src.models.m2_waveform     import build_m2
from src.models.m3_waveform_mag import build_m3
from src.models.m4_phase_aware  import build_m4

VARIANTS = {
    "M1": build_m1,
    "M2": build_m2,
    "M3": build_m3,
    "M4": build_m4,
}

def build_model(variant: str, **kwargs):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANTS)}")
    return VARIANTS[variant](**kwargs)
```

### 4.6 Model Input Preparation Helper

Put this in `src/data/splits.py` so both train and inference use identical slicing:

```python
import numpy as np

def get_model_inputs(X_feat: np.ndarray, X_wave: np.ndarray,
                     scaler_mag, scaler_phase, scaler_mag_full, variant: str):
    """
    Given raw feature matrix and waveform matrix, returns the correct
    tuple of inputs for the specified model variant.

    X_feat  : float32 (batch, 282)  — raw feature vectors (unscaled)
    X_wave  : float32 (batch, 500, 2) — normalized waveforms [v_norm, i_norm]
    variant : "M1", "M2", "M3", or "M4"
    """
    X_mag_raw   = X_feat[:, 24:52]
    X_phase_raw = np.concatenate([
        X_feat[:, 0:24],
        X_feat[:, 52:210],
        X_feat[:, 210:282],
    ], axis=1)
    X_mag_full  = X_feat[:, 0:52]   # for M1 and M3 (52 features)

    X_mag_s      = scaler_mag.transform(X_mag_raw)
    X_phase_s    = scaler_phase.transform(X_phase_raw)
    X_mag_full_s = scaler_mag_full.transform(X_mag_full)

    if variant == "M1":
        return X_mag_full_s
    elif variant == "M2":
        return X_wave
    elif variant == "M3":
        return [X_wave, X_mag_full_s]
    elif variant == "M4":
        return [X_wave, X_mag_s, X_phase_s]
    else:
        raise ValueError(f"Unknown variant {variant!r}")
```

### 4.7 Chunk 5 Acceptance Criteria

- [ ] Each model builds without error: `build_m1()`, `build_m2()`, `build_m3()`, `build_m4()`
- [ ] `tests/test_model_shapes.py` passes:
  - M1: single forward pass with input `(1, 52)` → output `(1, 7)`, all probs sum to 1
  - M2: single forward pass with `(1, 500, 2)` → `(1, 7)`
  - M3: inputs `[(1,500,2), (1,52)]` → `(1,7)`
  - M4: inputs `[(1,500,2), (1,28), (1,254)]` → `(1,7)`
- [ ] `model.summary()` exported to `artifacts/models/<variant>_summary.txt`
- [ ] All four models importable from `src.models.factory.build_model`

---

## 5. Chunk 6 — Training Pipeline

**Files:** `src/train/train.py`, `src/eval/evaluate.py`
**Test:** `tests/test_train_smoke.py`

### 5.1 Training Configuration

```python
# src/train/train.py

BATCH_SIZE  = 64
MAX_EPOCHS  = 100
LR_INIT     = 1e-3
LR_MIN      = 1e-6

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=LR_MIN,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INIT),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_inputs,
    y_train_onehot,
    validation_data=(val_inputs, y_val_onehot),
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
)
```

### 5.2 Label Encoding

```python
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_val_onehot   = tf.keras.utils.to_categorical(y_val,   num_classes=7)
y_test_onehot  = tf.keras.utils.to_categorical(y_test,  num_classes=7)
```

### 5.3 Run Artifact Structure

Every training run creates a timestamped directory:

```
artifacts/runs/<variant>_<YYYYMMDD_HHMMSS>/
├── best_model.keras           # saved by ModelCheckpoint
├── final_model.keras          # saved after training completes (may differ if early stopped)
├── scaler_mag.pkl             # copy from artifacts/scalers/ (for self-contained run)
├── scaler_phase.pkl
├── scaler_mag_full.pkl
├── training_history.json      # history.history dict
├── training_curves.png        # accuracy + loss plots
├── confusion_matrix_test.png  # on held-out test split
├── classification_report.txt  # sklearn text report
├── manifest.json              # run metadata
└── tb_logs/                   # TensorBoard (optional)
```

### 5.4 Run Manifest Format

```json
{
  "run_id": "M4_20260415_143022",
  "variant": "M4",
  "dataset_version": "synth_v1",
  "config_snapshot": { "batch_size": 64, "lr_init": 0.001, "max_epochs": 100 },
  "epochs_trained": 47,
  "best_val_accuracy": 0.9712,
  "test_accuracy": 0.9689,
  "per_class_f1": {"Normal": 0.98, "Sag": 0.97, "Swell": 0.94,
                    "Interruption": 0.99, "HarmonicDistortion": 0.95,
                    "Transient": 0.98, "Flicker": 0.92},
  "training_time_seconds": 2341,
  "model_params": 279842,
  "scaler_mag_path": "artifacts/scalers/scaler_mag_v1.pkl",
    "scaler_phase_path": "artifacts/scalers/scaler_phase_v1.pkl",
    "scaler_mag_full_path": "artifacts/scalers/scaler_mag_full_v1.pkl"
}
```

### 5.5 Evaluation Script

```python
# src/eval/evaluate.py

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

CLASS_NAMES = ["Normal","Sag","Swell","Interruption","HarmonicDistortion","Transient","Flicker"]

def evaluate_model(model, test_inputs, y_test_onehot, run_dir, variant):
    y_pred_probs = model.predict(test_inputs, batch_size=64)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {variant}")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix_test.png", dpi=150)
    plt.close()

    return report
```

### 5.6 Training Curves Plot

```python
def plot_history(history_dict, run_dir, variant):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history_dict["accuracy"]) + 1)

    ax1.plot(epochs, history_dict["accuracy"],     label="Train", color="blue")
    ax1.plot(epochs, history_dict["val_accuracy"], label="Val",   color="orange")
    ax1.set_title(f"{variant} — Accuracy"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs, history_dict["loss"],     label="Train", color="red")
    ax2.plot(epochs, history_dict["val_loss"], label="Val",   color="green")
    ax2.set_title(f"{variant} — Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close()
```

### 5.7 Smoke Test

`tests/test_train_smoke.py` must do a fast sanity run: 2 epochs on a 200-sample subset of the synthetic dataset, verify the model runs, losses decrease, and artifacts are written.

### 5.8 Chunk 6 Acceptance Criteria

- [ ] `python src/train/train.py --variant M4` completes without error
- [ ] `artifacts/runs/M4_<timestamp>/best_model.keras` exists and is loadable
- [ ] `confusion_matrix_test.png` and `training_curves.png` exist in run directory
- [ ] `manifest.json` exists with correct fields
- [ ] `tests/test_train_smoke.py` passes (2-epoch subset run)

---

## 6. Chunk 7 — Ablation Study

**File:** `src/eval/ablation.py`
**Output:** `artifacts/ablation/`

### 6.1 Ablation Matrix

Run all four variants on the **same fixed splits** (synth_v1 from Chunk 4):

| Variant | Waveform Branch | Magnitude Branch | Phase+DWT Branch | Primary input |
|---|---|---|---|---|
| M1 | ✗ | ✓ (52 feat) | ✗ | time+mag features only |
| M2 | ✓ | ✗ | ✗ | raw waveform only |
| M3 | ✓ | ✓ (52 feat) | ✗ | waveform + magnitude |
| M4 | ✓ | ✓ (28 feat) | ✓ (254 feat) | all three branches |

### 6.2 Key Comparison

The **primary scientific result** of the project is:

```
Phase contribution = M4_test_accuracy − M3_test_accuracy
```

This must be positive and statistically meaningful. If it is less than 0.5 percentage points on the synthetic test set, flag it and investigate the dataset — the Von Mises distributions may be too broad (reduce κ values) or the phase features may not have sufficient variance across classes.

### 6.3 Output Files

```
artifacts/ablation/
├── ablation_results.csv        # rows: variant, val_acc, test_acc, per-class F1
├── ablation_bar_chart.png      # grouped bar chart: test accuracy per variant
├── ablation_report.md          # human-readable summary table with delta annotations
└── confusion_matrices/
    ├── M1_confusion_test.png
    ├── M2_confusion_test.png
    ├── M3_confusion_test.png
    └── M4_confusion_test.png
```

### 6.4 `ablation_results.csv` Format

```csv
variant,val_accuracy,test_accuracy,normal_f1,sag_f1,swell_f1,interruption_f1,harmonic_f1,transient_f1,flicker_f1
M1,0.9123,0.9089,...
M2,0.9401,0.9387,...
M3,0.9612,0.9598,...
M4,0.9734,0.9712,...
```

### 6.5 `ablation_report.md` Format

The report must include this table and explicitly call out the phase delta:

```
| Model | Val Acc | Test Acc | Δ vs M3 |
|-------|---------|----------|---------|
| M1    | 91.2%   | 90.9%    | −6.8%   |
| M2    | 94.0%   | 93.9%    | −2.7%   |
| M3    | 96.1%   | 96.0%    | baseline|
| M4    | 97.3%   | 97.1%    | **+1.1%**|

Phase feature contribution (M4 − M3): +1.1 percentage points on test set.
```

### 6.6 Chunk 7 Acceptance Criteria

- [ ] All 4 variants trained and evaluated on identical splits
- [ ] `ablation_results.csv` exists with all 4 rows
- [ ] `ablation_report.md` exists with delta highlighted
- [ ] All 4 confusion matrices saved
- [ ] M4 test accuracy > M3 test accuracy (phase features must help)

---

## 7. Expected Performance Numbers

Do not fake these. These are calibrated estimates from the literature for your architecture class — report what you actually measure, and include these as context:

| Model | Expected Synthetic (noiseless) | Expected at 30dB SNR | Note |
|---|---|---|---|
| M1 | 88–93% | 82–87% | Magnitude-only lower bound |
| M2 | 93–97% | 88–92% | Waveform captures shape |
| M3 | 95–98% | 91–94% | Phase-free upper bound |
| M4 | 96–99% | 93–96% | Phase contribution: +1–3% vs M3 |

If your numbers are significantly below these ranges, likely causes:
1. Von Mises concentration (κ) too low → phase features not discriminative enough
2. Harmonic Distortion class subtypes not well-separated in feature space
3. Dataset imbalance (recheck per-class counts)
4. Learning rate too high → training instability

If M4 < M3 (phase hurts accuracy), likely causes:
1. Von Mises κ too high → over-concentrated → model overfits to specific phase angles that don't generalize
2. Phase feature slicing is wrong (re-verify the 52:210 indices)
3. Phase branch has too many parameters relative to dataset size → increase dropout to 0.4

---

## 8. File Naming and Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py   ← Chunk 4
│   └── splits.py                ← Chunk 4
├── models/
│   ├── __init__.py
│   ├── m1_baseline.py           ← Chunk 5
│   ├── m2_waveform.py           ← Chunk 5
│   ├── m3_waveform_mag.py       ← Chunk 5
│   ├── m4_phase_aware.py        ← Chunk 5
│   └── factory.py               ← Chunk 5
├── train/
│   ├── __init__.py
│   └── train.py                 ← Chunk 6
└── eval/
    ├── __init__.py
    ├── evaluate.py              ← Chunk 6
    └── ablation.py              ← Chunk 7

tests/
├── test_dataset_balance.py      ← Chunk 4
├── test_model_shapes.py         ← Chunk 5
└── test_train_smoke.py          ← Chunk 6

artifacts/
├── datasets/
│   └── synth_v1/                ← Chunk 4 output
├── scalers/                     ← Chunk 4 output
├── models/                      ← summaries
├── runs/                        ← Chunk 6 output
└── ablation/                    ← Chunk 7 output
```

---

## 9. What to Take from Legacy Code and What to Ignore

### Take (copy-paste and adapt):

From `legacy/Single_signal_LM/model_1/fft_dwt.py`:
- The **sag/swell/interruption mask pattern** (lines 116–139) — solid physics, adapt to `N=500, t=linspace(0, N/FS, N)`
- The **transient exponential decay model** (lines 140–147) — correct, adapt frequency range
- The **Gaussian noise injection pattern** using SNR-based power scaling (lines 156–162) — correct and reusable

From `legacy/Single_signal_LM/model_2/cnn.py`:
- `EarlyStopping + ModelCheckpoint` callback pattern (lines 154–168) — use as-is
- `tf.data.Dataset.shuffle + repeat + batch + prefetch` pipeline pattern — use as-is
- **Custom class weights concept** — you probably won't need it with balanced dataset, but keep in mind if Flicker confusion is high

### Ignore entirely:

- All architecture code from both models — wrong input shapes, wrong approach for phase analysis
- All `get_fft_spectrogram()` / `get_swt_scalogram()` functions — these produce 2D image representations incompatible with the current spec
- All class label definitions — both models have `Notch` as Class 6, not `Flicker`
- All sampling rate / window length assumptions — both use `fs=6000, N=6000`, wrong
- All signal amplitude values — `V1 = 5V` is not physical 230V mains

---

## 10. Critical Rules — Do Not Violate

1. **Never change `fs=5000`, `N=500`, or the class order.** These are locked in the config and the hardware. Changing them silently breaks the entire pipeline.

2. **Never use `np.mean()` or `np.std()` on raw phase angle arrays.** Phase angles wrap at ±π. Always use `scipy.stats.circmean(angles, high=np.pi, low=-np.pi)` and `circstd`. This is already done correctly in `src/dsp/features.py` — do not regress this in your dataset generator.

3. **Never inject harmonics with uniform random phase.** The entire point of the Harmonic Distortion class is that it uses Von Mises distributions with physically motivated parameters. If you use `np.random.uniform(0, 2*np.pi)` for harmonic phases, the phase features are uninformative and M4 will not outperform M3 — which makes the project's core claim false.

4. **Never fit scalers on validation or test data.** `scaler.fit_transform()` on training only. `scaler.transform()` on val and test. Fitting on test data leaks information and produces inflated accuracy numbers.

5. **Never hardcode paths.** Always use `Path(cfg["paths"]["artifacts_root"])` and its children.

6. **Always save confusion matrices per model.** The evaluator will look for these to assess per-class performance. A 97% overall accuracy that is 60% on Flicker is not acceptable — it needs to be visible in the confusion matrix.

7. **The test split is held out.** During ablation, never tune hyperparameters based on test accuracy. Tune on validation. Report test accuracy as a final number only.

8. **Fixed seed everywhere:**
   ```python
   import numpy as np, tensorflow as tf, random, os
   SEED = 42
   np.random.seed(SEED)
   tf.random.set_seed(SEED)
   random.seed(SEED)
   os.environ["PYTHONHASHSEED"] = str(SEED)
   ```
   Put this at the top of `train.py` and `synthetic_generator.py`.

---

## 11. How to Run Everything

After implementing all chunks, the complete ML pipeline runs as:

```bash
# From project root, with .venv activated

# Step 1: Generate dataset (takes ~5–15 min depending on CPU)
python -m src.data.synthetic_generator --output artifacts/datasets/synth_v1

# Step 2: Train all variants (can parallelize or run sequentially)
python src/train/train.py --variant M1
python src/train/train.py --variant M2
python src/train/train.py --variant M3
python src/train/train.py --variant M4

# Step 3: Run ablation (collects results from all runs and generates report)
python src/eval/ablation.py --runs-dir artifacts/runs --output artifacts/ablation

# Step 4: Verify
python -m pytest tests/test_dataset_balance.py tests/test_model_shapes.py tests/test_train_smoke.py -v
```

---

## 12. Definition of Done for Your Chunks

You are done when ALL of the following are true:

- [ ] `artifacts/datasets/synth_v1/` contains all 6 `.npy` files with correct shapes
- [ ] `artifacts/datasets/synth_v1/metadata.json` is valid and counts match
- [ ] `artifacts/scalers/scaler_mag_v1.pkl`, `scaler_phase_v1.pkl`, and `scaler_mag_full_v1.pkl` loadable
- [ ] All 4 models build and accept correct input shapes (verified by test)
- [ ] All 4 training runs complete and produce artifacts in `artifacts/runs/`
- [ ] M4 test accuracy > M3 test accuracy (phase features help, even if by a small margin)
- [ ] `artifacts/ablation/ablation_results.csv` has all 4 rows
- [ ] `artifacts/ablation/ablation_report.md` exists with delta highlighted
- [ ] All 4 confusion matrices saved
- [ ] `tests/test_dataset_balance.py`, `tests/test_model_shapes.py`, `tests/test_train_smoke.py` all pass
- [ ] No hardcoded constants — everything through config
- [ ] `manifest.json` in each run directory with test accuracy logged

---

*This document is the single source of truth for the ML subsystem. If anything in this document conflicts with `legacy/`, trust this document. If anything conflicts with `prd.md`, trust `prd.md`. If anything conflicts with `configs/default.yaml`, trust `default.yaml`.*
