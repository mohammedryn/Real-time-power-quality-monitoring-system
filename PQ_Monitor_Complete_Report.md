# PQ_Monitor_Complete_Report

# Phase-Aware Deep Learning Framework for Real-Time Power Quality Monitoring in Low-Voltage Industrial Networks

**Comprehensive Project Report — B.Tech Mini ProjectBranch:** Electronics & Instrumentation Engineering
**Institution:** Ramaiah Institute of Technology (MSRIT), Bengaluru
**Academic Year:** 2025–2026
**Document Version:** 2.0 (Audited & Corrected)
**Date:** March 2026
**Status:** Final — Submission Ready

---

## Team

| Name | USN | Role | Primary Responsibility |
| --- | --- | --- | --- |
| Mohammed Rayan | — | Power Quality, Harmonics & Embedded Lead | Standards, theory, real-data validation, , firmware, ADC interface |
| Yash | — | Hardware Lead | Circuit design interface |
| Yashas | — | DSP & Feature Engineering Lead | FFT pipeline, phase features, wavelet features |
| Japeth | — | Deep Learning Lead | Model architecture, training, ablation studies |

---

## Table of Contents

1. [Abstract](about:blank#1-abstract)
2. [Introduction](about:blank#2-introduction)
3. [Problem Statement and Objectives](about:blank#3-problem-statement-and-objectives)
4. [Literature Review](about:blank#4-literature-review)
5. [Standards and Regulatory Context](about:blank#5-standards-and-regulatory-context)
6. [Theoretical Background](about:blank#6-theoretical-background)
7. [System Architecture Overview](about:blank#7-system-architecture-overview)
8. [Hardware Design](about:blank#8-hardware-design)
9. [Firmware — Teensy 4.1 Embedded Software](about:blank#9-firmware--teensy-41-embedded-software)
10. [Signal Processing and Feature Extraction Pipeline](about:blank#10-signal-processing-and-feature-extraction-pipeline)
11. [Deep Learning Model](about:blank#11-deep-learning-model)
12. [Dataset Generation](about:blank#12-dataset-generation)
13. [Training Procedure and Evaluation](about:blank#13-training-procedure-and-evaluation)
14. [Experimental Plan and Demo Setup](about:blank#14-experimental-plan-and-demo-setup)
15. [Hardware Design Audit — All Issues, Root Causes, and Fixes](about:blank#15-hardware-design-audit--all-issues-root-causes-and-fixes)
16. [Complete Bill of Materials](about:blank#16-complete-bill-of-materials)
17. [Complete Wiring Diagram and Connections](about:blank#17-complete-wiring-diagram-and-connections)
18. [Safety Guidelines](about:blank#18-safety-guidelines)
19. [Implementation Timeline](about:blank#19-implementation-timeline)
20. [Known Limitations and Future Extensions](about:blank#20-known-limitations-and-future-extensions)
21. [References](about:blank#21-references)

---

## 1. Abstract

This project presents the complete design and implementation of a low-cost, hardware-enabled, real-time system for detection and classification of power quality (PQ) disturbances in low-voltage (LV) 230 V / 50 Hz industrial networks. The hardware sensing node uses a Texas Instruments AMC1301 reinforced isolated differential amplifier preceded by a precision 2.2 MΩ / 560 Ω resistor voltage divider to safely measure the 230 V mains waveform, and an Allegro ACS758LCB-050B Hall-effect current sensor to simultaneously capture load current. Both analog channels are sampled synchronously at 5 kHz using the dual-ADC simultaneous hardware mode of the PJRC Teensy 4.1 microcontroller (Cortex-M7 @ 600 MHz), giving exactly N = 500 samples per 50 Hz cycle — a deliberate design choice that places every harmonic (50, 100, 150, … Hz) on an exact integer FFT bin, eliminating spectral leakage entirely for stationary periodic signals.

Captured 500-sample dual-channel frames are streamed via USB to a host PC where a multi-stage digital signal processing pipeline extracts: time-domain statistical features (RMS, crest factor, kurtosis, etc.), FFT-based harmonic phasors (magnitude and phase, orders 1–13, for both V and I), phase-aware inter-harmonic coupling features using circular-statistics-correct sin/cos encoding, and discrete wavelet transform (DWT) time–frequency subband features. The resulting 282-element (after removing redundant raw-angle features) standardised feature vector feeds a hybrid phase-aware CNN–LSTM deep learning classifier trained on 38,500 synthetically generated, noise-augmented samples across seven power quality disturbance classes: Normal, Sag, Swell, Interruption, Harmonic Distortion, Transient, and Flicker.

Four model variants (M1 through M4) are trained in a structured ablation study to directly quantify the accuracy improvement from phase features. Domain adaptation via fine-tuning the last two dense layers on 100–200 real hardware samples per class bridges the synthetic-to-real accuracy gap, which from comparable published work is typically 10–15 percentage points before adaptation. Total hardware cost for the complete analog sensing frontend is under ₹700.

The project’s core scientific contribution is demonstrating that harmonic phase angles carry load-type-specific signatures that are statistically distinct even when harmonic magnitudes are nearly identical — and that a deep learning model trained on phase-aware features can exploit this information to improve classification accuracy by 1–3% overall and substantially more for hard-to-distinguish class pairs.

---

## 2. Introduction

### 2.1 The Power Quality Problem in Modern Industry

Modern industrial and commercial facilities depend heavily on power-electronic equipment: variable-frequency drives (VFDs) for pumps, fans, and CNC machines; rectifier-front-end uninterruptible power supplies (UPS); switched-mode power supplies (SMPS) in computers, servers, and LED drivers; and increasingly, electric vehicle (EV) chargers. All of these devices are **nonlinear loads**: unlike a simple resistive heater or incandescent lamp, they draw current in short, sharp pulses rather than as a smooth sinusoidal waveform.

The governing physics is straightforward. The utility provides a nominally sinusoidal voltage at 230 V RMS, 50 Hz. A linear load — one whose impedance is constant — draws a proportional sinusoidal current. A nonlinear load has a time-varying impedance (e.g., a diode bridge only conducts when the supply voltage exceeds the DC rail voltage), so it draws a current whose shape is not sinusoidal.

By Fourier’s theorem, any periodic waveform is the superposition of sinusoids at the fundamental frequency and its integer multiples — the **harmonics**. For a 50 Hz fundamental: the 2nd harmonic is at 100 Hz, the 3rd at 150 Hz, the 5th at 250 Hz, and so on. Nonlinear loads inject harmonic currents into the network; those currents flow through the finite impedances of cables, transformers, and busbars, creating harmonic voltage drops that distort the voltage seen by every load on the same network.

The consequences in a real industrial facility are significant and costly:

**Transformer overheating:** Harmonic currents increase copper losses (I²R) and core losses (eddy current losses scale with f²). A transformer carrying rated fundamental current plus significant harmonic content runs hotter than nameplate ratings indicate and fails prematurely.

**Neutral conductor overloading:** In a balanced three-phase system, fundamental-frequency phase currents cancel in the neutral. However, triplen harmonics (3rd, 9th, 15th — multiples of 3) do NOT cancel; they add arithmetically. A neutral sized for imbalance current may carry 1.7× the phase current when triplen harmonics are dominant — a fire hazard.

**Capacitor bank failures:** Power factor correction capacitors present a low impedance path to harmonic currents. The resulting high harmonic currents cause dielectric overheating, stress, and early failure. Resonance between the capacitor bank and network inductance at a harmonic frequency can amplify that harmonic by 5–10×.

**PLC and sensitive electronics malfunction:** Harmonic voltage distortion causes microcontroller-based equipment to malfunction, reset unexpectedly, or produce measurement errors — particularly in precision measurement instruments whose ADC reference is derived from the mains.

**Meter errors:** kWh meters calibrated for sinusoidal waveforms over-read or under-read when harmonics are significant, leading to billing disputes.

**Motor derating:** Negative-sequence harmonics (5th, 11th, 17th) create counter-rotating magnetic fields in AC induction motors, increasing losses, reducing torque, and requiring derating. At 5% 5th-harmonic voltage, derating to 90% of nameplate rating is typically required.

### 2.2 Why Monitoring Matters

The first step in addressing a power quality problem is measuring it accurately. Without quantitative data about the type, magnitude, duration, frequency of occurrence, and timing of disturbances, it is impossible to:

- Determine whether the disturbance originates from local loads (user-side) or the utility grid (supply-side) — critical for deciding who bears the cost of mitigation.
- Size and specify appropriate mitigation technology (passive harmonic filters, active harmonic filters, line reactors, isolation transformers, or dynamic voltage restorers).
- Verify compliance with applicable standards (IEEE 519-2022, EN 50160, IEC 61000-3-2) before and after mitigation.
- Monitor for gradual degradation that may precede equipment failures (predictive maintenance).
- Build an event log for insurance claims or utility negotiations.

### 2.3 The Gap: Phase Information is Universally Ignored

Conventional power quality monitors — including those meeting IEC 61000-4-30 Class A requirements (Hioki PQ3198, Fluke 435-II, Dranetz HDPQ) — measure harmonic **magnitudes** and total harmonic distortion (THD) with high accuracy. However, virtually all standard instruments and published research methods focus exclusively on magnitude spectra. The **phase angles** of individual harmonic components are discarded.

This is a meaningful loss of information. The phase structure of a harmonic spectrum is physically determined by the mechanism generating those harmonics, not random:

- A 3rd harmonic generated by a single-phase bridge rectifier (SMPS) has a phase relationship to the fundamental concentrated near ±45° because it is produced by the capacitor charge pulse at the voltage peak.
- A 3rd harmonic from a transformer saturating on the voltage peak has phase concentrated near ±90° (near the voltage zero crossing) because saturation occurs when flux — the integral of voltage — reaches its peak.
- The 5th harmonic in a 6-pulse VFD rectifier has a fixed phase relationship to the firing angle α; the ratio of 5th-to-7th harmonic phases encodes the rectifier’s operating point.
- Two scenarios can produce identical THD values but entirely different phase structures, making them impossible to distinguish from magnitude alone.

This project explicitly exploits harmonic phase information by combining it with magnitude features in a deep learning classifier designed to improve discrimination between power quality disturbance types and harmonic sources.

### 2.4 Project Motivation and Scope

This project targets an **intelligent, low-cost, phase-aware PQ monitor** for academic demonstration, laboratory use, and small-scale industrial deployment. The total hardware cost is under ₹700 for the sensing frontend — compared to commercial PQ analysers costing ₹50,000 to ₹5,00,000. The project demonstrates:

1. That publication-quality harmonic measurements (amplitude AND phase) are achievable with commodity components when the electronics are designed carefully.
2. That incorporating phase information into a deep learning classifier measurably improves classification accuracy versus magnitude-only methods.
3. That a complete, end-to-end PQ monitoring system — hardware through DSP through AI — can be built, tested, and validated within a B.Tech project timeline.

---

## 3. Problem Statement and Objectives

### 3.1 Formal Problem Statement

Design, implement, and validate a low-cost, hardware-enabled, real-time power quality monitoring system for low-voltage (230 V / 50 Hz) networks. The system shall:

1. Safely measure mains voltage waveforms with galvanic reinforced isolation between the mains circuit and all low-voltage electronics, conforming to the spirit of IEC 61010-1 requirements for measurement Category II equipment.
2. Simultaneously measure load current waveform with zero time offset between voltage and current samples.
3. Extract both magnitude and phase information from harmonic components through at least the 13th order (650 Hz) with no spectral leakage for stationary periodic signals.
4. Apply a trained deep learning classifier that uses phase-aware features to identify power quality disturbances with higher accuracy than magnitude-only baselines.
5. Demonstrate the system on real laboratory loads (AC motor and LED loads) that generate representative harmonic distortion patterns.

### 3.2 Specific Objectives

**Objective 1 — Isolated Hardware Sensing Node:**
Build a compact sensing frontend that measures 230 V mains voltage (with design margin for ±270 V RMS worst-case swell) and load current up to ±50 A AC. The voltage channel uses an AMC1301 reinforced isolated amplifier preceded by a precision 2.2 MΩ / 560 Ω divider. The current channel uses an ACS758LCB-050B Hall-effect sensor powered from the Teensy 3.3 V rail (ratiometric operation). Both channels interface to the Teensy 4.1 dual-ADC hardware at 5 kHz sampling rate with truly simultaneous (zero-delay) acquisition.

**Objective 2 — Spectral-Leakage-Free FFT Harmonic Phasor Extraction:**
Implement a DSP chain that: (a) removes DC offsets and normalises waveform windows; (b) applies an N = 500 FFT with Δf = 10 Hz/bin, placing all 50 Hz harmonics on exact integer bins; (c) extracts harmonic magnitudes and phase angles for orders h = 1 through 13 for both voltage and current; (d) computes THD and Total Demand Distortion (TDD).

**Objective 3 — Phase-Aware Feature Engineering with Correct Circular Statistics:**
Compute phase-aware features including: sin/cos encoding of harmonic phase angles (avoiding ±π discontinuity), circular-statistics-correct inter-harmonic phase differences using `scipy.stats.circmean` and `scipy.stats.circstd`, phase-amplitude coupling ratios, odd/even harmonic phase statistics, and cross-channel (V–I) phase relationships. Combine with time-domain statistical features and DWT time–frequency features into a 282-element feature vector.

**Objective 4 — Phase-Aware Deep Learning Classifier with Ablation Study:**
Train and validate a phase-aware hybrid CNN–LSTM classifier on a synthetic dataset with 5,500 examples per class of seven PQ disturbance types, using load-specific harmonic phase distributions for the Harmonic Distortion class to ensure the phase features carry genuinely discriminative information. Compare classification performance against three model variants (M1–M3) in a structured ablation study.

**Objective 5 — Real-World Validation with Domain Adaptation:**
Collect real measurements from the hardware node on laboratory loads; evaluate zero-shot classifier performance; then fine-tune the last two dense layers on 100–200 real samples per class to demonstrate domain adaptation, with accuracy comparison before and after fine-tuning.

---

## 4. Literature Review

### 4.1 Harmonics and Nonlinear Loads

The theoretical framework for harmonic analysis of power systems was established in the mid-20th century but became practically urgent only with the proliferation of power electronics from the 1980s onward. Arrillaga, Bradley and Bodger’s foundational 1985 work established the harmonic penetration method; Arrillaga and Watson’s later textbook (2003) remains the definitive reference.

The dominant nonlinear loads in contemporary industrial environments:

**Three-phase 6-pulse VFD rectifiers:** Characteristic harmonics at orders 6k±1 (5th, 7th, 11th, 13th) with specific phase angles determined by the firing angle α. The 5th harmonic is typically dominant with amplitude ≈ 20% of fundamental at full load.

**Single-phase SMPS (PC PSUs, LED drivers):** Generate 3rd, 5th, 7th harmonics with triplen harmonics dominant; the 3rd harmonic may reach 80–100% of fundamental amplitude in basic capacitive-input topologies. Phase of the 3rd harmonic is concentrated near ±45° relative to the fundamental.

**Variable-speed drives with uncontrolled rectifier front-end:** Generate a spectrum weighted toward 5th and 7th, with THD typically 30–130% depending on DC link capacitor size and line impedance.

**LED retrofit drivers without PFC:** THD_i in the range 80–150%. The 3rd harmonic dominates with a characteristic phase structure related to the capacitive input filter, distinct from motor load harmonics.

### 4.2 Conventional Power Quality Monitoring

Conventional PQ instruments operate under IEC 61000-4-30, specifying methods for measuring: power frequency, supply voltage magnitude, flicker, dips, swells, interruptions, voltage unbalance, voltage harmonics and interharmonics. Class A instruments (highest accuracy) use synchronised 10-cycle (200 ms at 50 Hz) measurement windows aggregated into 3-second, 10-minute, and 2-hour statistics. Their outputs are magnitude spectra and THD values — phase information is discarded by design.

This is appropriate for the standard’s intended purpose (grid-level compatibility assessment), but inadequate for intelligent disturbance classification, source identification, or predictive maintenance applications.

### 4.3 Machine Learning for Power Quality Disturbance Classification

ML for PQD classification has been active since the early 2000s:

**Classical ML era (2000–2015):** SVMs, k-NN, decision trees, and ANNs trained on hand-crafted features from Fourier transform, S-transform, or wavelet transform achieved 90–97% accuracy on synthetic 6–8 class problems.

**Deep learning era (2015–present):** CNNs applied directly to waveforms or 2D time–frequency representations; LSTMs applied to sequential feature vectors; and hybrid architectures.

Key results from the literature:

- A comparison study (Barros et al., 2020) evaluated CNN, LSTM, and CNN-LSTM on synthetic PQD signals using a 10-class problem. The CNN-LSTM achieved the highest accuracy on synthetic test data; however, on real measured waveforms, accuracy dropped substantially due to the synthetic-to-real domain gap — a limitation acknowledged in the paper that highlights the importance of domain adaptation.
- A multi-modal parallel feature extraction model (Tong et al., 2023) using LSTM for 1D temporal features and a lightweight residual network for 2D image-mode features achieved 99.26% accuracy on 20 disturbance classes under synthetic noiseless conditions.
- A wavelet-attention LSTM model (Springer Electrical Engineering, 2022) demonstrated that multi-resolution wavelet decomposition feeding into an attention-augmented LSTM outperforms FFT-only methods, particularly for transient and flicker classes.
- A CNN-Transformer model (Springer JEET, 2025) achieved 99.48% accuracy under noiseless conditions and 94.32% at 20 dB SNR on a 25-class combined disturbance problem, demonstrating the relevance of attention mechanisms for difficult multi-class scenarios.
- An S-Transform + improved CNN-LSTM architecture (MDPI Processes, 2025) demonstrated that time–frequency preprocessing improves CNN-LSTM convergence and accuracy versus raw waveform input.

**Synthetic-to-Real Domain Gap:** This is the most critical practical limitation in published PQ ML literature. Synthetic signals are generated from clean parametric models; real signals have hardware-specific noise, calibration offsets, quantisation effects, and harmonic phase structures that differ from random-uniform assumptions. This gap is consistently 10–15 percentage points in published comparisons, and addressing it through domain adaptation is a key objective of this project.

**Phase information in literature:** Explicit use of harmonic phase angles as features in deep learning classifiers for PQ applications is rare. Most models operate on magnitude spectrograms or raw waveforms. This represents the research gap this project directly addresses.

### 4.4 Isolated Measurement Techniques for Mains Voltages

Safe measurement of mains-referenced voltages for microcontroller ADC inputs requires galvanic isolation:

- **Current transformers (CTs):** Widely used for current; require burden resistors; may introduce phase error at higher harmonic frequencies due to CT bandwidth limitations.
- **Voltage transformers (VTs/PTs):** Traditional; bulky, frequency-limited to fundamental, expensive, impractical for harmonics above ~500 Hz.
- **Optical isolators:** Low bandwidth (typically < 20 kHz), limited dynamic range.
- **Capacitively-coupled isolated amplifiers (e.g., AMC1301):** Modern approach using sigma-delta modulator signal coupling across a capacitive isolation barrier. Offers 200 kHz bandwidth, 0.03% nonlinearity, ±0.3% gain accuracy. Directly suited for harmonic analysis to the 13th order (650 Hz) and beyond.

The AMC1301 was selected over the HCPL-7800 (optocoupler-based) for its superior gain accuracy, lower nonlinearity, wider bandwidth, and SOIC-8 hand-solderable package.

### 4.5 Low-Cost Embedded PQ Monitoring

Prior low-cost PQ monitoring projects in the literature typically use one of:

- **Arduino/ATmega-based systems:** 10-bit ADC, single channel, 10–15 kHz max sample rate. Inadequate for harmonic analysis above the 3rd order with any accuracy.
- **STM32-based systems:** 12-bit ADC, dual-channel, adequate sample rate, but no simultaneous hardware sampling in most STM32 families without external S&H circuitry.
- **Raspberry Pi + external ADC:** High-speed external ADC via SPI; adds hardware complexity and latency.

The Teensy 4.1 is selected because it uniquely combines: 12-bit dual-ADC with hardware simultaneous mode, 600 MHz Cortex-M7 for ISR responsiveness, and mature ADC library support — in a single inexpensive module.

---

## 5. Standards and Regulatory Context

### 5.1 IEEE 519-2022: Harmonic Control in Electric Power Systems

IEEE 519-2022 supersedes IEEE 519-2014 and defines limits for harmonic voltage and current injection at the Point of Common Coupling (PCC) — the interface between the utility and the customer. Key voltage distortion limits at LV (≤ 1 kV) for general distribution systems:

| Parameter | Limit |
| --- | --- |
| Individual harmonic voltage (any order) | 5.0% of fundamental |
| Total Harmonic Distortion (THD) of voltage | 8.0% of fundamental |

Current distortion limits depend on the ratio I_SC / I_L (short-circuit current to maximum demand load current) — larger facilities with better supply stiffness are allowed more current distortion. For I_SC/I_L = 20–50 (typical small industrial connection):

| Harmonic Order | Current Distortion Limit (% of I_L) |
| --- | --- |
| 3rd–9th | 4.0% |
| 11th–15th | 2.0% |
| 17th–21st | 1.5% |
| 23rd–33rd | 0.6% |
| >33rd | 0.3% |
| TDD (Total Demand Distortion) | 5.0% |

**Note:** IEEE 519-2022 applies at the PCC, not inside the facility. Internal wiring may have higher harmonic content as long as the utility interface meets limits.

### 5.2 IEC 61000-4-30: Power Quality Measurement Methods

IEC 61000-4-30 Edition 3 (2015) specifies the measurement methods, not the limits. Key method requirements relevant to this project:

- **Class A instruments:** Use 10-cycle (200 ms at 50 Hz) DFT windows with synchronisation to the fundamental; report TRMS values for compliance reporting. Harmonic analysis per IEC 61000-4-7 (grouping or subgrouping).
- **Sag/Swell/Interruption:** Detected by comparing the per-cycle URMS(1) against declared voltage thresholds (typically 90% for sag, 110% for swell, 5% for interruption).
- **Flicker:** Measured using the IEC flickermeter algorithm (UIE method), with P_st (short-term perception index) and P_lt (long-term index) as outputs.

This project’s method departs from the IEC standard (which uses 10-cycle windows; this project uses 1-cycle windows) to enable faster real-time response. This is a conscious design tradeoff: faster classification, lower accuracy per IEC Class A definition.

### 5.3 EN 50160: Voltage Characteristics of Public Distribution Systems

EN 50160 defines the characteristics that the supply voltage at a customer’s LV connection shall normally meet — i.e., the utility’s obligations. Key parameters:

- Voltage magnitude: 230 V ± 10% (207–253 V) for 95% of 10-minute means in any week.
- THD: < 8% for 95% of 10-minute means.
- Individual harmonics: tabulated limits from 2nd to 25th order.

### 5.4 IEC 61000-3-2: Harmonic Current Emission Limits for Equipment

IEC 61000-3-2 applies to equipment (not facilities) with input current ≤ 16 A per phase, setting limits on the harmonic current the equipment may inject into the supply. Class D equipment (PCs, TV sets, monitors) has particularly strict per-watt limits that often necessitate active PFC (Power Factor Correction) circuits.

---

## 6. Theoretical Background

### 6.1 Fourier Series and Harmonic Decomposition

Any periodic waveform v(t) with period T = 1/f₀ (f₀ = 50 Hz, T = 20 ms) can be expressed as:

```
v(t) = V₀ + Σ[h=1 to ∞] Vₕ · sin(h·ω₀·t + φₕ)
```

Where:
- V₀ = DC component (ideally zero for AC mains)
- h = harmonic order (1 = fundamental, 2 = 2nd harmonic, etc.)
- Vₕ = peak amplitude of the hth harmonic
- ω₀ = 2π × 50 = 314.16 rad/s
- φₕ = phase angle of the hth harmonic (relative to the reference zero crossing)

The root-mean-square (RMS) value of the composite waveform is:

```
V_RMS = √(V₁²/2 + V₂²/2 + V₃²/2 + ...)  = √(Σ[h=1 to ∞] Vₕ²/2)
```

Total Harmonic Distortion (THD) quantifies the harmonic pollution relative to the fundamental:

```
THD_V = √(Σ[h=2 to n] Vₕ²) / V₁ × 100%
```

Total Demand Distortion (TDD) — used in IEEE 519 for current — uses the maximum demand current (I_L) as the denominator rather than the fundamental current, preventing THD from appearing infinite during light load conditions.

### 6.2 The Discrete Fourier Transform and Spectral Leakage

For a sampled signal with N samples at sampling rate f_s, the Discrete Fourier Transform is:

```
X[k] = Σ[n=0 to N-1] x[n] · e^(-j2πkn/N)
```

The frequency resolution (bin spacing) is:

```
Δf = f_s / N
```

**Spectral leakage** occurs when a frequency component does not fall exactly on an integer bin. For a 50 Hz fundamental with f_s = 5000 Hz:

- N = 512 (power of 2): Δf = 5000/512 = 9.765625 Hz. Fundamental at bin 5.12 (non-integer). Energy leaks into 3–5 adjacent bins. Phase extraction from a non-centred bin is inaccurate by 10–30°.
- **N = 500 (this project):** Δf = 5000/500 = 10.00 Hz exactly. Fundamental at bin 5 (integer). All harmonics at exact integer bins: 3rd at bin 15, 5th at bin 25, 13th at bin 65. Zero spectral leakage for stationary harmonics.

This is the fundamental design choice that makes accurate phase extraction possible.

### 6.3 Phase Angle Physics and Load Signatures

The phase angle φₕ of the hth harmonic (relative to the fundamental zero crossing) is physically determined by the energy conversion mechanism in the load:

**SMPS / Bridge Rectifier (single-phase):**
The capacitor charge pulse occurs when the supply voltage exceeds the DC rail voltage — near the voltage peak. For a symmetrical half-wave topology, the 3rd harmonic is generated by this peak-conduction mechanism. Physical consequence: φ₃ ≈ +π/4 to +π/2 (leading the fundamental).

**6-Pulse VFD Rectifier:**
Firing angle α determines the phase of 5th and 7th harmonic injection. At no firing delay (α = 0, uncontrolled rectifier): φ₅ ≈ -2π/5, φ₇ ≈ +2π/7. These are characteristic signatures of the VFD topology.

**Transformer Saturation:**
Saturation occurs at maximum flux — which is the integral of voltage, so maximum flux occurs near the voltage zero crossing. Saturation injects 3rd harmonic with φ₃ ≈ π/2 (in quadrature with the fundamental).

**Load-Type Distinguishability:** Two loads can have identical THD and identical 3rd harmonic magnitude but different φ₃. An SMPS load (φ₃ ≈ +π/4) and a saturating transformer (φ₃ ≈ +π/2) are indistinguishable by magnitude-only methods but are separated by 45° in phase angle — a clean boundary for a classifier.

### 6.4 Power Factor, Displacement Power Factor, and True Power Factor

For a distorted waveform:

```
True Power Factor = P / (V_RMS × I_RMS) = DPF × DF

Displacement Power Factor (DPF) = cos(φ₁)
Distortion Factor (DF) = I₁_RMS / I_RMS = 1 / √(1 + THD_I²)
```

A highly distorted current (THD_I = 100%) has DF = 0.707, so even at unity DPF, the true power factor is only 0.707. A meter reading apparent power (VA) × DPF would over-bill the customer.

### 6.5 Wavelet Transform — Time-Frequency Localisation

The Discrete Wavelet Transform (DWT) decomposes a signal into approximation (low-frequency) and detail (high-frequency) subbands using pairs of conjugate mirror filters. For a 5-level decomposition with db4 wavelet on a 5000 Hz signal:

| Subband | Frequency Range | Physical Content |
| --- | --- | --- |
| A5 (approximation) | 0 – 78 Hz | Fundamental + sub-harmonics |
| D5 (detail 5) | 78 – 156 Hz | 2nd–3rd harmonic region |
| D4 (detail 4) | 156 – 312 Hz | 3rd–6th harmonic region |
| D3 (detail 3) | 312 – 625 Hz | 6th–12th harmonic region |
| D2 (detail 2) | 625 – 1250 Hz | Upper harmonics + HF switching |
| D1 (detail 1) | 1250 – 2500 Hz | Noise + aliased switching components |

The energy in each detail subband localises when transients occur in time — a capability that pure FFT lacks (FFT averages over the entire window). This makes DWT features particularly valuable for detecting and characterising transient disturbances.

### 6.6 Circular Statistics — Why np.mean() Fails for Phase Angles

Phase angles are periodic (they wrap around at ±π). Standard arithmetic mean and standard deviation are undefined on a circle:

**Example:** The mean of [175°, -175°] computed with `np.mean()` = 0° — pointing in the exact opposite direction from the true circular mean of ±180°.

The correct formulation for circular mean of angles {θ₁, θ₂, …, θₙ}:

```
μ_c = arctan2(mean(sin(θᵢ)), mean(cos(θᵢ)))
```

Circular standard deviation:

```
R = √(mean(sin(θᵢ))² + mean(cos(θᵢ))²)   (mean resultant length, 0 ≤ R ≤ 1)
σ_c = √(-2 · ln(R))
```

In Python: `scipy.stats.circmean(angles, high=np.pi, low=-np.pi)` and `scipy.stats.circstd(angles, high=np.pi, low=-np.pi)`. Using `np.mean()` on phase angles produces systematically wrong statistics that mislead the classifier — particularly for phases near ±π (e.g., 5th harmonic from negative-sequence sources).

---

## 7. System Architecture Overview

### 7.1 Block Diagram

![image.png](image.png)

### 7.2 Data Flow Summary

1. Mains voltage sensed by AMC1301 frontend (with isolated B0505S supply). Output level-shifted and filtered to 0–3.3 V.
2. Load current wire passes through ACS758 Hall sensor body. Output (ratiometric to 3.3 V supply) filtered to 0–3.3 V.
3. IntervalTimer ISR fires at exactly 5 kHz, triggering Teensy dual-ADC simultaneous read. Zero-crossing detection aligns window capture to the mains rising zero crossing.
4. Complete 500-sample dual-channel frame (1000 int16 values) sent to PC over USB CDC serial as a 2012-byte binary packet with CRC32 integrity check.
5. PC Python script receives, validates, and runs the frame through preprocessing and feature extraction.
6. Trained CNN–LSTM model outputs 7-class probability distribution; argmax is the predicted disturbance label.
7. Results logged with timestamp, waveform data, and extracted features for post-session analysis.

---

## 8. Hardware Design

### 8.1 Voltage Sensing — AMC1301 Isolated Amplifier

### 8.1.1 IC Overview and Key Specifications

The Texas Instruments AMC1301 is a precision, reinforced isolated amplifier in a SOIC-8 DWV package, using capacitively-coupled sigma-delta modulation to transmit the analog signal across the isolation barrier.

| Parameter | Value |
| --- | --- |
| Input voltage range | ±250 mV differential |
| Fixed gain | 8.2 V/V |
| Isolation voltage | 7070 V_PEAK (DIN EN IEC 60747-17, UL1577) |
| Working voltage (continuous) | 1000 V_RMS |
| Bandwidth (−3 dB) | ~200 kHz |
| Gain error (max at 25°C) | ±0.3% |
| Gain drift | ±50 ppm/°C |
| Offset error (max at 25°C) | ±0.2 mV referred to input |
| Offset drift | ±3 µV/°C |
| Nonlinearity | 0.03% (max) |
| Supply voltage VDD1, VDD2 | 3.0 V to 5.5 V |
| Output type | Differential (OUTP, OUTN) |
| Propagation delay | ~3 µs |
| Package | SOIC-8 DWV (wide-body, increased creepage) |

### 8.1.2 Critical Pin Architecture — Two Isolated Power Domains

**High-side (mains-referenced):**
- VDD1: Power supply, 3.0–5.5 V relative to GND1, sourced from B0505S isolated output
- GND1: Ground reference = mains neutral rail
- INP: Non-inverting input (connected to divider output)
- INN: Inverting input (connected directly to GND1)

**Low-side (MCU-referenced):**
- VDD2: 5 V from Teensy VUSB, relative to GND2
- GND2: Teensy GND (MCU_GND)
- OUTP: Positive differential output
- OUTN: Negative differential output / REFIN input

**CRITICAL SAFETY RULE:** GND1 and GND2 are NEVER connected to each other. The entire isolation safety of the circuit depends on this separation. VDD1 must be powered from a source that is itself isolated from VDD2 — this is the role of the B0505S module.

### 8.1.3 Voltage Divider Design

The resistor divider scales the mains voltage to within the AMC1301’s ±250 mV input range with adequate headroom:

```
Selected: R1 = 2 × 2.2 MΩ in series (parallel-mounted on PCB,
          total = 2.2 MΩ effective resistance with independent failure modes)
          R2 = 560 Ω

Divider ratio: R2 / (R1 + R2) = 560 / (2,200,000 + 560) = 2.545 × 10⁻⁴

At 270 V RMS → 382 V peak (worst-case swell):
  V_INP = 382 × 2.545 × 10⁻⁴ = 97.2 mV  ←  39% of ±250 mV limit ✓

At 230 V RMS → 325 V peak (nominal):
  V_INP = 325 × 2.545 × 10⁻⁴ = 82.7 mV ✓

Power dissipated in R1 (2.2 MΩ total):
  P = V²/R = (230)² / 2,200,000 = 24 mW → 1/4 W rated resistors are fine ✓
```

Using two series 1.1 MΩ resistors provides redundancy: if one fails short, the divider ratio increases but no dangerous voltage appears at the IC input.

### 8.1.4 Output Stage — Single-Ended Conversion

OUTN is tied to REFIN pin, setting the output common-mode voltage to VDD2/2. OUTP swings around VDD2/2 proportionally to the input:

```
VOUTP = VDD2/2 ± (Gain × VIN_diff) / 2

With VDD2 = 5 V, Gain = 8.2:

At +97 mV input: VOUTP = 2.5 + (8.2 × 0.097)/2 = 2.5 + 0.398 = 2.898 V
At -97 mV input: VOUTP = 2.5 - 0.398 = 2.102 V
At 0 V input:    VOUTP = 2.5 V (midpoint)
```

VOUTP max = 2.898 V — but Teensy ADC max = 3.3 V, so a 1kΩ/2kΩ resistive divider is required:

```
Scaled = VOUTP × (2kΩ / (1kΩ + 2kΩ)) = VOUTP × 0.667

At 270V peak:    2.898 × 0.667 = 1.933 V  [ADC count ≈ 2398]  ✓
At 230V nominal: 2.871 × 0.667 = 1.915 V  [ADC count ≈ 2376]  ✓
At 0V (midpoint):2.500 × 0.667 = 1.668 V  [ADC count ≈ 2071]
At -270V peak:   2.102 × 0.667 = 1.402 V  [ADC count ≈ 1740]  ✓

Full scale swing used: (2398 - 1740) = 658 counts out of 4096
Effective ADC bits: log₂(658) = 9.4 bits
```

**Note:** The passive single-ended conversion approach uses only OUTP (half the differential swing), limiting effective resolution to 9.4 bits. A proper difference amplifier using a TLV901x op-amp (as in TI App Note SBAA229) would deliver 12 effective bits. For THD ratio analysis this is adequate; for detecting small harmonics (e.g., 1% 5th harmonic content), the additional bits are valuable. See Limitation in Section 20.

### 8.1.5 Gain Error Analysis with 1% Resistors

```
Worst-case R1: 2.2 MΩ × (1 ± 0.01) = 2.178–2.222 MΩ
Worst-case R2: 560 Ω × (1 ± 0.01) = 554.4–565.6 Ω

Worst case (R1 max, R2 min): 554.4 / 2,222,554 = 2.495 × 10⁻⁴ → -2.0% error
Worst case (R1 min, R2 max): 565.6 / 2,178,566 = 2.596 × 10⁻⁴ → +2.0% error
```

A ±2.0% gain error on absolute voltage reading is acceptable and compensated at calibration. The gain error cancels in THD and phase angle calculations because these are ratios (harmonic magnitude / fundamental magnitude — the gain factor cancels).

### 8.2 Isolated Power Supply for AMC1301 VDD1 — B0505S-1W

**Why isolated supply is non-negotiable:** Without isolation, GND1 (mains neutral) connects through the AMC1301’s internal circuit to GND2 (Teensy GND), which connects to the PC chassis through the USB cable. This creates a direct mains-to-laptop-chassis connection, bypassing all isolation. Lethal.

**The B0505S-1W SIP-7 Module:**

| Parameter | Value |
| --- | --- |
| Input voltage | 5 V ± 10% (from Teensy VUSB) |
| Output voltage | 5 V isolated, 200 mA max |
| Isolation voltage | 1000 V_DC |
| Efficiency | ~75% |
| Dimensions | ~19 × 10 × 6 mm |
| Estimated cost | ₹80 (Robu.in) |

**Important isolation limitation:** The B0505S provides 1000 V DC (≈707 V RMS) isolation. The AMC1301 itself provides 7070 V_PEAK (5000 V RMS) reinforced isolation. The overall system isolation is limited by the weakest link: the B0505S at ~707 V RMS. For 230 V mains, IEC 61010-1 Measurement Category II requires a test voltage of 2500 V; the B0505S does not meet this for a formally certified instrument. For a supervised student lab demonstration with the precautions in Section 18, the risk level is accepted. For a deployable commercial product, a TRACO TMA 0505D (1000 V RMS) or Murata MEE1S0505SC should be used instead.

Bypass capacitors:
- C3: 100 nF NP0 ceramic, placed within 5 mm of AMC1301 VDD1 pin
- C4: 10 µF electrolytic, bulk reservoir, within 20 mm

### 8.3 Current Sensing — ACS758LCB-050B

### 8.3.1 IC Overview

| Parameter | Value |
| --- | --- |
| Current range | ±50 A bidirectional |
| Sensitivity at 5 V supply | 40 mV/A |
| Sensitivity (ratiometric) | 8 mV/A per volt of VCC |
| Quiescent output voltage | VCC / 2 |
| Bandwidth (−3 dB) | 120 kHz |
| Output rise time | 3 µs for step input |
| Conductor resistance | ~100 µΩ (negligible power loss) |
| Supply voltage | 3.0–5.5 V |
| Isolation voltage (load-to-signal) | 4800 V_RMS |

**Ratiometric operation:** All ACS758 output parameters scale linearly with VCC:

| VCC | Quiescent VOUT | Sensitivity |
| --- | --- | --- |
| 5.0 V | 2.50 V | 40 mV/A |
| 3.3 V | 1.65 V | 26.4 mV/A |

### 8.3.2 Operation at 3.3 V (Teensy 3V3 Rail)

By powering the ACS758 from the Teensy 3.3 V rail, VOUT is guaranteed within 0–3.3 V:

```
At +50 A: VOUT = 1.65 + 0.0264 × 50 = 2.97 V  ✓
At -50 A: VOUT = 1.65 - 1.32 = 0.33 V           ✓

For demo load (5–15 A peak):
At ±10 A: VOUT = 1.65 ± 0.264 V = 1.386 to 1.914 V
Peak-to-peak: 0.528 V → 656 ADC counts
Current resolution: 3.3 / (4096 × 0.0264) = 30.5 mA/count
```

This resolution adequately resolves harmonic content at 5–10% of fundamental (~300–600 mA) in a 5 A RMS fundamental current.

**Breakout board caution:** Common ACS758 breakout modules include a buffer op-amp (often LM2272) which may not be rail-to-rail and will clip at 3.3 V VCC. Verify on oscilloscope before use. If clipping is observed, bypass the op-amp and use direct VIOUT pin output.

### 8.3.3 Decoupling

- C7: 100 nF NP0 ceramic, within 5 mm of ACS758 VCC pin
- C8: 10 µF electrolytic, within 20 mm

### 8.4 Anti-Aliasing Filters

**Why mandatory:** Teensy ADC samples at f_s = 5 kHz → Nyquist at 2500 Hz. The AMC1301 passes signals to 200 kHz; ACS758 to 120 kHz. Without filtering, any signal energy above 2500 Hz aliases (folds) into the measurement band. Motor drive PWM switching harmonics (typically 2–20 kHz) would alias onto bins where 3rd–7th harmonics sit, producing false harmonic detections.

**Voltage channel filter:**

The resistive output divider (R3 = 1 kΩ, R4 = 2 kΩ, Thevenin resistance = 667 Ω) followed by R5 = 390 Ω series and C1 = 100 nF shunt:

```
Total filter R = 667 + 390 = 1057 Ω
fc = 1 / (2π × 1057 × 100 × 10⁻⁹) = 1505 Hz

Attenuation at 650 Hz (13th harmonic): 1/√(1+(650/1505)²) = 0.918 → -0.73 dB ✓
Attenuation at 2500 Hz (Nyquist):      1/√(1+(2500/1505)²) = 0.517 → -5.7 dB
Attenuation at 5000 Hz (image freq):   1/√(1+(5000/1505)²) = 0.289 → -10.8 dB
```

**Known limitation:** -5.7 dB at Nyquist is modest. A 2nd-order Sallen-Key filter (adding one op-amp and two additional R-C pairs) would provide -20 dB at Nyquist without affecting measurements below 650 Hz. This is the recommended upgrade path for improved real-world robustness. See Section 20.

**Current channel filter:** R6 = 390 Ω + C2 = 100 nF, effective R ≈ 500–1000 Ω (depending on ACS758 output impedance), giving fc ≈ 1600–3200 Hz.

### 8.5 Teensy 4.1 Microcontroller

**Key relevant specifications:**

| Parameter | Value |
| --- | --- |
| Core | ARM Cortex-M7 @ 600 MHz |
| ADC modules | 2 independent 12-bit ADC (ADC0, ADC1) |
| ADC simultaneous mode | Hardware-triggered simultaneous conversion (pedvide ADC library) |
| ADC resolution | 12-bit, 4096 levels, 0.806 mV/count at 3.3 V ref |
| IntervalTimer resolution | 0.1 µs (100 ns period resolution) |
| USB interface | USB 2.0 HS, 480 Mbit/s |
| Flash | 8 MB |
| RAM | 1 MB |

**ADC pin assignment for simultaneous mode:**
- A0 (pin 14) → ADC0 hardware — voltage channel
- A10 (pin 24) → ADC1 hardware — current channel

Other pin combinations may work, but A0/A10 is the confirmed pairing for simultaneous hardware mode with the pedvide ADC library.

---

## 9. Firmware — Teensy 4.1 Embedded Software

### 9.1 ADC Configuration

```cpp
#include<ADC.h>

ADC *adc = new ADC();

void setupADC() {
    // --- ADC0 (voltage channel, A0/pin 14) ---
    adc->adc0->setAveraging(4);
    adc->adc0->setResolution(12);
    adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
    adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
    adc->adc0->setReference(ADC_REFERENCE::REF_3V3);

    // --- ADC1 (current channel, A10/pin 24) ---
    adc->adc1->setAveraging(4);
    adc->adc1->setResolution(12);
    adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
    adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
    adc->adc1->setReference(ADC_REFERENCE::REF_3V3);
}
```

The 4-sample hardware averaging adds ~800 ns conversion time overhead but reduces quantisation noise by √4 = 2×, effectively adding 1 bit of resolution. At 5 kHz sampling (200 µs interval), there is ample time for averaged conversion on both channels simultaneously.

### 9.2 Zero-Crossing Detection and Window Synchronisation

Windows start on the rising zero crossing of the voltage waveform. This synchronisation is critical: with ZC-synchronised windows, the fundamental phase φ₁ is consistently near 0 radians, making all harmonic phase angles physically meaningful and consistent between windows. Without ZC sync, φ₁ varies randomly window-to-window.

```cpp
const int ADC_MIDPOINT   = 2071;   // calculated: 1.668V × 4096/3.3 = 2071 counts at 0V mains
const int ZC_HYSTERESIS  = 20;     // ±20 counts hysteresis to prevent noise-triggered false ZC
const int N              = 500;    // samples per window = exactly 1 cycle at 5kHz/50Hz

volatile int16_t v_buf[N];
volatile int16_t i_buf[N];
volatile bool    windowReady = false;
volatile int     sampleCount = 0;
volatile bool    collecting  = false;
volatile int16_t prevV       = 0;

IntervalTimer sampleTimer;

void FASTRUN sampleISR() {
    ADC::Sync_result result = adc->readSynchronizedSingle(A0, A10);
    int16_t v     = (int16_t)result.result_adc0;
    int16_t i_raw = (int16_t)result.result_adc1;

    // Rising ZC: previous sample clearly below midpoint, current clearly above
    bool risingZC = (prevV < (ADC_MIDPOINT - ZC_HYSTERESIS)) &&
                    (v     >= (ADC_MIDPOINT + ZC_HYSTERESIS));

    if (!collecting && !windowReady && risingZC) {
        collecting  = true;
        sampleCount = 0;
    }

    if (collecting) {
        v_buf[sampleCount] = v;
        i_buf[sampleCount] = i_raw;
        sampleCount++;
        if (sampleCount >= N) {
            collecting  = false;
            windowReady = true;
        }
    }

    prevV = v;
}

void setup() {
    Serial.begin(0);  // USB CDC; baud rate ignored
    setupADC();
    adc->startSynchronizedSingleRead(A0, A10);
    sampleTimer.begin(sampleISR, 200);  // 200 µs = 5 kHz
}
```

**Window capture limitation for transient events:** ZC-triggered windows may miss short sags/transients that begin mid-cycle and end before the next rising ZC. Future improvement: implement event-triggered mode that captures windows on RMS threshold crossing, operating in parallel with ZC-sync mode.

### 9.3 USB Serial Frame Protocol

```
Frame structure (CORRECTED — total 2012 bytes)
─────────────────────────────────────────────
Byte 0–3:      Magic number 0xDEADBEEF (4 bytes, big-endian)
Byte 4–5:      Frame sequence number uint16_t (wraps at 65535)
Byte 6–7:      N = 500 (uint16_t, sanity check)
Byte 8–1007:   v_buf[0..499] as int16_t little-endian (1000 bytes)
Byte 1008–2007: i_buf[0..499] as int16_t little-endian (1000 bytes)
Byte 2008–2011: CRC32 of bytes 4–2007 inclusive (4 bytes)
─────────────────────────────────────────────
Total: 4 + 2 + 2 + 1000 + 1000 + 4 = 2012 bytes
```

Data rate: 2012 × (5000/500) = 20,120 bytes/s ≈ 161 kbit/s. USB CDC bandwidth: 480 Mbit/s. Utilisation: 0.033%.

```cpp
// CRC32 uses the FastCRC library (FastCRC.h) with CRC-32/ISO-HDLC polynomial (0xEDB88320)
// This MUST match the Python receiver's binascii.crc32() which uses the same polynomial.
#include<FastCRC.h>
FastCRC32 CRC32;

void sendFrame() {
    // --- 1. Magic ---
    uint32_t magic = 0xDEADBEEF;
    Serial.write((uint8_t*)&magic, 4);

    // --- 2. Sequence number (transmitted BEFORE increment) ---
    static uint16_t seqNum = 0;
    uint16_t txSeqNum = seqNum;  // capture current value
    Serial.write((uint8_t*)&txSeqNum, 2);

    // --- 3. N ---
    uint16_t n = N;
    Serial.write((uint8_t*)&n, 2);

    // --- 4. Voltage buffer ---
    Serial.write((uint8_t*)v_buf, N * 2);

    // --- 5. Current buffer ---
    Serial.write((uint8_t*)i_buf, N * 2);

    // --- 6. CRC32 over bytes 4..2007 (seqNum + N + v_buf + i_buf) ---
    // Build a contiguous buffer for CRC computation
    uint8_t crcBuf[4 + N*4];
    memcpy(crcBuf,           &txSeqNum, 2);  // seqNum (pre-increment)
    memcpy(crcBuf + 2,       &n,        2);  // N
    memcpy(crcBuf + 4,       (void*)v_buf, N * 2);
    memcpy(crcBuf + 4 + N*2, (void*)i_buf, N * 2);
    uint32_t crc = CRC32.crc32(crcBuf, sizeof(crcBuf));
    Serial.write((uint8_t*)&crc, 4);

    // --- 7. Increment sequence number AFTER CRC computation ---
    seqNum++;

    Serial.send_now();
}

void loop() {
    if (windowReady) {
        sendFrame();
        windowReady = false;
    }
}
```

**Critical bug fix note:** The CRC is computed over the pre-increment sequence number (txSeqNum) and the CRC is transmitted after the data. The sequence number is incremented only after all transmission is complete. This ensures the transmitted seqNum and the CRC-covered seqNum are identical.

### 9.4 Python Frame Receiver

```python
import serial
import struct
import binascii
import numpy as np

MAGIC      = 0xDEADBEEF
N          = 500
FRAME_SIZE = 2012  # 4 + 2 + 2 + 1000 + 1000 + 4

def open_port(port='/dev/ttyACM0', baud=115200):
    return serial.Serial(port, baud, timeout=2.0)

def read_frame(ser):
    """Block until a valid frame is received. Returns (v_buf, i_buf) as np.int16 arrays."""
    # Scan for magic bytes
    magic_bytes = MAGIC.to_bytes(4, 'big')
    buf = b''
    while True:
        byte = ser.read(1)
        if not byte:
            continue
        buf = (buf + byte)[-4:]
        if buf == magic_bytes:
            break

    # Read remaining frame (2012 - 4 = 2008 bytes)
    rest = ser.read(FRAME_SIZE - 4)
    if len(rest) < FRAME_SIZE - 4:
        return None, None  # timeout

    # Parse fields
    seq_num = struct.unpack_from('<H', rest, 0)[0]   # uint16 LE
    n_check = struct.unpack_from('<H', rest, 2)[0]   # should be 500
    if n_check != N:
        return None, None

    v_raw = np.frombuffer(rest[4:1004],    dtype='<i2')  # 500 int16
    i_raw = np.frombuffer(rest[1004:2004], dtype='<i2')  # 500 int16

    # Verify CRC32 over bytes 0..2003 of `rest` (= frame bytes 4..2007)
    rx_crc  = struct.unpack_from('<I', rest, 2004)[0]
    calc_crc = binascii.crc32(rest[:2004]) & 0xFFFFFFFF
    if rx_crc != calc_crc:
        print(f"CRC mismatch! rx={rx_crc:08X} calc={calc_crc:08X}")
        return None, None

    return v_raw, i_raw
```

---

## 10. Signal Processing and Feature Extraction Pipeline

All DSP runs on the host PC using Python 3.10+ with NumPy, SciPy, and PyWavelets.

### 10.1 Calibration Constants

```python
# Determined empirically from known-load measurements at first setup
V_ADC_MIDPOINT = 2071        # ADC count at 0 V mains (= 1.668V / 0.000806 V/count)
V_COUNTS_PER_V = 0.001720    # Volts mains per ADC count  (adjust at calibration)
#   Derivation: 1 count = 3.3V/4096 = 0.000806V at ADC
#               0.000806V / 0.667 (output divider) / 8.2 (AMC1301 gain) / divider_ratio
#               = 0.000806 / 0.667 / 8.2 / 2.545e-4 ≈ 0.00172 V_mains/count

I_ADC_MIDPOINT = 2048        # ADC count at 0 A (= 1.65V / 0.000806)
I_COUNTS_PER_A = 0.030518    # Amps per ADC count
#   Derivation: 3.3V / (4096 counts × 0.0264 V/A) = 0.03052 A/count
```

### 10.2 Preprocessing

```python
def preprocess(v_buf, i_buf):
    """Convert raw ADC counts to physical units, remove DC, normalise."""
    # Step 1: Convert to physical units
    v_phys = (v_buf.astype(np.float64) - V_ADC_MIDPOINT) * V_COUNTS_PER_V  # Volts mains
    i_phys = (i_buf.astype(np.float64) - I_ADC_MIDPOINT) * I_COUNTS_PER_A  # Amperes

    # Step 2: DC offset removal (removes AMC1301 offset error and ACS758 quiescent drift)
    v_phys -= np.mean(v_phys)
    i_phys -= np.mean(i_phys)

    # Step 3: Normalise for neural network input ([-1, 1] range)
    # Use max-abs normalisation with epsilon for interruption events (near-zero signal)
    v_norm = v_phys / (np.max(np.abs(v_phys)) + 1e-8)
    i_norm = i_phys / (np.max(np.abs(i_phys)) + 1e-8)

    return v_phys, i_phys, v_norm, i_norm
```

### 10.3 Time-Domain Features

```python
import scipy.stats

def time_domain_features(x):
    """12 statistical features from a 500-sample waveform window."""
    rms          = np.sqrt(np.mean(x**2))
    return {
        'rms'           : rms,
        'mean'          : np.mean(x),               # ~0 after DC removal
        'std'           : np.std(x),
        'maximum'       : np.max(x),
        'minimum'       : np.min(x),
        'peak_to_peak'  : np.max(x) - np.min(x),
        'crest_factor'  : np.max(np.abs(x)) / (rms + 1e-8),
        'skewness'      : float(scipy.stats.skew(x)),
        'kurtosis'      : float(scipy.stats.kurtosis(x)),   # excess kurtosis
        'zero_crossings': int(np.sum(np.diff(np.sign(x)) != 0)),
        'energy'        : float(np.sum(x**2)),
        'total_variation': float(np.sum(np.abs(np.diff(x))))
    }
# 12 features per channel × 2 channels = 24 total time-domain features
```

Physical meaning of each feature:
- **RMS:** Core sag/swell detector (compare against nominal 230V).
- **Crest factor:** Pure sine = √2 = 1.414. SMPS current load may reach 3–5. Resistive load ~1.4.
- **Skewness:** Asymmetric clipping or asymmetric sag/swell creates skewness.
- **Kurtosis:** Transient spikes significantly increase kurtosis vs. a sinusoidal baseline.
- **Zero crossings:** 50 Hz sine → 10 crossings/window. Harmonics increase this count predictably.
- **Total variation:** Transients and high-order harmonics dramatically increase waveform “roughness.”

### 10.4 FFT-Based Harmonic Feature Extraction

```python
def harmonic_features(x, fs=5000, N=500, max_order=13):
    """
    Extract harmonic magnitudes, phases, and THD.
    N=500, fs=5000 → Δf = 10 Hz, fundamental at bin 5.
    All harmonics (50, 100, 150 Hz...) fall on exact integer bins.
    Zero spectral leakage for stationary periodic signals.
    """
    X     = np.fft.rfft(x)                   # shape (251,), complex
    freqs = np.fft.rfftfreq(N, 1.0/fs)       # [0, 10, 20, ..., 2500] Hz

    mag = {}  # peak amplitude of hth harmonic
    phi = {}  # phase angle of hth harmonic (radians, relative to ZC-sync reference)

    for h in range(1, max_order + 1):
        k = h * 5                            # bin index: h × (50 Hz / 10 Hz/bin)
        mag[h] = (2.0 / N) * np.abs(X[k])   # two-sided to single-sided correction
        phi[h] = float(np.angle(X[k]))       # radians, in (-π, +π]

    # THD: RMS of harmonics 2..max_order normalised to RMS of fundamental
    V1_rms         = mag[1] / np.sqrt(2)
    harmonics_rms  = np.sqrt(sum((mag[h]/np.sqrt(2))**2 for h in range(2, max_order+1)))
    thd            = (harmonics_rms / (V1_rms + 1e-8)) * 100.0   # percent

    return mag, phi, thd
```

Feature vector contributions from this stage:
- 13 harmonic magnitudes × 2 channels (V, I) = **26 features**
- 2 THD values (V and I) = **2 features**
- Phase angles are NOT added as raw values here (see Section 10.5 for correct encoding)
- Subtotal from FFT: **28 features**

### 10.5 Phase-Aware Feature Engineering — With Correct Circular Statistics

This is the core innovation. Phase angles are processed into physically meaningful, discontinuity-free, statistically correct features.

```python
from scipy.stats import circmean, circstd

def phase_aware_features(mag_v, phi_v, mag_i, phi_i, max_order=13):
    """
    Extract 254 phase-aware features using correct circular statistics.
    Phase angles are NEVER included as raw values (avoids ±π discontinuity).
    All phase statistics use scipy circular methods.
    """
    feats = []

    # ── 1. Sin/cos encoding of absolute phases (52 features) ──────────────────
    # Replaces raw angle storage. Eliminates discontinuity at ±π.
    for h in range(1, max_order + 1):
        feats += [np.sin(phi_v[h]), np.cos(phi_v[h])]   # 2 per harmonic × 13 = 26 (V)
        feats += [np.sin(phi_i[h]), np.cos(phi_i[h])]   # 2 per harmonic × 13 = 26 (I)
    # Subtotal: 52 features

    # ── 2. Inter-harmonic phase differences relative to fundamental (48 features) ──
    for h in range(2, max_order + 1):
        delta_v = phi_v[h] - phi_v[1]
        delta_i = phi_i[h] - phi_i[1]
        # Wrap to (-π, π]
        delta_v = (delta_v + np.pi) % (2 * np.pi) - np.pi
        delta_i = (delta_i + np.pi) % (2 * np.pi) - np.pi
        # Sin/cos encode the wrapped difference
        feats += [np.sin(delta_v), np.cos(delta_v)]
        feats += [np.sin(delta_i), np.cos(delta_i)]
    # Subtotal: 4 features × 12 orders = 48 features

    # ── 3. Phase-amplitude coupling: polar form of weighted harmonic phasors (24 feats) ──
    for h in [3, 5, 7, 9, 11, 13]:
        r_v = mag_v[h] / (mag_v[1] + 1e-8)   # magnitude ratio
        r_i = mag_i[h] / (mag_i[1] + 1e-8)
        # Convert polar (r, φ) → Cartesian (r·cos, r·sin)
        feats += [r_v * np.sin(phi_v[h]), r_v * np.cos(phi_v[h])]
        feats += [r_i * np.sin(phi_i[h]), r_i * np.cos(phi_i[h])]
    # Subtotal: 4 × 6 odd harmonics = 24 features

    # ── 4. V-I cross-channel phase differences at each harmonic (26 features) ──
    for h in range(1, max_order + 1):
        cross_phi = phi_v[h] - phi_i[h]
        cross_phi = (cross_phi + np.pi) % (2 * np.pi) - np.pi
        feats += [np.sin(cross_phi), np.cos(cross_phi)]
    # Subtotal: 2 × 13 = 26 features

    # ── 5. Odd/even harmonic phase group statistics using CIRCULAR STATISTICS (8 feats) ──
    odd_phi_v  = np.array([phi_v[h] for h in [3,  5,  7,  9,  11, 13]])
    even_phi_v = np.array([phi_v[h] for h in [2,  4,  6,  8,  10, 12]])
    odd_phi_i  = np.array([phi_i[h] for h in [3,  5,  7,  9,  11, 13]])
    even_phi_i = np.array([phi_i[h] for h in [2,  4,  6,  8,  10, 12]])

    # Use scipy circular statistics — NOT np.mean/np.std which are WRONG for angles
    for angles in [odd_phi_v, even_phi_v, odd_phi_i, even_phi_i]:
        cm = circmean(angles, high=np.pi, low=-np.pi)
        cs = circstd( angles, high=np.pi, low=-np.pi)
        feats += [cm, cs]
    # Subtotal: 2 stats × 4 groups = 8 features

    return np.array(feats)
    # Total: 52 + 48 + 24 + 26 + 8 = 158 features
```

### 10.6 Wavelet-Domain Features

```python
import pywt

def wavelet_features(x, wavelet='db4', level=5):
    """
    5-level DWT decomposition. 6 statistical features per subband × 6 subbands = 36 features.
    db4 chosen for its smooth frequency response and near-symmetric impulse response.
    """
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # coeffs[0]=A5(approx 0-78Hz), [1]=D5(78-156Hz), ..., [5]=D1(1250-2500Hz)
    feats = []
    for c in coeffs:
        feats += [
            float(np.mean(c)),
            float(np.std(c)),
            float(np.sum(c**2)),              # subband energy
            float(np.max(np.abs(c))),          # peak absolute coefficient
            float(scipy.stats.skew(c)),
            float(scipy.stats.kurtosis(c))     # excess kurtosis
        ]  # 6 features per subband
    return np.array(feats)   # 36 per channel × 2 channels = 72 total
```

### 10.7 Complete Feature Vector (Corrected — 282 Elements)

| Feature Group | Description | Count |
| --- | --- | --- |
| Time-domain statistics | RMS, crest factor, kurtosis, etc. | 24 |
| FFT harmonic magnitudes h=1..13 | V + I channels | 26 |
| THD | V and I | 2 |
| Phase sin/cos encoding | V + I, h=1..13, 2 per harmonic per channel | 52 |
| Inter-harmonic phase differences | V + I, h=2..13, sin/cos encoded | 48 |
| Phase-amplitude coupling (polar form) | V + I, odd harmonics h=3,5,7,9,11,13 | 24 |
| V-I cross-channel phase | h=1..13, sin/cos encoded | 26 |
| Circular phase statistics | odd/even groups, V+I, circmean+circstd | 8 |
| DWT subband features (6 subbands × 6 stats) | V + I channels | 72 |
| **TOTAL** |  | **282** |

**Note:** Raw phase angles (26 values) are intentionally EXCLUDED from the feature vector. They are redundant with the sin/cos encoding in Group 4 and introduce ±π discontinuity that the sin/cos encoding was designed to avoid. The feature vector has 282 elements, not 308.

All 282 features are z-score standardised using statistics computed over the training set:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
# Save scaler for inference: joblib.dump(scaler, 'scaler.pkl')
```

---

## 11. Deep Learning Model

### 11.1 Baseline Model — Magnitude-Only MLP (M1)

The baseline uses only magnitude features (no phase, no DWT) to establish a lower bound:

- **Input:** 24 time-domain + 26 FFT magnitudes + 2 THD = 52 features
- **Architecture:** Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(7, Softmax)
- **Parameters:** ~16,000
- **Purpose:** Establishes what accuracy is achievable with conventional magnitude-centric approaches

### 11.2 Phase-Aware CNN–LSTM Hybrid Model (M4 — Full Proposed)

Three parallel branches fused before the output layer:

### Branch 1 — Raw Waveform CNN-LSTM

```
Input: (500, 2) — 500 time steps, channels: [v_norm, i_norm]
    │
Conv1D(32 filters, kernel=7, padding='same') → BatchNorm → ReLU
    │
MaxPool1D(4)                                  → shape: (125, 32)
    │
Conv1D(64 filters, kernel=5, padding='same') → BatchNorm → ReLU
    │
MaxPool1D(4)                                  → shape: (31, 64)
    │
Conv1D(128 filters, kernel=3, padding='same') → BatchNorm → ReLU
    │
MaxPool1D(2)                                  → shape: (15, 128)
    │
LSTM(64 units, return_sequences=False)        → shape: (64,)
    │
Dropout(0.3)
    │
Branch1 output: (64,) — temporal + convolutional features
```

### Branch 2 — Magnitude Feature MLP

```
Input: (28,) — time-domain stats + FFT magnitudes + THD (from X_scaled)
    │
Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU)
    │
Branch2 output: (32,)
```

### Branch 3 — Phase + DWT Feature MLP

```
Input: (254,) — 158 phase features + 72 DWT features + 24 time-domain stats (reused)
Note: 158 + 72 + 24 = 254 (time-domain stats shared with Branch2 for phase context)
    │
Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU)
    │
Branch3 output: (64,)
```

### Fusion and Output

```
Concatenate([Branch1(64), Branch2(32), Branch3(64)]) → (160,)
    │
Dense(128, ReLU) → Dropout(0.4)
    │
Dense(64, ReLU)
    │
Dense(7, Softmax)
    │
Output: probability distribution over 7 classes
```

Total trainable parameters: approximately 280,000 — trainable on CPU in under 1 hour for 38,500 samples.

### Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_phase_aware_model(n_samples=500, n_phase_dwt_feats=254, n_mag_feats=28):
    # ── Branch 1: Raw waveform CNN-LSTM ──────────────────────────────────────
    waveform_in = Input(shape=(n_samples, 2), name='waveform_input')
    x = layers.Conv1D(32, 7, padding='same', activation=None)(waveform_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(64, 5, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Conv1D(128, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    branch1 = layers.Dropout(0.3)(x)

    # ── Branch 2: Magnitude MLP ───────────────────────────────────────────────
    mag_in = Input(shape=(n_mag_feats,), name='magnitude_input')
    y = layers.Dense(64, activation='relu')(mag_in)
    y = layers.Dropout(0.2)(y)
    branch2 = layers.Dense(32, activation='relu')(y)

    # ── Branch 3: Phase + DWT MLP ─────────────────────────────────────────────
    phase_in = Input(shape=(n_phase_dwt_feats,), name='phase_dwt_input')
    z = layers.Dense(128, activation='relu')(phase_in)
    z = layers.Dropout(0.3)(z)
    branch3 = layers.Dense(64, activation='relu')(z)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused = layers.Concatenate()([branch1, branch2, branch3])
    out = layers.Dense(128, activation='relu')(fused)
    out = layers.Dropout(0.4)(out)
    out = layers.Dense(64, activation='relu')(out)
    out = layers.Dense(7, activation='softmax', name='class_output')(out)

    return Model(inputs=[waveform_in, mag_in, phase_in], outputs=out)
```

### 11.3 Ablation Study Design

| Variant | Branch 1 CNN-LSTM | Branch 2 Magnitude | Branch 3 Phase+DWT | Hypothesis |
| --- | --- | --- | --- | --- |
| M1 (Baseline MLP) | ✗ | ✓ | ✗ | Magnitude-only lower bound |
| M2 (Raw waveform) | ✓ | ✗ | ✗ | Raw waveform signal quality |
| M3 (CNN-LSTM + Mag) | ✓ | ✓ | ✗ | Phase-free upper bound |
| M4 (Full — proposed) | ✓ | ✓ | ✓ | Phase contribution measurement |

Additionally, data variants:
- **V-only:** voltage channel only, no current
- **I-only:** current channel only
- **V+I:** both channels (full system)

The comparison M3 → M4 directly measures the accuracy contribution of harmonic phase features. The comparison V-only → V+I measures the contribution of simultaneous current measurement.

---

## 12. Dataset Generation

### 12.1 Synthetic Signal Models and Parameters

All training data is generated synthetically using parametric models with carefully chosen distributions. The critical design choice for the Harmonic Distortion class is using **load-specific phase distributions** (not uniform random) to ensure the phase features carry discriminative information.

**Class 0 — Normal:**

```
v(t) = V₁ sin(2π·50·t + φ₁) + Σ[small background harmonics] + ε(t)
Parameters:
  V₁ ~ Uniform(200, 245) V peak
  φ₁ ~ Uniform(0, 2π) rad
  Background THD ~ Uniform(1%, 4%) with random harmonic mix
  ε(t): AWGN with σ ∈ [0.002, 0.010] × V₁
```

**Class 1 — Voltage Sag:**

```
v(t) = V₁ sin(2π·50·t) × [1 - d · rect((t - t_start)/duration)]
Parameters:
  Sag depth d ~ Uniform(0.1, 0.9)          (10%–90% magnitude reduction)
  Duration ~ Uniform(0.5, 30) cycles
  t_start ~ Uniform(0, T_window)
```

**Class 2 — Voltage Swell:**

```
v(t) = V₁ sin(2π·50·t) × [1 + d · rect((t - t_start)/duration)]
Parameters:
  Swell magnitude d ~ Beta(1.5, 5.0)       (biased toward small swells: 5%–25% typical)
  Duration ~ Uniform(0.5, 30) cycles
```

Note: Beta(1.5, 5.0) for swell depth is more physically realistic than Uniform(0.1, 0.8). Large swells (>30%) are rare in real systems and oversampling them would bias the classifier.

**Class 3 — Interruption:**

```
v(t) ≈ V_residual + ε(t) for t ∈ [t_start, t_end]
Parameters:
  Residual ~ Uniform(0, 0.1) × V₁
  Duration ~ Uniform(1, 5) cycles
```

**Class 4 — Harmonic Distortion (with load-specific phase distributions):**

This class uses THREE load subtypes with distinct harmonic phase signatures:

```
Subtype A — SMPS / Bridge Rectifier (single-phase):
  3rd harmonic: φ₃ ~ Von Mises(μ=+π/4, κ=3.0)   ← concentrated near +45°
  5th harmonic: φ₅ ~ Von Mises(μ=-π/5, κ=2.5)
  7th harmonic: φ₇ ~ Von Mises(μ=+π/7, κ=2.0)
  V₃/V₁ ~ Uniform(0.3, 0.9),  V₅/V₁ ~ Uniform(0.1, 0.4)

Subtype B — 6-Pulse VFD Rectifier:
  5th harmonic: φ₅ ~ Von Mises(μ=-2π/5, κ=4.0)  ← locked to firing angle
  7th harmonic: φ₇ ~ Von Mises(μ=+2π/7, κ=4.0)
  V₅/V₁ ~ Uniform(0.15, 0.25), V₇/V₁ ~ Uniform(0.08, 0.15)
  (odd triplen harmonics suppressed in 3-phase balanced load)

Subtype C — Transformer Saturation:
  3rd harmonic: φ₃ ~ Von Mises(μ=+π/2, κ=3.5)   ← in quadrature (near zero crossing)
  V₃/V₁ ~ Uniform(0.05, 0.20)  (lower THD than SMPS)
```

By using Von Mises distributions (the circular equivalent of Gaussian), the synthetic data correctly represents the physical concentration of harmonic phases around characteristic values, while preserving realistic variability from unit-to-unit differences.

**Class 5 — Transient:**

```
v_transient(t) = A_t · e^(-t'/τ) · sin(2π·f_t·t')    injected at t_start
Parameters:
  A_t ~ Uniform(0.1, 1.5) × V₁
  τ ~ Uniform(0.1, 2.0) ms (decay time constant)
  f_t ~ Uniform(500, 2000) Hz (transient oscillation frequency)
  Polarity ~ {+1, -1} equally probable
```

**Class 6 — Flicker:**

```
v_flicker(t) = V₁ · [1 + m · sin(2π·f_f·t)] · sin(2π·50·t + φ₁)
Parameters:
  Modulation depth m ~ Uniform(0.05, 0.20)     (5%–20%)
  Flicker frequency f_f ~ Uniform(1, 25) Hz    (human perception range)
```

### 12.2 Synthetic Current Channel Generation

For each voltage class, a corresponding current waveform is generated:

```python
def generate_current(v_signal, class_id, params):
    """Generate a realistic load current for the given voltage class."""
    t = np.linspace(0, 1/50, 500, endpoint=False)

    if class_id == 0:  # Normal: mainly resistive + slight inductive
        dpf = np.random.uniform(0.85, 1.0)   # displacement power factor
        phi_shift = np.arccos(dpf)
        i = params['I1'] * np.sin(2*np.pi*50*t - phi_shift) + \
            np.random.normal(0, params['noise_sigma'], 500)

    elif class_id == 4:  # Harmonic: SMPS, VFD, or transformer per subtype
        # Phase-coherent current harmonics with load-specific structure
        # (parallel to voltage harmonic generation above)
        ...

    return i
```

### 12.3 Dataset Size and Split

| Split | Samples per class | Total samples |
| --- | --- | --- |
| Training | 4,000 | 28,000 |
| Validation | 750 | 5,250 |
| Test | 750 | 5,250 |
| **Total** | **5,500** | **38,500** |

All classes are balanced. Equal class sizes prevent majority-class bias.

### 12.4 Noise Augmentation

Each training sample is augmented at runtime during training:

| Augmentation | Parameter | Rationale |
| --- | --- | --- |
| AWGN | SNR ∈ {20, 30, 40, 50, ∞} dB chosen uniformly | Robustness to measurement noise |
| Amplitude jitter | Fundamental amplitude ± 5% | ADC calibration drift |
| Phase offset jitter | Fundamental phase ± 10° | Imperfect ZC alignment |
| Sampling jitter | ± 0.5 sample period (interpolated) | IntervalTimer imprecision |

These augmentations effectively triple the training set size and significantly improve generalisation from synthetic to real hardware data.

---

## 13. Training Procedure and Evaluation

### 13.1 Data Preparation

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Assume: X_wave (38500, 500, 2), X_mag (38500, 28), X_phase (38500, 254), y (38500,)
# Integer labels 0–6 → one-hot
y_onehot = tf.keras.utils.to_categorical(y, num_classes=7)

# Scale only feature inputs (not waveform which is already normalised)
scaler_mag   = StandardScaler().fit(X_mag_train)
scaler_phase = StandardScaler().fit(X_phase_train)

X_mag_train_s   = scaler_mag.transform(X_mag_train)
X_phase_train_s = scaler_phase.transform(X_phase_train)

joblib.dump(scaler_mag,   'scaler_mag.pkl')
joblib.dump(scaler_phase, 'scaler_phase.pkl')
```

### 13.2 Training Configuration

```python
model = build_phase_aware_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy', save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./tb_logs')
]

history = model.fit(
    [X_wave_train, X_mag_train_s, X_phase_train_s],
    y_train_onehot,
    validation_data=(
        [X_wave_val, X_mag_val_s, X_phase_val_s],
        y_val_onehot
    ),
    batch_size=64,
    epochs=100,
    callbacks=callbacks
)
```

### 13.3 Evaluation Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_probs = model.predict([X_wave_test, X_mag_test_s, X_phase_test_s])
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(y_test_onehot, axis=1)

class_names = ['Normal', 'Sag', 'Swell', 'Interruption',
               'Harmonic', 'Transient', 'Flicker']

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — M4 Phase-Aware Model')
plt.savefig('confusion_matrix_M4.png', dpi=150, bbox_inches='tight')
```

### 13.4 Expected Performance — Honest Assessment

| Model | Expected Accuracy (Synthetic noiseless) | Expected at 30 dB SNR | Expected on Real Hardware (zero-shot) | After Domain Adaptation |
| --- | --- | --- | --- | --- |
| M1: Magnitude MLP | 91–94% | 85–88% | 72–80% | N/A (not adapted) |
| M2: CNN-LSTM raw | 95–97% | 90–93% | 78–85% | 88–93% |
| M3: CNN-LSTM + Mag | 96–98% | 92–95% | 80–87% | 90–94% |
| M4: Full (proposed) | 97–99% | 94–97% | 82–89% | 91–95% |

**Why the real hardware numbers are lower:** Synthetic signals have clean parametric harmonic phase structures. Real waveforms have: ADC quantisation noise, power supply ripple, ground bounce, cable coupling, load switching transients, and background grid harmonics — all absent from the synthetic model. The 10–15 percentage point drop is consistent with published comparisons in the PQ classification literature (Barros et al. 2020). Domain adaptation via fine-tuning recovers 5–8 percentage points of this gap.

### 13.5 Domain Adaptation Procedure

```python
# Load pre-trained M4, freeze all layers except the last two Dense layers
model.trainable = False
for layer in model.layers[-4:]:   # Dense(64) + Dropout(0.4) + Dense(7)
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),   # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune on 100–200 real samples per class (700–1400 total)
history_ft = model.fit(
    [X_wave_real, X_mag_real_s, X_phase_real_s],
    y_real_onehot,
    validation_split=0.2,
    batch_size=16,
    epochs=30,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)
```

---

## 14. Experimental Plan and Demo Setup

### 14.1 Laboratory Load Configuration

**Load A — Resistive (Sinusoidal Baseline):**
60 W or 100 W incandescent lamp or resistance wire heater. Near-ideal sinusoidal current. Baseline for verifying “Normal” classification and confirming calibration.

**Load B — LED Lamp (Harmonic Source):**
Common LED bulb without PFC. Typically draws heavily distorted current (THD_i = 80–150%). 3rd harmonic is dominant with a characteristic phase structure related to the capacitive input filter. Expected classifier output: Harmonic Distortion (Subtype A — SMPS).

**Load C — AC Motor (Multiple Event Types):**
By varying motor load and supply voltage (via Variac autotransformer):
- Nominal operation: Normal (with minor motor harmonics)
- Reduced voltage: Sag condition
- Elevated voltage: Swell condition
- Motor starting (inrush): Transient followed by steady-state harmonics

**Variac Safety Note:** Standard Variacs are autotransformers — their output neutral is electrically connected to mains neutral. This is NOT galvanically isolated. The AMC1301 provides isolation between the mains-referenced measurement circuit and the MCU. Keep the Variac output terminals covered when not actively adjusting the voltage. Do not operate the Variac above 250 V. See Section 18 for full safety protocol.

### 14.2 Step-by-Step Demo Procedure

**Step 1 — Pre-power verification (no mains):**
- Measure GND1 to GND2 resistance: must be > 10 MΩ. If < 1 MΩ, there is a wiring fault. Do NOT proceed.
- Verify B0505S input/output with bench supply (5 V in → 5 V isolated out measured at VDD1 pin).

**Step 2 — Quiescent reading (mains, no load):**
- Connect to mains through 1 A fuse. No load connected.
- Expected: 50 Hz fundamental ~230 V, THD_V < 5%, phase readings stable within ±5°.
- Classifier output: Normal (with occasional transient from switching loads on same phase).

**Step 3 — Load A (Resistive):**
- Connect 100 W resistive load.
- Expected: Current approximately sinusoidal, V-I phase angle ~0°, current THD < 5%.
- Classifier: Normal.

**Step 4 — Load B (LED lamp):**
- Connect LED lamp.
- Expected: FFT shows dominant 3rd harmonic at 150 Hz with characteristic phase ≈ +45°.
- Record: raw waveform screenshot, FFT bar chart, classifier probability distribution.
- Classifier: Harmonic Distortion.

**Step 5 — Load C (AC motor, various conditions):**
- Nominal → sag (Variac to 180 V) → swell (Variac to 250 V).
- Record classification output at each operating point.

**Step 6 — Combined load:**
- Load A + Load B simultaneously.
- Observe THD increase and classifier response to mixed load condition.

### 14.3 Real Data Collection for Domain Adaptation

Collect at minimum 100 windows per class (200 recommended) from hardware measurements where class label is known from physical setup. Store with timestamp and load configuration metadata. Use these for the fine-tuning procedure in Section 13.5.

---

## 15. Hardware Design Audit — All Issues, Root Causes, and Fixes

This section documents every design issue identified during circuit review. Issues are classified as Critical (C), Significant (S), or Minor (M).

### Issue 1 [C] — Voltage Divider Too Small — AMC1301 Input Clips on Swells

**Original design:** R1 = 1 MΩ, R2 = 768 Ω → divider ratio = 7.68 × 10⁻⁴
At 270 V RMS peak = 382 V: V_INP = 293 mV — exceeds ±250 mV IEC-compliant operation limit.

**Risk:** Input clipping, nonlinear distortion, potential IC damage during transients.

**Fix:** R1 = 2.2 MΩ, R2 = 560 Ω → divider ratio = 2.545 × 10⁻⁴. At 382 V peak: 97 mV (39% of limit). Comfortable headroom.

---

### Issue 2 [C] — ACS758 at 5 V Output into 3.3 V ADC — ADC Pin Damage

**Original design:** ACS758 powered from 5 V. Quiescent VOUT = 2.5 V. At +30 A: VOUT = 2.5 + 1.2 = 3.7 V. Teensy ADC absolute maximum input = 3.3 V.

**Risk:** Immediate permanent damage to Teensy ADC1 pin on any current > 20.8 A.

**Fix:** Power ACS758 from Teensy 3.3 V rail. Quiescent = 1.65 V, max VOUT at ±50 A = 2.97 V. Safely within 3.3 V range at all times.

---

### Issue 3 [C] — AMC1301 VDD1 Has No Isolated Supply — Lethal Shock Hazard

**Original design:** VDD1 powered from same 5 V rail as VDD2.

**Risk:** GND1 (mains neutral) connected through IC internal circuit to GND2 (Teensy GND) → to PC chassis via USB. Direct mains-to-chassis connection. Bypasses all isolation. Potentially lethal.

**Fix:** B0505S-1W isolated DC-DC converter. Input from Teensy VUSB. Output (isolated 5V) to AMC1301 VDD1 only. GND1 floats relative to all MCU/PC grounds.

---

### Issue 4 [C] — AMC1301 Output Voltage Exceeds Teensy ADC Range (Without Divider)

**Original design:** VDD2 = 5 V, output centred at 2.5 V, peak excursion ±2.05 V → range 0.45–4.55 V. Teensy ADC max = 3.3 V.

**Risk:** Positive half-cycles clipped above 3.3 V. Systematic measurement error > 30%. Differential waveform highly asymmetric → incorrect harmonic phases.

**Fix:** 1 kΩ / 2 kΩ resistive divider on OUTP scales output to (0.30–3.03 V) for ±270 V mains. Maps to 0.91× Teensy full-scale — excellent ADC utilisation.

---

### Issue 5 [C] — N = 512 FFT Creates Spectral Leakage — Phase Angles Inaccurate

**Original design:** N = 512 (power-of-two for speed), fs = 5000 → Δf = 9.765625 Hz. Fundamental at bin 5.12 (non-integer). Harmonic energy spreads over 3–5 adjacent bins.

**Risk:** Phase angle extraction from non-centred bins is inaccurate by 10–30°. This completely undermines the project’s core innovation (phase-aware classification) because the extracted phases are numerically meaningless artefacts of leakage, not physical signatures.

**Fix:** N = 500, fs = 5000 → Δf = 10.00 Hz exactly. All harmonics at exact integer bins. Zero spectral leakage for stationary harmonics.

---

### Issue 6 [S] — No Anti-Aliasing Filter — Aliased Noise Corrupts Harmonic Bins

**Original design:** No filter between sensor outputs and ADC inputs.

**Risk:** AMC1301 bandwidth 200 kHz; ACS758 bandwidth 120 kHz. Both pass substantial energy above Nyquist (2500 Hz). Switching noise and motor drive PWM aliases into 0–2500 Hz band, creating spurious harmonic-like content.

**Fix:** RC low-pass filter on each channel. Voltage: 390 Ω + 100 nF, fc ≈ 1505 Hz. Current: 390 Ω + 100 nF, fc ≈ 1600–3200 Hz.

**Residual limitation:** -5.7 dB at Nyquist is insufficient for industrial environments with VFDs. A 2nd-order Sallen-Key filter is recommended for production use.

---

### Issue 7 [S] — Sequential ADC Reads Create Inter-Channel Phase Error

**Original design:** `analogRead(A0)` then `analogRead(A10)` in sequence.

**Analysis:** Each analogRead ≈ 3–4 µs. At 650 Hz (13th harmonic): delay = 3 µs → phase error = (3e-6 / (1/650)) × 360° = 0.70°. This is small but accumulates into the cross-channel phase features (V-I phase at h=13), potentially corrupting the V-I power flow direction calculation used in future source localisation.

**Fix:** `adc->readSynchronizedSingle(A0, A10)` — true hardware simultaneous acquisition. Zero inter-channel delay.

---

### Issue 8 [S] — 0.1% Resistors Not Locally Available

**Original design:** Specified 0.1% tolerance for R1, R2.

**Fix:** 1% metal-film resistors (readily available at SP Road, Bengaluru) introduce ±2.0% absolute gain error on voltage reading. Compensated by one-time calibration. THD and phase calculations are gain-error-immune.

---

### Issue 9 [M] — Capacitor Across R2 Corrupts Harmonic Measurements

**Original design:** 0.1 µF placed across R2 (560 Ω) “for HF filtering.”

**Analysis:** At 650 Hz: X_C = 1/(2π×650×100nF) = 2449 Ω. In parallel with 560 Ω: effective impedance = 460 Ω — an 18% reduction from nominal. This creates an 18% attenuation of the 13th harmonic specifically relative to the fundamental, appearing as a false reduction in high-order harmonic content. Masquerades as a filter response but is actually an accuracy error.

**Fix:** Remove capacitor entirely. Anti-aliasing is done after the AMC1301 output, not inside the divider.

---

### Issue 10 [M] — GND Labelling Ambiguity — Safety Risk from Misinterpretation

**Original design:** GND at bottom of voltage divider unlabelled.

**Risk:** If a builder incorrectly connects R2’s low-side to MCU_GND (Teensy ground), mains neutral is directly connected to the MCU — catastrophic isolation defeat.

**Fix:** All schematics and PCB labels use explicit names: MAINS_GND (= GND1 = mains neutral side) and MCU_GND (= GND2 = Teensy ground). Physical colour coding: MAINS_GND wiring in orange, MCU_GND in black.

---

### Issue 11 [M] — Absolute Phase Meaningless Without Zero-Crossing Reference

**Original design:** FFT computed on arbitrary window boundaries, no mains synchronisation.

**Risk:** φ₁ changes randomly window-to-window. The inter-harmonic phase differences Δφₕ = φₕ − φ₁ are stable, but sin/cos encoded absolute phases in Branch 3 are meaningless random vectors. The model cannot learn the physical phase signatures.

**Fix:** ZC synchronisation in firmware (Section 9.2). φ₁ is consistently ≈ 0 radians. All harmonic phases are physically meaningful.

---

### Issue 12 [C — Software] — CRC32 Computed Over Post-Increment Sequence Number

**Original firmware bug in `sendFrame()`:**

```cpp
Serial.write((uint8_t*)&seqNum, 2);   // Transmit seqNum = N
seqNum++;                              // Now seqNum = N+1
uint32_t crc = computeCRC32(&seqNum, ...);  // CRC over N+1, not N — MISMATCH!
```

**Risk:** Every single frame fails CRC verification on the Python receiver. System appears to receive data but discards all frames as corrupt.

**Fix:** Capture `txSeqNum = seqNum` before transmit; compute CRC over `txSeqNum`; increment seqNum only after CRC transmission. (See corrected code in Section 9.3.)

---

### Issue 13 [C — Software] — Raw Phase Angles Included in Feature Vector (Redundant + Harmful)

**Original design:** 26 raw phase angles (phi_v[h] and phi_i[h] for h=1..13) included in the feature vector alongside their sin/cos encoding.

**Risk:** The sin/cos encoding was introduced specifically to avoid the ±π discontinuity in raw angle representation. Including raw angles reintroduces this discontinuity. Additionally, the features are linearly redundant: raw_angle → (sin, cos) is invertible, so the model receives redundant information that increases input dimensionality without additional information content, wasting model capacity.

**Fix:** Exclude raw phase angles from the feature vector. Use only sin/cos encoded versions. Reduces feature vector from 308 to 282 elements.

---

### Issue 14 [C — Software] — np.mean/np.std Used on Phase Angles (Circular Statistics Bug)

**Original design:** `np.mean(odd_phases_v)` and `np.std(odd_phases_v)` for phase group statistics.

**Risk:** Arithmetic mean of angles near ±π is catastrophically wrong (can point in the opposite direction from the true mean). This creates systematically misleading features for load types whose harmonic phases cluster near ±π (e.g., negative-sequence harmonics from VFDs).

**Fix:** `scipy.stats.circmean()` and `scipy.stats.circstd()` with `high=np.pi, low=-np.pi`. (See corrected code in Section 10.5.)

---

### Issue 15 [S] — B0505S Isolation Weaker Than AMC1301 — System Isolation Degraded

**Analysis:** AMC1301 provides 7070 V_PEAK reinforced isolation. B0505S provides 1000 V_DC (≈707 V RMS) isolation only. Overall system isolation is limited to the weaker component: 707 V RMS. IEC 61010-1 Measurement Category II requires 2500 V test voltage for 230 V equipment — B0505S does not meet this for a formally certified instrument.

**Mitigation for lab use:** All precautions in Section 18 are followed, including 1 A fuse, physical enclosure, and supervisor presence.

**Production upgrade:** Replace B0505S with TRACO TMA 0505D (1000 V RMS isolation) or Murata MEE1S0505SC.

---

### Issue 16 [M] — Synthetic Swell Distribution Unrealistic

**Original design:** Swell depth d ~ Uniform(0.1, 0.8) → 110%–180% of nominal with equal probability.

**Reality:** In power distribution networks, swells above 130–140% of nominal are extremely rare (usually indicate faults, not normal disturbance conditions) and short in duration. Uniform distribution over 110–180% overrepresents large swells.

**Fix:** Beta(1.5, 5.0) distribution for swell depth, biasing toward smaller swells (10–25%) that represent the vast majority of real swell events.

---

## 16. Complete Bill of Materials

### 16.1 Sensing Frontend — Core Analog Circuit

| Ref | Component | Part Number | Qty | Tol. | Source | Cost (₹) |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | Isolated differential amplifier | AMC1301DWVR (SOIC-8 DWV) | 1 | — | Robu.in / LCSC | 131 |
| U2 | Hall-effect current sensor | ACS758LCB-050B-PFF-T breakout | 1 | — | Evelta / Amazon | 416 |
| U3 | Isolated DC-DC converter | B0505S-1W SIP-7 | 1 | — | Robu.in | 80 |
| R1a,b | High-voltage divider upper | 1.1 MΩ, 1/4 W, 1% metal film (×2 in series) | 2 | 1% | SP Road | 15 |
| R2 | High-voltage divider lower | 560 Ω, 1/4 W, 1% metal film | 1 | 1% | SP Road | 5 |
| R3 | Output divider upper | 1 kΩ, 1/4 W, 5% | 1 | 5% | SP Road | 2 |
| R4 | Output divider lower | 2 kΩ, 1/4 W, 5% | 1 | 5% | SP Road | 2 |
| R5 | Anti-alias series R (V ch) | 390 Ω, 1/4 W, 5% | 1 | 5% | SP Road | 2 |
| R6 | Anti-alias series R (I ch) | 390 Ω, 1/4 W, 5% | 1 | 5% | SP Road | 2 |
| R7a,b | REFIN bias divider | 10 kΩ, 1/4 W, 5% (×2) | 2 | 5% | SP Road | 3 |
| C1 | Anti-alias cap (V ch) | 100 nF, 50 V, NP0/C0G ceramic | 1 | 5% | SP Road | 3 |
| C2 | Anti-alias cap (I ch) | 100 nF, 50 V, NP0/C0G ceramic | 1 | 5% | SP Road | 3 |
| C3 | VDD1 HF bypass | 100 nF, 50 V, NP0 ceramic | 1 | — | SP Road | 2 |
| C4 | VDD1 bulk reservoir | 10 µF, 25 V, electrolytic | 1 | — | SP Road | 3 |
| C5 | VDD2 HF bypass | 100 nF, 50 V, NP0 ceramic | 1 | — | SP Road | 2 |
| C6 | VDD2 bulk | 10 µF, 25 V, electrolytic | 1 | — | SP Road | 3 |
| C7 | ACS758 VCC HF bypass | 100 nF, 50 V, NP0 ceramic | 1 | — | SP Road | 2 |
| C8 | ACS758 VCC bulk | 10 µF, 25 V, electrolytic | 1 | — | SP Road | 3 |
| C9 | REFIN filter | 100 nF, 25 V, ceramic | 1 | — | SP Road | 2 |
| F1 | Mains fuse + holder | 1 A, 250 V, slow-blow ceramic + PCB holder | 1 | — | SP Road | 25 |
| — | Perfboard | 10 × 8 cm double-sided | 1 | — | SP Road | 30 |
| — | Mains terminal block | 2-pin 5.08mm pitch, 300V rated | 1 | — | SP Road | 15 |
| — | ABS enclosure | 120 × 80 × 40 mm plastic box | 1 | — | SP Road | 40 |

**Frontend Subtotal: ₹791**

### 16.2 Microcontroller and Host Interface

| Component | Part Number | Source | Cost (₹) |
| --- | --- | --- | --- |
| PJRC Teensy 4.1 | Teensy 4.1 with headers | Amazon / Robu | 3,800 |
| USB-A to Micro-USB cable | — | SP Road | 80 |

**MCU Subtotal: ₹3,880**

### 16.3 Test Loads

| Component | Notes | Source | Cost (₹) |
| --- | --- | --- | --- |
| 100 W incandescent lamp + socket | Load A: sinusoidal reference | Any hardware shop | 80 |
| LED bulb (non-PFC, 7–12 W) | Load B: harmonic source | Any electrical shop | 60 |
| AC motor (small, 50–200 W) | Load C: multiple event types | Lab stock | (lab) |

**Loads Subtotal: ₹140**

### 16.4 Grand Total

| Category | Cost (₹) |
| --- | --- |
| Sensing frontend | 791 |
| Teensy 4.1 + cable | 3,880 |
| Test loads | 140 |
| Miscellaneous (wire, solder, flux, heat-shrink) | 150 |
| **Grand Total** | **~₹4,961** |
| Sensing frontend alone | **₹791** |

---

## 17. Complete Wiring Diagram and Connections

```
═══════════════════════════════════════════════════════════════════════
  MAINS INPUT (230V AC, 50Hz)  —  DANGER: HIGH VOLTAGE
  Use proper insulated wire rated ≥ 300V AC, ≥ 0.5 mm²
═══════════════════════════════════════════════════════════════════════

LIVE ─────────┬───────────────────────────────────────────────────►
              │
         [F1: 1A 250V Slow-Blow Fuse]
              │
              ├──────────────────────────────── through ACS758 hole ──►
              │   (Live wire physically threaded through sensor body)
              │
         [R1a: 1.1MΩ, 1% metal film]  ← high voltage rated 1/4W or above
              │
         [R1b: 1.1MΩ, 1% metal film]  ← two in series = 2.2MΩ total
              │
              ├──────────────────────────────────────────────────────
              │                                                      │
         [R2: 560Ω, 1%]               ← bottom of voltage divider  │
              │                                                      │
         MAINS_GND ←──────────────────────────────────────────────┘
           (= Neutral wire = GND1 = AMC1301 high-side reference)
           NEVER connect MAINS_GND to MCU_GND!

Voltage at R1/R2 junction (= AMC1301 INP):
  230V RMS nominal: ±82.7 mV peak (safe)
  270V RMS swell:   ±97.2 mV peak (safe, 39% of ±250mV limit)

═══════════════════════════════════════════════════════════════════════
  B0505S-1W ISOLATED DC-DC CONVERTER
═══════════════════════════════════════════════════════════════════════

Teensy VUSB (5V) ──────────────────► B0505S Pin IN+ (5V input)
Teensy GND (MCU_GND) ──────────────► B0505S Pin IN- (input GND)
                                      ┌──────────────────────────────┐
                                      │  ⚡ ISOLATION BARRIER ⚡     │
                                      │  ~1000V DC (707V RMS)        │
                                      └──────────────────────────────┘
B0505S Pin OUT+ (isolated 5V) ─────► AMC1301 Pin 1 (VDD1)
                                      [C3: 100nF NP0 to MAINS_GND]
                                      [C4: 10µF electrolytic to MAINS_GND]
B0505S Pin OUT- (isolated GND) ────► AMC1301 Pin 4 (GND1) = MAINS_GND
                                      This is the MAINS NEUTRAL reference.
                                      NEVER connect to MCU_GND.

═══════════════════════════════════════════════════════════════════════
  AMC1301 SOIC-8 DWV PINOUT AND CONNECTIONS
═══════════════════════════════════════════════════════════════════════

  Pin 1 (VDD1)  ← B0505S OUT+ isolated 5V + C3 (100nF NP0) to GND1
  Pin 2 (INN)   ← MAINS_GND (= GND1 = Neutral)
  Pin 3 (INP)   ← R1/R2 junction (divided mains voltage)
  Pin 4 (GND1)  ← MAINS_GND (= B0505S OUT- = Neutral)
  Pin 5 (VDD2)  ← Teensy VUSB (5V) + C5 (100nF NP0) + C6 (10µF) to MCU_GND
  Pin 6 (OUTP)  ─────────────────────────── voltage signal path (see below)
  Pin 7 (OUTN/REFIN) ── tied to midpoint of R7a/R7b bias divider (= 2.5V)
                         R7a: 10kΩ from VDD2 (5V) to junction
                         R7b: 10kΩ from junction to MCU_GND
                         C9: 100nF from junction to MCU_GND
  Pin 8 (GND2)  ← MCU_GND (Teensy GND)

REFIN set to 2.5V → OUTP midpoint = 2.5V (at 0V mains input)

─── OUTPUT SIGNAL CHAIN ───────────────────────────────────────────────

AMC1301 Pin 6 (OUTP)
         │
        [R3: 1kΩ]   ← Output voltage divider upper leg
         │
         ├──────────────────────────────────────────────────────────►
         │                                                           │
        [R4: 2kΩ]   ← Output voltage divider lower leg           [R5: 390Ω] ← anti-alias series R
         │                                                           │
       MCU_GND                                               [C1: 100nF NP0] to MCU_GND
                                                                     │
                                                     ─────────────►  Teensy A0 (pin 14)

Expected voltages at Teensy A0:
  Mains = 0V:      1.668V (ADC count ≈ 2071)
  Mains = +270V:   1.933V (ADC count ≈ 2398)
  Mains = -270V:   1.402V (ADC count ≈ 1740)
  All within 0–3.3V ✓

═══════════════════════════════════════════════════════════════════════
  ACS758 CURRENT SENSOR
═══════════════════════════════════════════════════════════════════════

Teensy 3.3V pin ──────────────────────► ACS758 VCC
                                         [C7: 100nF NP0 < 5mm from VCC pin]
                                         [C8: 10µF electrolytic < 20mm from VCC pin]
Teensy GND (MCU_GND) ─────────────────► ACS758 GND

Load current path:
  Live wire (AFTER fuse) physically threaded through ACS758 sensor hole
  Minimum wire gauge: 1.5 mm² (16 AWG) for up to 10A continuous

ACS758 VOUT ──[R6: 390Ω]──┬──────────► Teensy A10 (pin 24)
                            │
                   [C2: 100nF NP0] to MCU_GND

Expected voltages at Teensy A10 (VCC = 3.3V):
  0A current:  1.650V (ADC count ≈ 2048)
  +10A:        1.914V (ADC count ≈ 2376)
  -10A:        1.386V (ADC count ≈ 1720)
  All within 0–3.3V ✓

═══════════════════════════════════════════════════════════════════════
  TEENSY 4.1 CONNECTIONS SUMMARY
═══════════════════════════════════════════════════════════════════════

  Pin       Signal                      Notes
  ────────  ──────────────────────────  ─────────────────────────────
  VUSB      → B0505S IN+               5V only when USB-powered
  VUSB      → AMC1301 Pin 5 (VDD2)    Low-side supply
  3.3V      → ACS758 VCC              Ratiometric supply
  GND       → MCU_GND (all low-side)
  A0 pin14  → Voltage signal           ADC0 hardware channel
  A10 pin24 → Current signal           ADC1 hardware channel
  USB       → PC                       Data + power
```

---

## 18. Safety Guidelines

### 18.1 Mandatory Pre-Build Precautions

This circuit connects to 230 V mains. Improper construction or use can cause **electric shock resulting in serious injury or death.** All team members read and acknowledge these precautions before assembly begins.

**Enclosure:**
All mains-connected components — fuse (F1), mains terminal block, R1a, R1b, R2, the ACS758 with its load wire, and AMC1301 high-side (pins 1–4) — must be enclosed in an insulated ABS plastic box before any mains connection. No exposed high-voltage conductors at any time during operation.

**Creepage and clearance:**
Minimum creepage distance between MAINS_GND and MCU_GND traces/wires on the perfboard: ≥ 6 mm (reinforced isolation, 250 V working voltage, pollution degree 2 per IEC 61010-1). Add a physical slot cut through the perfboard if needed to guarantee this spacing.

**Fuse:**
F1 (1 A, 250 V, slow-blow ceramic) MUST be present in the LIVE conductor, before any circuit connection point. Use only 1 A; do not use any rating higher than 2 A for this project.

**Isolation verification before EVERY power-on:**
Set DMM to high-resistance (Ω) mode. Measure resistance between MAINS_GND and MCU_GND. Required: > 10 MΩ. If < 1 MΩ: wiring fault creating dangerous connection. Do NOT apply mains until resolved.

**USB connection:**
The Teensy USB cable connects to a PC/laptop. While AMC1301 provides reinforced isolation, the PC is still in the circuit. Keep the PC on a non-conductive surface (rubber mat). Do not touch the laptop chassis while the mains connector is live.

**One-hand rule:**
When the mains connector is live, interact with the circuit using only one hand. Keep the other hand in your pocket or behind your back. This prevents current from flowing across the chest (across the heart) if accidental contact occurs.

**Supervisor presence:**
NEVER energise the mains connection without at least one other person present who knows the location of the circuit breaker and the emergency procedure below.

### 18.2 AC Motor-Specific Precautions

**Inductive kickback:** AC motors generate voltage spikes significantly above 270 V at the moment of disconnection. Always disconnect by removing the mains plug, NOT by disconnecting motor wires under power.

**Motor body grounding:** Motor frame must be connected to Earth (PE) ground. Do not connect motor frame to MAINS_GND or MCU_GND.

**Variac (autotransformer):** Variac output neutral IS electrically connected to mains neutral — it is NOT galvanically isolated. Keep all Variac output terminals covered when not adjusting. Do not exceed 250 V output.

### 18.3 Emergency Procedure

If any component smokes, sparks, or emits unusual odour:
1. Immediately pull the mains plug from the wall socket.
2. Do NOT attempt to diagnose while live.
3. Wait 30 seconds for capacitors to discharge.
4. Verify with DMM (AC Voltage mode) that mains terminals read 0 V AC before touching anything.
5. Investigate root cause, resolve, and repeat isolation verification test before next power-on.

---

## 19. Implementation Timeline

### 19.1 Detailed 10-Week Schedule

| Week | Milestone | Owner | Deliverable / Verification |
| --- | --- | --- | --- |
| 1 | Theory & specifications complete | Rayan | Documented class definitions, harmonic parameter ranges, IEEE 519 limit table |
| 1 | Python synthetic signal generator (all 7 classes, load-specific phase distributions) | Japeth | `synthetic_generator.py` with Von Mises phase sampling for Class 4 subtypes |
| 2 | Time-domain + FFT feature extraction with N=500 | Yashas | `feature_extractor.py` — unit tests confirm harmonic magnitudes AND phases correct on synthetic known-phase test signals |
| 2 | Baseline MLP model (M1) trained and evaluated | Japeth | Confusion matrix, accuracy vs SNR table for M1 |
| 3 | Phase-aware features with correct circular statistics | Yashas | `phase_features.py` — verified with circular mean test: mean([+175°, -175°]) ≈ ±180°, not 0° |
| 3 | DWT feature extraction | Yashas | `wavelet_features.py` — energy-frequency validation per subband |
| 4 | CNN-LSTM model (M4) implemented and trained | Japeth | Model summary, training curves, M1–M4 ablation accuracy table |
| 4 | Full ablation study (M1–M4 × V-only/I-only/V+I) | Japeth | Accuracy table (12 conditions), confusion matrices, ablation bar chart |
| 5 | B0505S + AMC1301 circuit assembled on perfboard | Yash | Scope output tracks 230V waveform shape and scale correctly |
| 5 | ACS758 circuit assembled and tested | Yash | Output voltage at known test current verified |
| 6 | Teensy firmware complete: dual-ADC, ZC sync, USB serial with correct CRC | Yash | `pq_firmware.ino` — captured frames pass CRC check in Python on 100 consecutive frames |
| 6 | Python USB receiver + preprocessing complete | Yashas | Real-time waveform display on PC from hardware |
| 7 | Real waveform FFT validation | Rayan + Yashas | Captured FFT from LED lamp matches expected harmonic signature (3rd dominant, characteristic phase) |
| 7 | End-to-end pipeline integration test | All | Real hardware → USB → features → trained model → classification displayed in real time |
| 8 | Real data collection: Load A, B, C (100–200 windows/class minimum) | Rayan + Yash | Labelled dataset of real hardware measurements |
| 8 | Domain adaptation fine-tuning | Japeth | Accuracy comparison before/after fine-tuning on real data |
| 9 | Results compilation | All | Final accuracy tables, confusion matrices, representative waveform and FFT plots |
| 9 | Report first draft | All | Sections distributed per team role |
| 10 | Report review, revision, finalisation | All | Final project report |
| 10 | Demo preparation and rehearsal | All | Slide deck + live demo script + fallback offline demo (recorded waveforms) |

---

## 20. Known Limitations and Future Extensions

### 20.1 Known Limitations

**Limitation 1 — ADC Effective Resolution: 9.4 bits**
The passive single-ended OUTN/REFIN approach uses only half the AMC1301 differential output swing. The resulting 9.4 effective ADC bits is adequate for detecting harmonics at 5%+ of fundamental, but misses low-THD distortion (1–3% harmonic content). A difference amplifier output stage using TLV901x (as per TI SBAA229 App Note) would double the effective swing and deliver 12 effective bits for ~₹20 additional cost.

**Limitation 2 — Anti-Alias Filter Order (-5.7 dB at Nyquist)**
The 1st-order RC filter provides inadequate rejection for real industrial environments with VFDs. A 2nd-order Sallen-Key filter would provide -20 dB at Nyquist with the same component types.

**Limitation 3 — B0505S Isolation (707 V RMS vs IEC 61010-1 requirement of 2500 V)**
Sufficient for supervised lab use; inadequate for a formally certified commercial instrument. Upgrade to TRACO TMA 0505D for production use.

**Limitation 4 — ZC-Triggered Windows Miss Mid-Cycle Events**
Short transients and sags that begin mid-cycle and end before the next rising ZC may not be fully captured in any single window. A parallel event-triggered capture mode would address this.

**Limitation 5 — Synthetic-to-Real Domain Gap (10–15% accuracy drop without fine-tuning)**
The synthetic dataset, even with Von Mises phase distributions, does not fully capture the noise characteristics and harmonic structure of real hardware signals. Domain adaptation is necessary for high-accuracy real-world deployment.

**Limitation 6 — Single-Phase Only**
The current design measures one phase voltage and one load current. Three-phase industrial systems require three voltage channels and three current channels.

### 20.2 Future Extensions

**Extension 1 — Three-Phase System:**
Add three AMC1301 channels (one per phase voltage, each with dedicated B0505S) and three ACS758 channels. Enables sequence component analysis (positive/negative/zero sequence), unbalance detection, and detection of negative-sequence harmonics (5th, 11th) that cause motor heating.

**Extension 2 — Embedded Inference:**
Compress the trained CNN-LSTM using TFLite quantisation (INT8 weights, 4× size reduction) and deploy on Raspberry Pi 4 or 5. Eliminates PC dependency for field deployment. TFLite Micro for Cortex-M7 may be feasible for the MLP baseline; the full CNN-LSTM benefits from Raspberry Pi compute.

**Extension 3 — Wireless Monitoring Network:**
Replace USB cable with ESP32 module (UART-to-WiFi bridge). Multiple nodes report to a central MQTT broker; Node-RED or Grafana dashboard for visualisation. Enables simultaneous multi-point monitoring for harmonic source localisation.

**Extension 4 — Harmonic Source Localisation:**
With simultaneous measurements at two points, harmonic active power P_h = V_h × I_h × cos(φ_Vh − φ_Ih) determines power flow direction per harmonic. Negative P_h indicates harmonic source is downstream (load-side); positive P_h indicates upstream (grid-side). This is a direct application of the project’s phase measurement capability.

**Extension 5 — K-Factor Transformer Derating Calculator:**

```
K = Σ[h=1 to n] (Iₕ/I₁)² × h²
```

K-factor quantifies additional transformer heating from harmonics. A transformer rated K=1 may need 70–80% derating if K > 4. This is a directly deployable output of the monitoring system.

**Extension 6 — 2nd-Order Anti-Alias Filter:**
Add Sallen-Key topology for −20 dB at Nyquist. Suitable for VFD-loaded industrial environments.

**Extension 7 — Attention Mechanism:**
Replace or augment the LSTM branch with a multi-head self-attention layer (Transformer encoder). Published work (Zhang et al., 2025) shows CNN-Transformer outperforms CNN-LSTM at 94.32% at 20 dB SNR vs. typical CNN-LSTM at 89–92%.

---

## 21. References

1. Texas Instruments, *AMC1301 Precision, ±250-mV Input, Reinforced Isolated Amplifier*, Datasheet SBAS667G, Apr 2016, Rev. Apr 2023.
2. Texas Instruments, *Using the AMC130x in Single-Ended Configurations*, Application Note SBAA229, 2019.
3. Allegro Microsystems, *ACS758 Automotive-Grade, Fully Integrated, Hall Effect-Based Linear Current Sensor IC Datasheet*, Rev. 9, 2018.
4. PJRC, *Teensy 4.1 Documentation*, https://www.pjrc.com/store/teensy41.html, accessed Mar 2026.
5. P. Villanueva, *ADC Library for Teensy 4.x/3.x/LC*, GitHub: pedvide/ADC, 2022.
6. IEEE SA Standards Board, *IEEE Standard 519-2022: Harmonic Control in Electric Power Systems*, approved May 2022.
7. IEC 61000-4-30 Ed.3, *Testing and Measurement Techniques — Power Quality Measurement Methods*, 2015.
8. IEC 61000-4-7, *Testing and Measurement Techniques — General Guide on Harmonics and Interharmonics Measurements and Instrumentation*, 2002.
9. IEC 61010-1 Ed.3, *Safety Requirements for Electrical Equipment for Measurement, Control, and Laboratory Use*, 2010.
10. EN 50160:2010+A1:2015, *Voltage Characteristics of Electricity Supplied by Public Electricity Networks*, 2015.
11. J. Arrillaga, N.R. Watson, *Power System Harmonics*, 2nd ed., Wiley, 2003.
12. A. Barros, J. Dent, M. Orr, *“A Comparison of Power Quality Disturbance Detection and Classification Methods Using CNN, LSTM and CNN-LSTM,”* Applied Sciences, vol. 10, no. 19, 2020. doi:10.3390/app10196755.
13. Z. Tong, J. Zhong, J. Li, et al., *“A power quality disturbances classification method based on multi-modal parallel feature extraction,”* Scientific Reports, vol. 13, art. 17655, 2023. doi:10.1038/s41598-023-44399-7.
14. I. Topaloglu, *“Deep Learning Based a New Approach for Power Quality Disturbances Classification,”* Journal of Electrical Engineering & Technology, vol. 18, pp. 77–88, 2023. doi:10.1007/s42835-022-01177-1.
15. N.H. Abdzaid et al., *“LSTM power quality disturbance classification with wavelets and attention mechanism,”* Electrical Engineering, Springer, 2022. doi:10.1007/s00202-022-01667-5.
16. X. Zhang et al., *“Identification and classification of power quality disturbances using CNN-transformer,”* Journal of Electrical Engineering & Technology, Springer, 2025. doi:10.1007/s42835-025-02213-6.
17. Q. Li et al., *“Power Quality Disturbance Classification Strategy Based on Fast S-Transform and an Improved CNN-LSTM Hybrid Model,”* MDPI Processes, vol. 13, no. 3, 2025. doi:10.3390/pr13030743.
18. B. Singh et al., *“A multimodal deep learning model with differential evolution-based optimized features for classification of power quality disturbances,”* Journal of Electrical Systems and Information Technology, Springer, 2025. doi:10.1186/s43067-025-00194-0.
19. S. Mallat, *A Wavelet Tour of Signal Processing*, 3rd ed., Academic Press, 2009.
20. S. Hochreiter, J. Schmidhuber, *“Long Short-Term Memory,”* Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
21. N.I. Fisher, *Statistical Analysis of Circular Data*, Cambridge University Press, 1993.
22. P.E. Jupp, K.V. Mardia, *“A Unified View of the Theory of Directional Statistics,”* International Statistical Review, vol. 57, no. 3, pp. 261–294, 1989.
23. A. Kong, *“Teensy DAQ: ADC Oversampling on the Teensy 4.0,”* andykong.org/blog/teensydaq/, 2020.
24. SciPy Community, *scipy.stats.circmean / circstd*, SciPy v1.11 Documentation, https://docs.scipy.org/doc/scipy/reference/stats.html, 2023.
25. C. Baker, *FastCRC Library for Teensy*, GitHub: FrankBoesing/FastCRC, 2021.

---

## Appendix A — Python Environment Setup

```bash
# Create virtual environment
python3.10 -m venv pq_env
source pq_env/bin/activate     # Linux/Mac
# pq_env\Scripts\activate      # Windows

# Install dependencies
pip install numpy scipy pyserial pywavelets scikit-learn tensorflow matplotlib seaborn joblib

# Verify key package versions
python -c "import tensorflow as tf; print(tf.__version__)"          # should be >= 2.13
python -c "import scipy; print(scipy.__version__)"                   # should be >= 1.10
python -c "import pywt; print(pywt.__version__)"                     # should be >= 1.4
```

---

## Appendix B — Quick-Start Demo Checklist

**Hardware (before any mains connection):**
- [ ] All mains-connected components enclosed in ABS box
- [ ] Fuse F1 (1A, 250V) installed in LIVE conductor
- [ ] Creepage distance MAINS_GND to MCU_GND ≥ 6 mm verified visually
- [ ] Isolation test passed: MAINS_GND to MCU_GND > 10 MΩ on DMM
- [ ] B0505S powered from bench 5V supply: outputs 5V isolated (verified with floating DMM)
- [ ] Supervisor present; circuit breaker location identified

**Software (before connecting Teensy):**
- [ ] Python environment activated
- [ ] `scaler_mag.pkl`, `scaler_phase.pkl`, `best_model.keras` in working directory
- [ ] Correct serial port identified (`ls /dev/ttyACM*` on Linux)
- [ ] Python receiver script runs without errors on dummy/recorded data

**First power-on sequence:**
1. Connect Teensy USB (no mains yet)
2. Verify Teensy VUSB = 5V and 3V3 = 3.3V with DMM
3. Connect mains through 1A fuse with NO LOAD
4. Verify ADC counts near 2071 (voltage channel quiescent) and 2048 (current channel)
5. Connect Load A; verify sinusoidal current waveform on Python display
6. Run classifier; verify “Normal” output

---

*End of Report*

---

**Document version:** 2.0 (Audited & Corrected) | **Date:** March 2026 | **Team:** Mohammed Rayan, Yash, Yashas, Japeth | **Status:** Final — Submission Ready