import os
import argparse
import numpy as np
from scipy.stats import vonmises
from tqdm import tqdm
import concurrent.futures

import dsp

def normalize_waveforms(v_wave: np.ndarray, i_wave: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Strict float32 normalization for Teensy compatibility."""
    v = np.asarray(v_wave, dtype=np.float32)
    i = np.asarray(i_wave, dtype=np.float32)
    v_scale = np.float32(max(float(np.max(np.abs(v))), eps))
    i_scale = np.float32(max(float(np.max(np.abs(i))), eps))
    return (v / v_scale).astype(np.float32), (i / i_scale).astype(np.float32)

def generate_multi_label_sample(labels, rng):
    """
    Layers multiple PQ disturbances onto a single fundamental waveform.
    labels: [Normal, Sag, Swell, Interruption, Harmonics, Transient, Flicker]
    """
    FS = 5000
    N = 500
    t = np.linspace(0, N/FS, N, endpoint=False)
    
    # 1. Shared Fundamental
    V1 = rng.uniform(200, 245) * np.sqrt(2)
    phi1 = rng.uniform(0, 2 * np.pi)
    I1 = rng.uniform(1.0, 15.0)
    dpf = rng.uniform(0.85, 1.0)
    phi_i = phi1 - np.arccos(dpf)
    
    v_env = np.ones(N)
    i_env = np.ones(N)
    
    # 2. Base Amplitude Modulations (Mutually Exclusive)
    if labels[1]:  # Sag
        depth = rng.uniform(0.1, 0.9)
        duration = rng.uniform(0.5, 30.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_env[mask] *= (1.0 - depth)
        i_env[mask] *= (1.0 - depth)
    elif labels[2]:  # Swell
        depth = float(rng.beta(1.5, 5.0)) * 0.7
        duration = rng.uniform(0.5, 30.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_env[mask] *= (1.0 + depth)
        i_env[mask] *= (1.0 + depth)
    elif labels[3]:  # Interruption
        residual = rng.uniform(0.0, 0.1)
        duration = rng.uniform(1.0, 5.0) / 50.0
        t_start = rng.uniform(0, max(0, (N/FS) - duration))
        mask = (t >= t_start) & (t <= t_start + duration)
        v_env[mask] *= residual
        i_env[mask] *= residual

    # 3. Flicker Modulation (Class 6)
    if labels[6]:
        m = rng.uniform(0.05, 0.20)
        f_f = rng.uniform(1.0, 25.0)
        flicker_mod = (1 + m * np.sin(2 * np.pi * f_f * t))
        v_env *= flicker_mod
        i_env *= flicker_mod

    # Apply envelopes to fundamental
    v_sig = V1 * v_env * np.sin(2 * np.pi * 50 * t + phi1)
    i_sig = I1 * i_env * np.sin(2 * np.pi * 50 * t + phi_i)

    # 4. Additive Modulations
    # Harmonics (Class 4)
    if labels[4]:
        subtype = rng.integers(0, 3)
        if subtype == 0:   # SMPS
            phi3 = vonmises.rvs(kappa=3.0, loc=np.pi/4, random_state=rng)
            phi5 = vonmises.rvs(kappa=2.5, loc=-np.pi/5, random_state=rng)
            phi7 = vonmises.rvs(kappa=2.0, loc=np.pi/7, random_state=rng)
            r3, r5, r7 = rng.uniform(0.30, 0.90), rng.uniform(0.10, 0.40), rng.uniform(0.05, 0.20)
            v_sig += (r3 * 0.05 * V1) * np.sin(2 * np.pi * 150 * t + phi3) + \
                     (r5 * 0.02 * V1) * np.sin(2 * np.pi * 250 * t + phi5)
            i_sig += (r3 * I1) * np.sin(2 * np.pi * 150 * t + phi3) + \
                     (r5 * I1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * I1 * 0.5) * np.sin(2 * np.pi * 350 * t + phi7)
        elif subtype == 1:  # VFD
            phi5 = vonmises.rvs(kappa=4.0, loc=-2*np.pi/5, random_state=rng)
            phi7 = vonmises.rvs(kappa=4.0, loc=2*np.pi/7, random_state=rng)
            r5, r7 = rng.uniform(0.15, 0.25), rng.uniform(0.08, 0.15)
            v_sig += (r5 * 0.03 * V1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * 0.02 * V1) * np.sin(2 * np.pi * 350 * t + phi7)
            i_sig += (r5 * I1) * np.sin(2 * np.pi * 250 * t + phi5) + \
                     (r7 * I1) * np.sin(2 * np.pi * 350 * t + phi7)
        else:  # Transformer
            phi3 = vonmises.rvs(kappa=3.5, loc=np.pi/2, random_state=rng)
            r3 = rng.uniform(0.05, 0.20)
            v_sig += (r3 * V1) * np.sin(2 * np.pi * 150 * t + phi3)
            i_sig += (r3 * I1 * 0.8) * np.sin(2 * np.pi * 150 * t + phi3)
    else:
        # Background normal THD
        thd_bg = rng.uniform(0.01, 0.04)
        for h in [3, 5, 7]:
            amp = (thd_bg / np.sqrt(3)) * V1 * rng.uniform(0.5, 1.5)
            v_sig += amp * np.sin(2 * np.pi * h * 50 * t + rng.uniform(0, 2 * np.pi))

    # Transient (Class 5)
    if labels[5]:
        A_t = rng.uniform(0.1, 1.5) * V1
        tau = rng.uniform(0.1, 2.0) / 1000.0
        f_t = rng.uniform(500, 2000)
        t_start = rng.uniform(0.0, (N/FS) - 0.005)
        polarity = rng.choice([-1, 1])
        t_rel = t - t_start
        transient = np.zeros_like(t)
        mask_t = t_rel >= 0
        transient[mask_t] = polarity * A_t * np.exp(-t_rel[mask_t] / tau) * np.sin(2 * np.pi * f_t * t_rel[mask_t])
        v_sig += transient

    # AWGN
    v_sig += rng.normal(0, rng.uniform(0.002, 0.010) * V1, N)
    i_sig += rng.normal(0, rng.uniform(0.002, 0.010) * I1, N)

    return v_sig, i_sig

def worker_task(args):
    idx, target_labels, seed = args
    rng = np.random.default_rng(seed)
    
    v_sig, i_sig = generate_multi_label_sample(target_labels, rng)
    X_full = dsp.extract_features(v_sig, i_sig)
    v_norm, i_norm = normalize_waveforms(v_sig, i_sig)
    
    X_wave = np.stack([v_norm, i_norm], axis=-1)
    X_mag = X_full[28:56]
    X_phase = np.concatenate([X_full[0:28], X_full[56:214], X_full[214:298]], axis=0)
    
    return idx, X_wave, X_mag, X_phase, target_labels

def get_combinations():
    """Generates all 32 valid real-world PQ combinations."""
    combos = []
    # Base states: 0=Normal, 1=Sag, 2=Swell, 3=Interruption
    for base in [0, 1, 2, 3]:
        for h in [0, 1]:      # Harmonics
            for t in [0, 1]:  # Transient
                for f in [0, 1]:  # Flicker
                    label = [0, 0, 0, 0, 0, 0, 0]
                    if base == 0:
                        if h == 0 and t == 0 and f == 0:
                            label[0] = 1  # Purely Normal
                    else:
                        label[base] = 1   # Sag, Swell, or Interruption
                    
                    label[4], label[5], label[6] = h, t, f
                    combos.append(label)
    return combos

def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Label PQ dataset.")
    parser.add_argument("--samples_per_combo", type=int, default=4000, 
                        help="Number of samples per combination (Total = args * 32).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    combos = get_combinations()
    total_samples = args.samples_per_combo * len(combos)
    
    print(f"Generating 32 unique combinations...")
    print(f"Allocating float32 memory for {total_samples} multi-label samples...")
    
    X_wave_arr = np.empty((total_samples, 500, 2), dtype=np.float32)
    X_mag_arr  = np.empty((total_samples, 28), dtype=np.float32)
    X_phase_arr = np.empty((total_samples, 270), dtype=np.float32)
    y_arr = np.empty((total_samples, 7), dtype=np.int32) # Matrix shape (N, 7) for multi-label
    
    tasks = []
    idx = 0
    for combo in combos:
        for i in range(args.samples_per_combo):
            unique_seed = args.seed + idx
            tasks.append((idx, combo, unique_seed))
            idx += 1

    max_workers = args.workers if args.workers else os.cpu_count()
    print(f"Firing up multiprocessing pool with {max_workers} CPU cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = max(1, total_samples // (max_workers * 4))
        results = executor.map(worker_task, tasks, chunksize=chunk_size)
        
        with tqdm(total=total_samples, desc="Generating Multi-Label Dataset") as pbar:
            for res in results:
                returned_idx, X_wave, X_mag, X_phase, labels = res
                
                X_wave_arr[returned_idx] = X_wave
                X_mag_arr[returned_idx] = X_mag
                X_phase_arr[returned_idx] = X_phase
                y_arr[returned_idx] = labels
                
                pbar.update(1)

    output_dir = os.path.join(os.getcwd(), "dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"pq_multilabel_dataset_{total_samples}.npz")
    
    np.savez_compressed(
        output_file, 
        X_wave=X_wave_arr, 
        X_mag=X_mag_arr, 
        X_phase=X_phase_arr, 
        y=y_arr  # Now saves the full one-hot matrix
    )
    
    print(f"\nMulti-Label Dataset saved successfully to: {output_file}")
    print(f"X_wave shape:  {X_wave_arr.shape}")
    print(f"y shape:       {y_arr.shape}  <-- Notice the (, 7) shape")

if __name__ == "__main__":
    main()