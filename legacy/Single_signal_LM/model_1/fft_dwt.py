import numpy as np
import os
import json
from skimage.transform import resize
import pywt
import concurrent.futures
import time

# --- 1. CNN TENSOR FORMATTER ---
def format_tensor_for_cnn(data_matrix):
    """Normalizes, resizes to 128x128, and adds the Keras channel dimension."""
    val_min = np.min(data_matrix)
    val_max = np.max(data_matrix)
    
    # Normalize to 0-1
    normalised = (data_matrix - val_min) / (val_max - val_min + 1e-10)
    normalised = np.clip(normalised, 0, 1)

    # Resize to 128x128
    img = resize(normalised, (128, 128), anti_aliasing=True)
    
    # Cast to float32 for GPU and add channel dim (128, 128, 1)
    tensor = np.expand_dims(img.astype(np.float32), axis=-1)
    return tensor

# --- 2. SIGNAL PROCESSING FUNCTIONS ---
def get_fft_spectrogram(signal, sf=6000, window_size=1800, overlap=1500):
    step_size = window_size - overlap
    mags = []

    for start in range(0, len(signal) - window_size, step_size):
        end = start + window_size
        chunk = signal[start:end]
        window = np.hanning(len(chunk))
        wn_signal = window * chunk

        freqs = np.fft.rfftfreq(len(chunk), 1/sf)
        mag = np.abs(np.fft.rfft(wn_signal)) * (2 / len(chunk)) * 2
        mags.append(mag)

    matrix = np.array(mags).T
    db_matrix = 20 * np.log10(matrix + 1e-10)

    # Filter frequencies (0 to 300Hz)
    mask = (freqs >= 0) & (freqs <= 300)
    cnn_data = db_matrix[mask, :]
    
    return format_tensor_for_cnn(cnn_data)

def get_swt_scalogram(signal, wavelet='db10', level=6):
    # Denoise
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')

    # Scalogram
    coeffs_clean = pywt.wavedec(denoised_signal, wavelet, level=level)
    full_rec = []
    for i in range(len(coeffs_clean)):
        res = pywt.upcoef('d' if i > 0 else 'a', coeffs_clean[i], wavelet, 
                          level=level if i == 0 else level-i+1, take=len(denoised_signal))
        full_rec.append(res)
    
    scalogram = np.abs(np.array(full_rec))
    db_scalogram = 20 * np.log10(scalogram + 1e-10)
    
    return format_tensor_for_cnn(db_scalogram)

def get_phase_features(voltage_sig, current_sig, fs=6000):
    fft_v = np.fft.rfft(voltage_sig)
    fft_i = np.fft.rfft(current_sig)
    freqs = np.fft.rfftfreq(len(voltage_sig), d=1/fs)
    
    # We drop the 950Hz harmonic to make room for the RMS Voltage
    target_harmonics = [50, 150, 250, 350, 450, 550, 650, 750, 850]
    features = []
    
    # Calculate the phase difference for the first 9 slots
    for target_f in target_harmonics:
        idx = np.argmin(np.abs(freqs - target_f))
        diff = np.angle(fft_v[idx]) - np.angle(fft_i[idx])
        wrapped_diff = np.angle(np.exp(1j * diff))
        features.append(wrapped_diff)
        
    # Slot 10: Inject the RMS Voltage (The Sag/Swell Tiebreaker!)
    rms_voltage = np.sqrt(np.mean(voltage_sig**2))
    features.append(rms_voltage)
        
    return np.array(features[:10], dtype=np.float32)

# --- 3. PARALLEL WORKER FUNCTION ---
def generate_single_sample(i, save_dir, fs):
    """Worker function to generate and save a single sample."""
    
    # CRITICAL: Re-seed numpy so parallel processes don't generate identical data
    np.random.seed(int(time.time() * 1000) % 123456789 + os.getpid() + i)
    
    sample_id = f"sample_{i:04d}"
    t = np.linspace(0, 1, fs, endpoint=False)
    
    # Randomly select 1 of the 7 classes
    label = np.random.randint(0, 7)
    
    # Base Fundamental Signals (50 Hz)
    v_sig = 5 * np.sin(2 * np.pi * 50 * t)
    i_sig = 4 * np.sin(2 * np.pi * 50 * t - 0.5)

    # --- INJECT ANOMALIES ---
    if label == 0:
        pass 
    elif label == 1:
        start_t = np.random.uniform(0.1, 0.4)
        end_t = start_t + np.random.uniform(0.1, 0.4)
        sag_depth = np.random.uniform(0.1, 0.9) 
        sag_mask = (t >= start_t) & (t <= end_t)
        v_sig[sag_mask] *= sag_depth
    elif label == 2:
        start_t = np.random.uniform(0.2, 0.5)
        end_t = start_t + np.random.uniform(0.1, 0.4)
        swell_mag = np.random.uniform(1.1, 1.8)
        swell_mask = (t >= start_t) & (t <= end_t)
        v_sig[swell_mask] *= swell_mag
    elif label == 3:
        h3_amp = np.random.uniform(0.05, 0.15) * 5
        h5_amp = np.random.uniform(0.02, 0.10) * 5
        h7_amp = np.random.uniform(0.01, 0.05) * 5
        v_sig += h3_amp * np.sin(2 * np.pi * 150 * t + np.random.uniform(0, 2*np.pi))
        v_sig += h5_amp * np.sin(2 * np.pi * 250 * t + np.random.uniform(0, 2*np.pi))
        i_sig += (h3_amp * 1.5) * np.sin(2 * np.pi * 150 * t + np.random.uniform(0, 2*np.pi))
        i_sig += (h5_amp * 1.5) * np.sin(2 * np.pi * 250 * t + np.random.uniform(0, 2*np.pi))
        i_sig += (h7_amp * 1.5) * np.sin(2 * np.pi * 350 * t + np.random.uniform(0, 2*np.pi))
    elif label == 4:
        start_t = np.random.uniform(0.1, 0.5)
        end_t = start_t + np.random.uniform(0.1, 0.3)
        int_mask = (t >= start_t) & (t <= end_t)
        v_sig[int_mask] *= np.random.uniform(0.0, 0.05)
        i_sig[int_mask] *= np.random.uniform(0.0, 0.05)
    elif label == 5:
        start_t = np.random.uniform(0.1, 0.7)
        f_transient = np.random.uniform(300, 2000)
        trans_amp = np.random.uniform(0.3, 0.8) * 5
        decay_rate = np.random.uniform(30, 80)
        transient = trans_amp * np.exp(-decay_rate * (t - start_t)) * np.sin(2 * np.pi * f_transient * (t - start_t))
        transient[t < start_t] = 0 
        v_sig += transient
    elif label == 6:
        notch_depth = np.random.uniform(0.2, 0.6) * 5 
        notch_width = 0.0005 
        for notch_time in np.arange(0.1, 0.9, 1/150): 
            notch = notch_depth * np.exp(-((t - notch_time) / notch_width)**2)
            v_sig -= notch

    #Add a gausian noise at random to all signals
    v_sig_p = np.mean(v_sig ** 2)
    i_sig_p = np.mean(i_sig ** 2)
    v_noise_p = v_sig_p / (10 ** (np.random.uniform(20, 35) / 10))
    i_noise_p = i_sig_p / (10 ** (np.random.uniform(20, 35) / 10))
    v_noise = np.random.normal(0, np.sqrt(v_noise_p), len(v_sig))
    i_noise = np.random.normal(0, np.sqrt(i_noise_p), len(i_sig))
    v_sig += v_noise
    i_sig += i_noise

    # --- PROCESS SIGNALS THROUGH PIPELINE ---
    v_fft = get_fft_spectrogram(v_sig, sf=fs)
    v_dwt = get_swt_scalogram(v_sig)
    i_fft = get_fft_spectrogram(i_sig, sf=fs)
    i_dwt = get_swt_scalogram(i_sig)
    phase = get_phase_features(v_sig, i_sig, fs=fs)

    # --- SAVE TO DISK ---
    save_path = os.path.join(save_dir, f"{sample_id}.npz")
    np.savez_compressed(
        save_path,
        v_fft=v_fft,
        v_dwt=v_dwt,
        i_fft=i_fft,
        i_dwt=i_dwt,
        phase=phase
    )
    
    return sample_id, int(label)

# --- 4. DATASET GENERATION MANAGER ---
def generate_pq_dataset(num_samples=700, save_dir="Single_signal_LM/dataset/train", fs=6000, max_workers=None):
    os.makedirs(save_dir, exist_ok=True)
    sample_ids = []
    labels_dict = {}

    print(f"Generating {num_samples} mathematical power signals in '{save_dir}' using parallel processing...")

    # Using ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(generate_single_sample, i, save_dir, fs): i for i in range(num_samples)}
        
        # Process results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            sample_id, label = future.result()
            sample_ids.append(sample_id)
            labels_dict[sample_id] = label
            
            completed_count += 1
            if completed_count % 50 == 0:
                print(f"Processed {completed_count}/{num_samples} samples...")

    # Sort sample IDs before saving to metadata to keep the JSON organized
    sample_ids.sort()

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({"sample_ids": sample_ids, "labels_dict": labels_dict}, f)

    print("Real dataset generation complete!")

if __name__ == "__main__":
    # Enter the required amount of samples
    generate_pq_dataset(num_samples=10500, max_workers=None)