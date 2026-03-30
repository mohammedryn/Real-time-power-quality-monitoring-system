import numpy as np
import pywt
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def signal_gen(sf, label):
    t = np.linspace(0, 1, sf, endpoint=True)

    v = 6 * np.sin(2 * np.pi * 50 * t)
    i = 4 * np.sin(2 * np.pi * 50 * t - 0.5)

    if label == 0:
        pass 
    elif label == 1:
        start_t = np.random.uniform(0.1, 0.4)
        end_t = start_t + np.random.uniform(0.1, 0.4)
        sag_depth = np.random.uniform(0.1, 0.85) 
        sag_mask = (t >= start_t) & (t <= end_t)
        v[sag_mask] *= sag_depth
    elif label == 2:
        start_t = np.random.uniform(0.2, 0.5)
        end_t = start_t + np.random.uniform(0.1, 0.4)
        swell_mag = np.random.uniform(1.15, 1.8)
        swell_mask = (t >= start_t) & (t <= end_t)
        v[swell_mask] *= swell_mag
    elif label == 3:
        h3_amp = np.random.uniform(0.05, 0.15) * 5
        h5_amp = np.random.uniform(0.02, 0.10) * 5
        h7_amp = np.random.uniform(0.01, 0.05) * 5
        v += h3_amp * np.sin(2 * np.pi * 150 * t + np.random.uniform(0, 2*np.pi))
        v += h5_amp * np.sin(2 * np.pi * 250 * t + np.random.uniform(0, 2*np.pi))
        i += (h3_amp * 1.5) * np.sin(2 * np.pi * 150 * t + np.random.uniform(0, 2*np.pi))
        i += (h5_amp * 1.5) * np.sin(2 * np.pi * 250 * t + np.random.uniform(0, 2*np.pi))
        i += (h7_amp * 1.5) * np.sin(2 * np.pi * 350 * t + np.random.uniform(0, 2*np.pi))
    elif label == 4:
        start_t = np.random.uniform(0.1, 0.5)
        end_t = start_t + np.random.uniform(0.1, 0.3)
        int_mask = (t >= start_t) & (t <= end_t)
        v[int_mask] *= np.random.uniform(0.0, 0.05)
        i[int_mask] *= np.random.uniform(0.0, 0.05)
    elif label == 5:
        start_t = np.random.uniform(0.1, 0.7)
        f_transient = np.random.uniform(300, 2000)
        trans_amp = np.random.uniform(0.3, 0.8) * 5
        decay_rate = np.random.uniform(30, 80)
        transient = trans_amp * np.exp(-decay_rate * (t - start_t)) * np.sin(2 * np.pi * f_transient * (t - start_t))
        transient[t < start_t] = 0 
        v += transient
    elif label == 6:
        notch_depth = np.random.uniform(0.2, 0.6) * 5 
        notch_width = 0.0005 
        for notch_time in np.arange(0.1, 0.9, 1/150): 
            notch = notch_depth * np.exp(-((t - notch_time) / notch_width)**2)
            v -= notch

    # Add random Gaussian noise
    v_sig_p = np.mean(v ** 2)
    i_sig_p = np.mean(i ** 2)
    v_noise_p = v_sig_p / (10 ** (np.random.uniform(20, 35) / 10))
    i_noise_p = i_sig_p / (10 ** (np.random.uniform(20, 35) / 10))
    v_noise = np.random.normal(0, np.sqrt(v_noise_p), len(v))
    i_noise = np.random.normal(0, np.sqrt(i_noise_p), len(i))
    v += v_noise
    i += i_noise

    return v, i

def get_fft_features(signal, sf):
    n = len(signal)
    window = np.hanning(n)
    wn_signal = signal * window
    mag = np.abs(np.fft.rfft(wn_signal)) * (2 / n) * 2
    return mag.astype(np.float32)

def get_dwt_features(signal, wavelet='db20', level=6):
    r_coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(r_coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    n_coeffs = [r_coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in r_coeffs[1:]]
    n_signal = pywt.waverec(n_coeffs, wavelet, mode='per')

    coeffs = pywt.wavedec(n_signal, wavelet, level=level)
    features = []
    for i in range(len(coeffs)):
        is_approx = 'a' if i == 0 else 'd'
        rec_level = level if i == 0 else level - i + 1
        reconstructed = pywt.upcoef(is_approx, coeffs[i], wavelet, level=rec_level, take=len(n_signal))
        features.append(reconstructed)

    return np.array(features, dtype=np.float32)

def phase(voltage_sig, current_sig, fs=6000):
    fft_v = np.fft.rfft(voltage_sig)
    fft_i = np.fft.rfft(current_sig)
    freqs = np.fft.rfftfreq(len(voltage_sig), d=1/fs)
    
    target_harmonics = [50, 150, 250, 350, 450, 550, 650, 750, 850]
    features = []
    
    for target_f in target_harmonics:
        idx = np.argmin(np.abs(freqs - target_f))
        diff = np.angle(fft_v[idx]) - np.angle(fft_i[idx])
        wrapped_diff = np.angle(np.exp(1j * diff))
        features.append(wrapped_diff)
        
    rms_voltage = np.sqrt(np.mean(voltage_sig**2))
    features.append(rms_voltage)
        
    return np.array(features[:10], dtype=np.float32)

def file_save(args):
    index, label, sf = args

    v, i = signal_gen(sf, label)

    ch1_v_fft = get_fft_features(v, sf)
    ch2_i_fft = get_fft_features(i, sf)
    ch3_v_dwt = get_dwt_features(v)
    ch4_i_dwt = get_dwt_features(i)
    ch5_phase = phase(v, i, sf)

    folder = f"model_2/dataset/label_{label}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"sample_{index}.npz")

    np.savez_compressed(
        filename, 
        v_fft=ch1_v_fft, 
        i_fft=ch2_i_fft, 
        v_dwt=ch3_v_dwt, 
        i_dwt=ch4_i_dwt, 
        phase_rms=ch5_phase
    )
    return filename

def create_dataset(total_samples, sf=6000):
    num_classes = 7
    samples_per_class = total_samples // num_classes
    actual_total = samples_per_class * num_classes
    
    print(f"Targeting {actual_total} total samples ({samples_per_class} per class)...")
    print("Generating task list...")
    
    tasks = []
    task_idx = 0
    for label in range(num_classes):
        for _ in range(samples_per_class):
            tasks.append((task_idx, label, sf))
            task_idx += 1
            
    num_cores = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Distributing workload across {num_cores} CPU cores...")
    print("Generation in progress. Please wait...")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(executor.map(file_save, tasks))
        
    print("Dataset generation complete! Files saved to the 'dataset' directory.")

if __name__ == '__main__':
    TARGET_TOTAL_SAMPLES = 7000 
    create_dataset(total_samples=TARGET_TOTAL_SAMPLES, sf=6000)