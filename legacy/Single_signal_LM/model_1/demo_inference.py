import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import pywt
import os

# Suppress some TensorFlow terminal spam for a cleaner demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# --- 1. SIGNAL PROCESSING FUNCTIONS (With RMS Fix) ---
def format_tensor_for_cnn(data_matrix):
    val_min = np.min(data_matrix)
    val_max = np.max(data_matrix)
    normalised = (data_matrix - val_min) / (val_max - val_min + 1e-10)
    normalised = np.clip(normalised, 0, 1)
    img = resize(normalised, (128, 128), anti_aliasing=True)
    tensor = np.expand_dims(img.astype(np.float32), axis=-1)
    return tensor

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
    mask = (freqs >= 0) & (freqs <= 300)
    return format_tensor_for_cnn(db_matrix[mask, :])

def get_swt_scalogram(signal, wavelet='db10', level=6):
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(new_coeffs, wavelet, mode='per')

    coeffs_clean = pywt.wavedec(denoised_signal, wavelet, level=level)
    full_rec = []
    for i in range(len(coeffs_clean)):
        res = pywt.upcoef('d' if i > 0 else 'a', coeffs_clean[i], wavelet, 
                          level=level if i == 0 else level-i+1, take=len(denoised_signal))
        full_rec.append(res)
    scalogram = np.abs(np.array(full_rec))
    return format_tensor_for_cnn(20 * np.log10(scalogram + 1e-10))

def get_phase_features(voltage_sig, current_sig, fs=6000):
    fft_v = np.fft.rfft(voltage_sig)
    fft_i = np.fft.rfft(current_sig)
    freqs = np.fft.rfftfreq(len(voltage_sig), d=1/fs)
    
    target_harmonics = [50, 150, 250, 350, 450, 550, 650, 750, 850]
    features = []
    for target_f in target_harmonics:
        idx = np.argmin(np.abs(freqs - target_f))
        diff = np.angle(fft_v[idx]) - np.angle(fft_i[idx])
        features.append(np.angle(np.exp(1j * diff)))
        
    rms_voltage = np.sqrt(np.mean(voltage_sig**2))
    features.append(rms_voltage)
    return np.array(features[:10], dtype=np.float32)

# --- 2. LIVE SIGNAL GENERATOR ---
def generate_live_signal(fs=6000, force_label=None):
    t = np.linspace(0, 1, fs, endpoint=False)
    # Pick a random anomaly if one isn't forced
    label = np.random.randint(0, 7) if force_label is None else force_label
    
    v_sig = 5 * np.sin(2 * np.pi * 50 * t)
    i_sig = 4 * np.sin(2 * np.pi * 50 * t - 0.5)

    if label == 1:   # Sag
        mask = (t >= 0.2) & (t <= 0.6)
        v_sig[mask] *= 0.3
    elif label == 2: # Swell
        mask = (t >= 0.2) & (t <= 0.6)
        v_sig[mask] *= 1.5
    elif label == 3: # Harmonics
        v_sig += 0.5 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 250 * t)
        i_sig += 0.8 * np.sin(2 * np.pi * 150 * t) + 0.4 * np.sin(2 * np.pi * 250 * t)
    elif label == 4: # Interruption
        mask = (t >= 0.3) & (t <= 0.5)
        v_sig[mask] *= 0.02
        i_sig[mask] *= 0.02
    elif label == 5: # Transient
        transient = 2.5 * np.exp(-40 * (t - 0.4)) * np.sin(2 * np.pi * 800 * (t - 0.4))
        transient[t < 0.4] = 0 
        v_sig += transient
    elif label == 6: # Notch
        for notch_time in np.arange(0.1, 0.9, 1/150): 
            v_sig -= 2.0 * np.exp(-((t - notch_time) / 0.0005)**2)

    v_sig += np.random.normal(0, 0.05, len(t))
    i_sig += np.random.normal(0, 0.05, len(t))
    
    return v_sig, i_sig, t, label

# --- 3. MAIN DEMO EXECUTION ---
if __name__ == "__main__":
    CLASSES = ['Normal', 'Voltage Sag', 'Voltage Swell', 'Harmonics', 'Interruption', 'Transient', 'Voltage Notch']
    
    print("Loading AI Brain (Trained Keras Model)...")
    model = tf.keras.models.load_model('Single_signal_LM/trained_pqm_model.keras')
    
    print("Generating a random power grid anomaly...")
    # Change force_label to a number 0-6 if you want to test a specific one!
    v_sig, i_sig, t, true_label = generate_live_signal(force_label=5) 
    
    print("Processing Signal through STFT and DWT pipelines...")
    v_fft = np.expand_dims(get_fft_spectrogram(v_sig), axis=0) # Add batch dimension (1, 128, 128, 1)
    v_dwt = np.expand_dims(get_swt_scalogram(v_sig), axis=0)
    i_fft = np.expand_dims(get_fft_spectrogram(i_sig), axis=0)
    i_dwt = np.expand_dims(get_swt_scalogram(i_sig), axis=0)
    phase = np.expand_dims(get_phase_features(v_sig, i_sig), axis=0) # Add batch dimension (1, 10)
    
    print("Asking RTX 3050 for Prediction...")
    predictions = model.predict((v_fft, v_dwt, i_fft, i_dwt, phase), verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # --- 4. DISPLAY DASHBOARD ---
    plt.figure(figsize=(12, 6))
    plt.plot(t, v_sig, label='Voltage (V)', color='blue')
    plt.plot(t, i_sig, label='Current (A)', color='orange', alpha=0.8)
    
    plt.title(f"Live Power Quality Analysis\nTrue Anomaly: {CLASSES[true_label]} | AI Prediction: {CLASSES[predicted_class_idx]} ({confidence:.2f}%)", 
              fontsize=14, fontweight='bold', 
              color='green' if true_label == predicted_class_idx else 'red')
    
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    
    # Automatically save it so you don't hit the non-interactive backend error
    plt.tight_layout()
    plt.savefig('live_demo_result.png')
    print(f"\n✅ DEMO COMPLETE!")
    print(f"True Event: {CLASSES[true_label]}")
    print(f"AI Predict: {CLASSES[predicted_class_idx]} (Confidence: {confidence:.2f}%)")
    print("Check 'live_demo_result.png' to see the waveform!")