import numpy as np
import tensorflow as tf
import os
from cnn import get_fft_features, get_dwt_features, phase, signal_gen

def predict_single_event(voltage_sig, current_sig, sf=6000, model_path='best_pq_model.keras'):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    v_fft = get_fft_features(voltage_sig, sf)
    i_fft = get_fft_features(current_sig, sf)
    v_dwt = get_dwt_features(voltage_sig)
    i_dwt = get_dwt_features(current_sig)
    phase_rms = phase(voltage_sig, current_sig, sf)

    v_fft_in = np.expand_dims(np.expand_dims(v_fft, axis=-1), axis=0)
    i_fft_in = np.expand_dims(np.expand_dims(i_fft, axis=-1), axis=0)
    
    v_dwt_in = np.expand_dims(np.transpose(v_dwt), axis=0)
    i_dwt_in = np.expand_dims(np.transpose(i_dwt), axis=0)
    
    phase_in = np.expand_dims(phase_rms, axis=0)

    print("Running prediction...")
    prediction_probs = model.predict(
        [v_fft_in, v_dwt_in, i_fft_in, i_dwt_in, phase_in], 
        verbose=0
    )
    
    class_names = [
        'Normal (0)', 'Sag (1)', 'Swell (2)', 'Harmonics (3)', 
        'Interruption (4)', 'Transient (5)', 'Notch (6)'
    ]
    
    predicted_index = np.argmax(prediction_probs[0])
    confidence = np.max(prediction_probs[0]) * 100
    predicted_label = class_names[predicted_index]
    
    print("\n" + "="*40)
    print(f"⚡ DETECTED EVENT: {predicted_label}")
    print(f"🎯 CONFIDENCE:   {confidence:.2f}%")
    print("="*40 + "\n")
    
    return predicted_index, confidence

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'best_pq_model.keras')
    
    if not os.path.exists(model_file):
        print(f"Error: Could not find {model_file}. Has the training finished?")
    else:
        print("Generating a test 'Voltage Sag' signal...")
        test_v, test_i = signal_gen(sf=6000, label=1)
        
        predict_single_event(test_v, test_i, sf=6000, model_path=model_file)