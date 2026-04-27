import os
import argparse
import numpy as np
import tensorflow as tf
import time

# Import your custom modules
import dsp
from data_gen import generate_multi_label_sample, normalize_waveforms

CLASS_NAMES = [
    "Normal", 
    "Sag", 
    "Swell", 
    "Interruption", 
    "Harmonics", 
    "Transient", 
    "Flicker"
]

# The custom tuned thresholds for Multi-Label detection
# Tuned to be highly sensitive to Swells (Class 2) and Transients (Class 5)
CUSTOM_THRESHOLDS = [0.50, 0.50, 0.35, 0.50, 0.50, 0.35, 0.50]

def main():
    parser = argparse.ArgumentParser(description="Run Ultra-Fast Multi-Label Inference on a generated PQ sample.")
    parser.add_argument("--model", type=str, default="pqm_multilabel_model.keras", help="Path to trained model.")
    parser.add_argument("--labels", type=str, default="0,1,0,0,1,0,0", 
                        help="Comma-separated target labels. Example: 0,1,0,0,1,0,0 (Sag + Harmonics)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model '{args.model}' not found.")
        return

    # Parse target labels into an array
    try:
        target_labels = [int(x.strip()) for x in args.labels.split(',')]
        if len(target_labels) != 7:
            raise ValueError
    except ValueError:
        print("Error: --labels must be exactly 7 comma-separated integers (0s or 1s).")
        return

    # 1. Load Model
    print(f"Loading model '{args.model}'...")
    model = tf.keras.models.load_model(args.model)

    # 2. CREATE COMPILED PREDICTION FUNCTION (3ms Latency Secret)
    @tf.function(reduce_retracing=True)
    def fast_predict(w, m, p):
        return model([w, m, p], training=False)

    # 3. Generate a live complex sample
    true_active_classes = [CLASS_NAMES[i] for i, val in enumerate(target_labels) if val == 1]
    print(f"\nGenerating live test sample for combination: {' + '.join(true_active_classes)}")
    
    rng = np.random.default_rng()
    v_sig, i_sig = generate_multi_label_sample(target_labels, rng)

    # 4. Extract and normalize features
    X_full = dsp.extract_features(v_sig, i_sig)
    v_norm, i_norm = normalize_waveforms(v_sig, i_sig)

    # 5. Format and Convert to Tensors for GPU Optimization
    # Note: X_full[214:] is used to automatically grab all DWT features, adapting if you have 254 or 266 total features.
    X_wave = tf.convert_to_tensor(np.stack([v_norm, i_norm], axis=-1)[np.newaxis, ...], dtype=tf.float32)
    X_mag  = tf.convert_to_tensor(X_full[28:56][np.newaxis, ...], dtype=tf.float32)
    X_phase = tf.convert_to_tensor(np.concatenate([X_full[0:28], X_full[56:214], X_full[214:]], axis=0)[np.newaxis, ...], dtype=tf.float32)

    # 6. Run Inference & Measure Time
    print("Running optimized inference...\n")
    
    # Warm-up run to compile the C++ graph (First run is always slow)
    _ = fast_predict(X_wave, X_mag, X_phase)

    # Actual timed prediction
    start_time = time.perf_counter()
    predictions = fast_predict(X_wave, X_mag, X_phase).numpy()[0]
    end_time = time.perf_counter()
    
    inference_time_ms = (end_time - start_time) * 1000

    # 7. Parse Multi-Label Results
    print("--- Multi-Label Inference Results ---")
    print(f"{'PQ Class':<15} | {'Ground Truth':<12} | {'Thresh':<6} | {'Model Confidence':<18} | {'Predicted'}")
    print("-" * 75)

    predicted_active_classes = []

    for i in range(7):
        prob = predictions[i]
        thresh = CUSTOM_THRESHOLDS[i]
        
        is_true = target_labels[i] == 1
        is_pred = prob >= thresh
        
        if is_pred:
            predicted_active_classes.append(CLASS_NAMES[i])

        gt_text = "YES" if is_true else "--"
        pred_text = "DETECTED" if is_pred else "--"
        
        # Color formatting for terminal (Green for correct, Red for incorrect)
        if is_true == is_pred:
            color_start = "\033[92m" # Green
        else:
            color_start = "\033[91m" # Red
        color_end = "\033[0m"

        print(f"{CLASS_NAMES[i]:<15} | {gt_text:<12} | {thresh:<6.2f} | {color_start}{prob*100:>6.2f}%{color_end}             | {color_start}{pred_text}{color_end}")

    print("\n--- Summary ---")
    print(f"Input Combination : {' + '.join(true_active_classes)}")
    print(f"Model Diagnostics : {' + '.join(predicted_active_classes) if predicted_active_classes else 'None Detected'}")
    print(f"Inference Time    : {inference_time_ms:.2f} ms")

if __name__ == "__main__":
    main()