import os
import argparse
import numpy as np
import tensorflow as tf
import time

# Import your custom modules
import signal_gen
import dsp
from data_gen import normalize_waveforms

# Class mapping for readable outputs
CLASS_NAMES = {
    0: "Normal",
    1: "Sag",
    2: "Swell",
    3: "Interruption",
    4: "Harmonic Distortion",
    5: "Transient",
    6: "Flicker"
}

def main():
    parser = argparse.ArgumentParser(description="Run fast inference on a single generated PQ sample.")
    parser.add_argument("--model", type=str, default="model_3/pqm_model.keras", help="Path to the trained model.")
    parser.add_argument("--target_class", type=int, default=4, choices=range(7), 
                        help="The actual class ID to generate and test (0-6).")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    # 1. Load the model
    print(f"Loading model '{args.model}'...")
    model = tf.keras.models.load_model(args.model)

    # 2. CREATE COMPILED PREDICTION FUNCTION (The Speed Secret)
    # This freezes the model into a C++ graph for 3ms latency
    @tf.function(reduce_retracing=True)
    def fast_predict(w, m, p):
        return model([w, m, p], training=False)

    # 3. Generate a live sample
    print(f"\nGenerating live test sample for class: {CLASS_NAMES[args.target_class]}...")
    rng = np.random.default_rng()
    v_sig, i_sig = signal_gen.generate_sample(args.target_class, rng)

    # 4. Extract and normalize features
    X_full = dsp.extract_features(v_sig, i_sig)
    v_norm, i_norm = normalize_waveforms(v_sig, i_sig)

    # 5. Format and convert to Tensors (Optimized for GPU)
    X_wave = tf.convert_to_tensor(np.stack([v_norm, i_norm], axis=-1)[np.newaxis, ...], dtype=tf.float32)
    X_mag  = tf.convert_to_tensor(X_full[28:56][np.newaxis, ...], dtype=tf.float32)
    
    # Ensure this concatenate matches your latest feature count (e.g., 266)
    X_phase_raw = np.concatenate([X_full[0:28], X_full[56:214], X_full[214:]], axis=0)
    X_phase = tf.convert_to_tensor(X_phase_raw[np.newaxis, ...], dtype=tf.float32)

    # 6. Run Inference & Measure Time
    print("Running optimized inference...")
    
    # Initial "Warm-up" prediction (First run is always slow)
    _ = fast_predict(X_wave, X_mag, X_phase)

    # Actual timed prediction
    start_time = time.perf_counter()
    predictions = fast_predict(X_wave, X_mag, X_phase)
    end_time = time.perf_counter()
    
    inference_time_ms = (end_time - start_time) * 1000
    
    # 7. Parse Results (Convert back to numpy for indexing)
    preds_np = predictions.numpy()[0]
    pred_class_id = np.argmax(preds_np)
    confidence = preds_np[pred_class_id] * 100

    print("\n" + "="*35)
    print("      INFERENCE RESULTS")
    print("="*35)
    print(f"Actual Input    : {CLASS_NAMES[args.target_class]}")
    print(f"Predicted Class : {CLASS_NAMES[pred_class_id]}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"Inference Time  : {inference_time_ms:.2f} ms") # <--- Added this
    print("-" * 35)
    
    # Show probabilities for all classes
    for i, prob in enumerate(preds_np):
        indicator = " <-- PREDICTED" if i == pred_class_id else ""
        print(f"  {CLASS_NAMES[i]:<20}: {prob*100:>6.2f}% {indicator}")

if __name__ == "__main__":
    main()