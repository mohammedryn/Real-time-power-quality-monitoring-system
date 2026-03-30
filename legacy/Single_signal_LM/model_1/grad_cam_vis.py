import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os
from data_pipeline import PQMDataGenerator 

def generate_gradcam_heatmap(sample_inputs, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a specific convolutional layer."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(sample_inputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def plot_gradcam_overlay(original_img, heatmap, title="Grad-CAM", save_path=None):
    """Overlays the heatmap onto the original image and optionally saves it."""
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap) / 255.0

    if len(original_img.shape) == 2 or original_img.shape[-1] == 1:
        original_img = np.squeeze(original_img)
        original_img = np.stack((original_img,)*3, axis=-1)
    
    superimposed_img = jet_heatmap * 0.4 + original_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 1)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Signal Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Raw Activation Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    MODEL_PATH = 'Single_signal_LM/trained_pqm_model.keras'
    DATASET_PATH = 'Single_signal_LM/dataset/train'
    METADATA_PATH = f'{DATASET_PATH}/metadata.json'
    CLASS_NAMES = ['Normal', 'Sag', 'Swell', 'Harmonics', 'Interruption', 'Transient', 'Notch']
    
    # The exact name of the layer you want to inspect (must match your model architecture)
    TARGET_LAYER = 'Voltage_DWT_last_conv' 

    # --- 2. LOAD RESOURCES ---
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Loading metadata...")
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # For visualization, we can just grab a few samples from the validation logic
    sample_ids = metadata['sample_ids']
    labels_dict = metadata['labels_dict']

    print("Initializing Data Generator...")
    data_gen = PQMDataGenerator(
        sample_ids=sample_ids, 
        labels_dict=labels_dict, 
        data_dir=DATASET_PATH,
        batch_size=1, # Batch size of 1 makes isolating a single sample very easy
        n_classes=7,
        shuffle=True # Shuffle to get a random anomaly each time you run the script
    )

    # --- 3. EXTRACT A SAMPLE & PREDICT ---
    # Grab one batch (which is exactly 1 sample because batch_size=1)
    sample_inputs, sample_label = data_gen[0]
    
    true_class_idx = np.argmax(sample_label[0])
    true_class_name = CLASS_NAMES[true_class_idx]
    
    # Get the model's prediction
    predictions = model.predict(sample_inputs, verbose=0)
    pred_class_idx = np.argmax(predictions[0])
    pred_class_name = CLASS_NAMES[pred_class_idx]
    
    print(f"\nTarget Layer: {TARGET_LAYER}")
    print(f"True Label: {true_class_name}")
    print(f"Predicted Label: {pred_class_name}")

    # --- 4. GENERATE & PLOT GRAD-CAM ---
    print("Generating Grad-CAM heatmap...")
    heatmap = generate_gradcam_heatmap(sample_inputs, model, TARGET_LAYER, pred_index=pred_class_idx)
    
    if "DWT" in TARGET_LAYER:
        image_to_plot = sample_inputs["Voltage_DWT"][0] if "Voltage" in TARGET_LAYER else sample_inputs["Current_DWT"][0]
    elif "FFT" in TARGET_LAYER:
        image_to_plot = sample_inputs["Voltage_FFT"][0] if "Voltage" in TARGET_LAYER else sample_inputs["Current_FFT"][0]
    else:
        raise ValueError("Target layer must contain 'FFT' or 'DWT'...")

    # Remove the channel dimension for plotting (e.g., 128x128x1 -> 128x128)
    image_to_plot = np.squeeze(image_to_plot)

    # Plot and save
    os.makedirs('Single_signal_LM/Results', exist_ok=True)
    save_filename = f"Single_signal_LM/Results/gradcam_{true_class_name}_{TARGET_LAYER}.png"
    
    plot_gradcam_overlay(
        image_to_plot, 
        heatmap, 
        title=f"Grad-CAM ({TARGET_LAYER})\nTrue: {true_class_name} | Pred: {pred_class_name}",
        save_path=save_filename
    )