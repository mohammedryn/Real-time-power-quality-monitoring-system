import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Your canonical class mapping
CLASS_MAPPING = {
    0: "Normal",
    1: "Sag",
    2: "Swell",
    3: "Interruption",
    4: "HarmonicDistortion",
    5: "Transient",
    6: "Flicker"
}

# Create an ordered list of names for the plot labels
class_names = [CLASS_MAPPING[i] for i in range(7)]

def plot_confusion_matrix(data_path, model_path):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X_wave = data['X_wave']
    X_mag = data['X_mag']
    X_phase = data['X_phase']
    y = data['y']
    
    # 1. Recreate the EXACT same validation set.
    # By using the exact same random_state=42 and stratify=y, 
    # train_test_split will give us the exact same 20% of data the model used for validation.
    print("Recreating validation set...")
    _, X_wave_val, _, X_mag_val, _, X_phase_val, _, y_val = train_test_split(
        X_wave, X_mag, X_phase, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y 
    )
    
    X_val_packed = [X_wave_val, X_mag_val, X_phase_val]
    
    # 2. Load the best saved model
    print(f"Loading best model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Generate Predictions
    print("Generating predictions on validation data...")
    prediction_probs = model.predict(X_val_packed)
    
    # Convert probabilities (e.g., [0.1, 0.8, 0.1...]) into class IDs (e.g., 1)
    y_pred = np.argmax(prediction_probs, axis=1)
    
    # 4. Calculate the Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # 5. Print a detailed text report (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # 6. Plot the Confusion Matrix
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    # annot=True puts the numbers inside the boxes
    # fmt='d' formats them as whole integers
    # cmap='Blues' uses a nice blue color gradient
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Power Quality Classification - Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Event Type', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Event Type', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save and show
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nSaved confusion matrix plot to 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    # Update these paths to match your system
    DATASET_PATH = "/home/japheth/Projects/MiniProject/Single_signal_LM/model_3/dataset/pq_dataset_35000.npz" 
    MODEL_PATH = "/home/japheth/Projects/MiniProject/Single_signal_LM/model_3/pqm_model.keras" 
    
    plot_confusion_matrix(DATASET_PATH, MODEL_PATH)