import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report
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

class_names = [CLASS_MAPPING[i] for i in range(7)]

def evaluate_multilabel_model(data_path, model_path):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X_wave = data['X_wave']
    X_mag = data['X_mag']
    X_phase = data['X_phase']
    y = data['y']
    
    # 1. Recreate the EXACT same validation set (10% split)
    print("Recreating validation set...")
    _, X_wave_val, _, X_mag_val, _, X_phase_val, _, y_val = train_test_split(
        X_wave, X_mag, X_phase, y, 
        test_size=0.1, 
        random_state=42
    )
    
    X_val_packed = [X_wave_val, X_mag_val, X_phase_val]
    
    # 2. Load the best saved model
    print(f"Loading best model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Generate Predictions
    print("Generating predictions on validation data...")
    prediction_probs = model.predict(X_val_packed)
    
    # 4. Apply Thresholding (0.5)
    # Because we use sigmoid, we get probabilities. Anything > 50% confidence is a '1'.
    thresholds = np.array([0.5, 0.5, 0.35, 0.5, 0.5, 0.35, 0.5])
    y_pred = (prediction_probs >= thresholds).astype(int)
    
    # 5. Print Classification Report
    print("\n" + "="*50)
    print("      MULTI-LABEL CLASSIFICATION REPORT")
    print("="*50)
    # scikit-learn natively supports 2D multi-label arrays for this report
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    
    # 6. Calculate Multi-Label Confusion Matrices
    # This returns an array of shape (7, 2, 2)
    mcm = multilabel_confusion_matrix(y_val, y_pred)
    
    # 7. Plot the 7 Matrices in a Grid
    print("\nPlotting confusion matrices...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (matrix, class_name) in enumerate(zip(mcm, class_names)):
        ax = axes[i]
        
        # Format the 2x2 matrix into a heatmap
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: NO', 'Pred: YES'],
                    yticklabels=['Actual: NO', 'Actual: YES'],
                    cbar=False)
        
        ax.set_title(f"Class: {class_name}", fontsize=12, fontweight='bold', pad=10)
    
    # Turn off the empty 8th subplot (since we only have 7 classes)
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig('multilabel_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved plot to 'multilabel_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    # Update these paths!
    DATASET_PATH = "/home/japheth/Projects/MiniProject/Single_signal_LM/model_4/dataset/pq_multilabel_dataset_128000.npz"
    MODEL_PATH = "best_pq_multilabel_model.keras" 
    
    evaluate_multilabel_model(DATASET_PATH, MODEL_PATH)