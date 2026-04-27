import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Import your model builder from your model.py file
from model import build_model

def plot_training_curves(history):
    """Plots and saves the training and validation metrics."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("Saved training curves to 'training_curves.png'")
    plt.show()

def train_model(data_path):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    X_wave = data['X_wave']
    X_mag = data['X_mag']
    X_phase = data['X_phase']
    y = data['y']
    
    print(f"Total samples loaded: {len(y)}")
    
    # 1. Split the data
    print("Splitting data into train and validation sets...")
    (X_wave_train, X_wave_val, 
     X_mag_train, X_mag_val, 
     X_phase_train, X_phase_val, 
     y_train, y_val) = train_test_split(
        X_wave, X_mag, X_phase, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y 
    )

    X_train_packed = [X_wave_train, X_mag_train, X_phase_train]
    X_val_packed = [X_wave_val, X_mag_val, X_phase_val]

    # 2. Calculate Dynamic Class Weights
    print("Calculating class weights...")
    unique_classes = np.unique(y_train)
    weights_array = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_train
    )
    
    # Keras requires class weights to be passed as a dictionary
    class_weight_dict = dict(zip(unique_classes, weights_array))
    class_weight_dict[5] = class_weight_dict[5] * 1.6
    class_weight_dict[2] = class_weight_dict[2] * 2.5
    class_weight_dict[1] = class_weight_dict[1] * 1.2
    
    
    print("Using Class Weights:")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.4f}")

    # 3. Initialize Model with a slightly lower learning rate (for stability)
    print("Building model architecture...")
    model = build_model()
    
    # 4. Set up Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=12, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath='best_pq_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,      
        patience=4,      
        min_lr=1e-6,     
        verbose=1
    )

    # 5. Train the Model with class weights
    print("Starting training...")
    history = model.fit(
        x=X_train_packed,
        y=y_train,
        validation_data=(X_val_packed, y_val),
        batch_size=128,          # Increased batch size for smoother gradients
        epochs=100, 
        class_weight=class_weight_dict,  # <--- Weights injected here
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    
    print("\nTraining complete! Best model saved as 'best_pq_model.keras'")
    return history, model

if __name__ == "__main__":
    DATASET_PATH = "/home/japheth/Projects/MiniProject/Single_signal_LM/model_3/dataset/pq_dataset_35000.npz" # Update this path!
    
    history, trained_model = train_model(DATASET_PATH)
    plot_training_curves(history)