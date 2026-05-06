import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Import your model builder from your model.py file
from model import build_model

def plot_training_curves(history):
    """Plots and saves the training and validation metrics for multi-label training."""
    
    # Keras sometimes names the metric 'accuracy' and sometimes 'binary_accuracy' 
    # depending on the backend version. This safely checks for both.
    acc_key = 'binary_accuracy' if 'binary_accuracy' in history.history else 'accuracy'
    val_acc_key = 'val_binary_accuracy' if 'val_binary_accuracy' in history.history else 'val_accuracy'
    
    acc = history.history[acc_key]
    val_acc = history.history[val_acc_key]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    plt.title('Model Binary Accuracy (Multi-Label)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss (Binary Crossentropy)')
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
    print(f"Label shape verified: {y.shape}") # Should print (128000, 7)
    
    # 1. Split the data
    # Removed stratify=y because it doesn't work out-of-the-box for multi-label targets
    print("Splitting data into train and validation sets...")
    (X_wave_train, X_wave_val, 
     X_mag_train, X_mag_val, 
     X_phase_train, X_phase_val, 
     y_train, y_val) = train_test_split(
        X_wave, X_mag, X_phase, y, 
        test_size=0.1,  # 10% validation is plenty for 128k samples
        random_state=42 
    )

    X_train_packed = [X_wave_train, X_mag_train, X_phase_train]
    X_val_packed = [X_wave_val, X_mag_val, X_phase_val]

    # 2. Initialize Model
    print("Building model architecture...")
    model = build_model()
    
    # Force Multi-label Compilation (Just to be safe!)
    print("Compiling model for Multi-Label Classification...")
    custom_adam = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=custom_adam,
        loss='binary_crossentropy', # CRITICAL: Treats each class independently
        metrics=['binary_accuracy'] # CRITICAL: Evaluates each class independently
    )
    
    # 3. Set up Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=12, 
        restore_best_weights=True,
        verbose=1
    )
    
    # Changed monitor to 'val_loss' instead of accuracy. 
    # For multi-label, minimizing loss is a more stable save criteria than accuracy.
    checkpoint = ModelCheckpoint(
        filepath='best_pq_multilabel_model.keras',
        monitor='val_loss', 
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

    # 4. Train the Model
    print("Starting multi-label training...")
    history = model.fit(
        x=X_train_packed,
        y=y_train,
        validation_data=(X_val_packed, y_val),
        batch_size=256,          # Increased for speed and stability with 128k samples
        epochs=100, 
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    
    print("\nTraining complete! Best model saved as 'best_pq_multilabel_model.keras'")
    return history, model

if __name__ == "__main__":
    # Update this path to point to your new 128k combined dataset!
    DATASET_PATH = "/home/japheth/Projects/MiniProject/Single_signal_LM/model_4/dataset/pq_multilabel_dataset_128000.npz" 
    
    history, trained_model = train_model(DATASET_PATH)
    plot_training_curves(history)