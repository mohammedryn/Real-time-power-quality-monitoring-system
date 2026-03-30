from keras import layers, models
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import glob, math, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def build_cnn(fft_shape=(3001, 1), dwt_shape=(6000, 7), phase_shape=(10,)):

    def create_conv1d_branch(input_shape, name):
        inputs = layers.Input(shape=input_shape, name=name)

        x = layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=4)(x)

        x = layers.Bidirectional(LSTM(64, return_sequences=False, name=f"{name}_lstm"))(x)
        x = layers.Dropout(0.3)(x)

        return inputs, x
    
    v_fft_in, v_fft_feat = create_conv1d_branch(fft_shape, "Voltage_FFT")
    i_fft_in, i_fft_feat = create_conv1d_branch(fft_shape, "Current_FFT")

    def create_dwt_branch(input_shape, name):
        inputs = layers.Input(shape=input_shape, name=name)
        x = layers.Conv1D(32, kernel_size=5, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=4)(x) 

        x = layers.Conv1D(64, kernel_size=3, padding='same', use_bias=False, name=f"{name}_last_conv")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        return inputs, x

    v_dwt_in, v_dwt_feat = create_dwt_branch(dwt_shape, "Voltage_DWT")
    i_dwt_in, i_dwt_feat = create_dwt_branch(dwt_shape, "Current_DWT")

    phase_in = layers.Input(shape=phase_shape, name="Phase_difference")
    p_norm = layers.BatchNormalization(name="Phase_BatchNorm")(phase_in)
    p = layers.Dense(64, activation='relu')(p_norm)
    p = layers.BatchNormalization()(p)
    p = layers.Dense(128, activation='relu')(p)
    p = layers.Dropout(0.2)(p)
    phase_feat = layers.Dense(64, activation='relu')(p)

    merged = layers.Concatenate(name="Feature_fusion")([
        v_fft_feat, i_fft_feat,
        v_dwt_feat, i_dwt_feat,
        phase_feat
    ])

    z = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(merged)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(z)

    output = layers.Dense(7, activation='softmax', name="PQ_event")(z)

    model = models.Model(
        inputs=[v_fft_in, v_dwt_in, i_fft_in, i_dwt_in, phase_in],
        outputs=output
    )
    
    c_optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=c_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


model = build_cnn()
model.summary()

def npz_data_generator(file_paths, num_classes=7):
    for path in file_paths:
        normalized_path = os.path.normpath(path)
        label_str = [part for part in normalized_path.split(os.sep) if 'label_' in part][0]
        label = int(label_str.split('_')[1])
        label_onehot = to_categorical(label, num_classes=num_classes)
        
        data = np.load(path)
        
        v_fft = np.expand_dims(data['v_fft'], axis=-1)
        i_fft = np.expand_dims(data['i_fft'], axis=-1)
        
        v_dwt = np.transpose(data['v_dwt'])
        i_dwt = np.transpose(data['i_dwt'])
        
        phase = data['phase_rms']
        
        inputs_dict = {
            "Voltage_FFT": v_fft,
            "Voltage_DWT": v_dwt,
            "Current_FFT": i_fft,
            "Current_DWT": i_dwt,
            "Phase_difference": phase
        }
        
        yield inputs_dict, label_onehot

def create_tf_dataset(file_paths, batch_size=32, is_training=True):
    """Wraps the generator in a high-performance tf.data.Dataset."""
    dataset = tf.data.Dataset.from_generator(
        lambda: npz_data_generator(file_paths),
        output_signature=(
            {
                "Voltage_FFT": tf.TensorSpec(shape=(3001, 1), dtype=tf.float32),
                "Voltage_DWT": tf.TensorSpec(shape=(6000, 7), dtype=tf.float32),
                "Current_FFT": tf.TensorSpec(shape=(3001, 1), dtype=tf.float32),
                "Current_DWT": tf.TensorSpec(shape=(6000, 7), dtype=tf.float32),
                "Phase_difference": tf.TensorSpec(shape=(10,), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(7,), dtype=tf.float32) # The one-hot label
        )
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        
    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "label_*", "*.npz")
    all_files = glob.glob(dataset_path)
    print(f"Total samples found: {len(all_files)}")
    
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    print(f"Training samples (80%): {len(train_files)}")
    print(f"Validation samples (20%): {len(val_files)}")
    
    BATCH_SIZE = 32
    train_dataset = create_tf_dataset(train_files, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_tf_dataset(val_files, batch_size=BATCH_SIZE, is_training=False)
    
    train_steps = math.ceil(len(train_files) / BATCH_SIZE)
    val_steps = math.ceil(len(val_files) / BATCH_SIZE)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpointpath = os.path.join(script_dir, 'best_pq_model.keras')
    checkpoint = ModelCheckpoint(
        filepath=checkpointpath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    custom_weights = {
        0: 0.5,
        1: 1.5,
        2: 1.5,
        3: 1.0,
        4: 1.0,
        5: 1.5,
        6: 1.5
    }

    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stop, checkpoint],
        class_weight=custom_weights,
        verbose=1
    )

    def plot_training_history(history_dict):
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, 'b-', linewidth=2, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, 'b-', linewidth=2, label='Training Loss')
        plt.plot(epochs_range, val_loss, 'r-', linewidth=2, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('model_2/results/training_metrics.png', dpi=300)

    plot_training_history(history.history)

    def evaluate_model_performance(trained_model, validation_dataset, val_steps):
        print("\nGenerating predictions for the Confusion Matrix...")
        
        y_true = []
        y_pred = []
        
        for batch_x, batch_y in validation_dataset.take(val_steps):
            predictions = trained_model.predict(batch_x, verbose=0)
            
            y_pred.extend(np.argmax(predictions, axis=1))
            
            y_true.extend(np.argmax(batch_y.numpy(), axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        class_names = [
            'Normal (0)', 
            'Sag (1)', 
            'Swell (2)', 
            'Harmonics (3)', 
            'Interruption (4)', 
            'Transient (5)', 
            'Notch (6)'
        ]

        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar=False)
        
        plt.title('Power Quality Classification - Confusion Matrix', pad=20, fontsize=14)
        plt.ylabel('True Event', fontsize=12)
        plt.xlabel('Predicted Event', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('model_2/results/confusion_matrix.png', dpi=300)

    evaluate_model_performance(model, val_dataset, val_steps)