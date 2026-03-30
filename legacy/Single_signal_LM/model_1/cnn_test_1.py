import tensorflow as tf
from keras import layers, models
import json, os
from data_pipeline import PQMDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

def build_cnn(img_shape=(128, 128, 1), phase_shape=(10, )):

    def create_branch(name):
        inputs = layers.Input(shape=img_shape, name=name)

        x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        #Grad cam acces(For visulaization)
        x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name=f"{name}_last_conv")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        return inputs, x
    
    # VOLTAGE STREAM
    v_fft_in, v_fft_feat = create_branch("Voltage_FFT")
    v_dwt_in, v_dwt_feat = create_branch("Voltage_DWT")

    # CURRENT STREAM
    i_fft_in, i_fft_feat = create_branch("Current_FFT")
    i_dwt_in, i_dwt_feat = create_branch("Current_DWT")

    # PHASE DIFFERENCE STREAM
    phase_in = layers.Input(shape=phase_shape, name="Phase_difference")
    
    p_norm = layers.BatchNormalization(name="Phase_BatchNorm")(phase_in)
    p = layers.Dense(16, activation='relu')(p_norm)
    phase_feat = layers.Dense(64, activation='relu')(p)

    # MERGING ALL 5 BRANCHES
    merged = layers.Concatenate(name="Feature_fusion")([
        v_fft_feat, v_dwt_feat,
        i_fft_feat, i_dwt_feat,
        phase_feat
    ])

    z = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(merged)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(z)

    # OUTPUT LAYER
    output = layers.Dense(7, activation='softmax', name="PQ_event")(z)

    # COMPILE THE MODEL
    model = models.Model(
        inputs=[v_fft_in, v_dwt_in, i_fft_in, i_dwt_in, phase_in],
        outputs=output
    )
    
    c_optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=c_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

pqm_model = build_cnn()
pqm_model.summary()

def create_tf_dataset(keras_generator):
    
    def gen():
        while True:
            for i in range(len(keras_generator)):
                yield keras_generator[i]
            
            keras_generator.on_epoch_end()

    output_signature = (
        {
            "Voltage_FFT":tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32), # v_fft
            "Voltage_DWT":tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32), # v_dwt                
            "Current_FFT":tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32), # i_fft
            "Current_DWT":tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32), # i_dwt
            "Phase_difference":tf.TensorSpec(shape=(None, 10), dtype=tf.float32) # phase
        },
        tf.TensorSpec(shape=(None, 7), dtype=tf.float32)                # labels
    )

    # 3. Convert to a tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# 1. Load the metadata we just generated
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset", "train")
with open(f'{dataset_path}/metadata.json', 'r') as f:
    metadata = json.load(f)

my_sample_ids = metadata['sample_ids']
my_labels_dict = metadata['labels_dict']

#80-20 split
train_ids, val_ids = train_test_split(my_sample_ids, test_size=0.2, random_state=42)


# 2. Initialize the custom data generator
print("Initializing Data Generator...")
training_generator = PQMDataGenerator(
    sample_ids=train_ids, 
    labels_dict=my_labels_dict, 
    data_dir=dataset_path,
    batch_size=32, # See gpu performance memory spill and change accordingly
    n_classes=7,
    shuffle=True
)

val_generator = PQMDataGenerator(
    sample_ids=val_ids,
    labels_dict=my_labels_dict,
    data_dir=dataset_path,
    batch_size=12,
    n_classes=7,
    shuffle=False
)

y_train_integers = [np.argmax(my_labels_dict[sid]) if isinstance(my_labels_dict[sid], list) else my_labels_dict[sid] for sid in train_ids]

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_integers),
    y=y_train_integers
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Computed class weights:", class_weights_dict)

lr_sheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

train_dataset = create_tf_dataset(training_generator)
val_dataset = create_tf_dataset(val_generator)

print("Starting Training...")
history = pqm_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    steps_per_epoch=len(training_generator),
    validation_steps=len(val_generator),
    class_weight=class_weights_dict,
    callbacks=[lr_sheduler],
)
print("Pipeline Test Successful!")

# --- 4. VISUALIZE TRAINING RESULTS ---
def plot_training_history(history):
    print("Generating training performance graphs...")
    
    # Create a figure with two side-by-side graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation accuracy', color='pink', linewidth=2)
    ax1.set_title('CNN Model Accuracy over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (0.0 to 1.0)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Graph 2: Loss (Error)
    ax2.plot(history.history['loss'], label='Training Loss', color='red', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation loss', color='green', linewidth=2)
    ax2.set_title('CNN Model Loss (Error) over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Categorical Crossentropy Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("model_1/Results/Training_history.png")
    print("Graph saved in your folder")

# Call the function using the 'history' object generated by model.fit()
plot_training_history(history)

# --- 5. SAVE THE TRAINED MODEL ---
# (Uncomment this line when you do your final large training run!)
pqm_model.save('model_1/trained_pqm_model.keras')
print("Model saved to disk!")

def evaluate_model_performance(model, val_dataset):
    print("\n--- Generating Confusion Matrix ---")
    
    # 2. Ask the model to predict the classes for all signals
    print("Running predictions on the dataset...")
    predictions = model.predict(val_dataset, steps=len(val_generator))
    y_pred = np.argmax(predictions, axis=1) # Convert probabilities to class indices (0-6)
    
    # 3. Extract the exact True Labels from the generator
    y_true = []
    for batch_x, batch_y in val_dataset.take(len(val_generator)):
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))
    y_true = np.array(y_true)
    
    # Ensure dimensions match (in case of dropped incomplete batches)
    y_pred = y_pred[:len(y_true)]

    # 4. Calculate the Matrix
    class_names = ['Normal', 'Sag', 'Swell', 'Harmonics', 'Interruption', 'Transient', 'Notch']
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Plot the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Power Quality Disturbance - Confusion Matrix')
    plt.ylabel('TRUE Anomaly (What it actually was)')
    plt.xlabel('PREDICTED Anomaly (What the CNN thought it was)')
    
    # Save as image just like the training history
    plt.tight_layout()
    plt.savefig('model_1/Results/confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'!")
    
    # Print a text summary of Precision and Recall
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate_model_performance(pqm_model, val_dataset)