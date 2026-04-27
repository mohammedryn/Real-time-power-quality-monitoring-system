import numpy as np
from keras.optimizers import Adam
from keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, 
                                     Dense, Concatenate, Dropout, BatchNormalization)
from keras.models import Model

def build_model():
    """
    Builds a 3-branch neural network for Power Quality classification.
    Expects 3 inputs: raw waveform, magnitude features, and phase/DWT features.
    Outputs probabilities for 7 distinct power quality classes.
    """
    
    # =====================================================================
    # BRANCH 1: Raw Normalized Waveform [CNN + LSTM]
    # Input Shape: (500, 2) - [v_norm, i_norm]
    # =====================================================================
    input_wave = Input(shape=(500, 2), name='wave_input')
    
    # Feature extraction over time
    x1 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(input_wave)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    
    # Sequential learning
    x1 = LSTM(units=64, return_sequences=False)(x1)
    
    # Branch 1 specific representation
    x1 = Dense(64, activation='relu')(x1)
    branch1_out = Dropout(0.3)(x1)

    # =====================================================================
    # BRANCH 2: Magnitude Features [Dense Network]
    # Input Shape: (28,) - FFT magnitudes + THD only
    # =====================================================================
    input_mag = Input(shape=(28,), name='mag_input')
    
    x2 = Dense(64, activation='relu')(input_mag)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    
    # Branch 2 specific representation
    branch2_out = Dense(32, activation='relu')(x2)

    # =====================================================================
    # BRANCH 3: Phase + DWT + Time-domain Features [Dense Network]
    # Input Shape: (258,)
    # =====================================================================
    input_phase = Input(shape=(270,), name='phase_input')
    
    x3 = Dense(128, activation='relu')(input_phase)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)
    
    # Branch 3 specific representation
    x3 = Dense(64, activation='relu')(x3)
    branch3_out = Dropout(0.2)(x3)

    # =====================================================================
    # FINAL FUSION LEARNING & OUTPUT
    # =====================================================================
    # Combine the learned features from all three branches
    merged = Concatenate(name='fusion_layer')([branch1_out, branch2_out, branch3_out])

    # Joint learning on the fused representations
    xf = Dense(128, activation='relu')(merged)
    xf = Dropout(0.4)(xf)
    xf = Dense(64, activation='relu')(xf)
    
    # Final Output Layer: 7 Classes (0 through 6)
    output = Dense(7, activation='sigmoid', name='main_output')(xf)

    # =====================================================================
    # BUILD AND COMPILE
    # =====================================================================
    model = Model(inputs=[input_wave, input_mag, input_phase], outputs=output)
    
    optimizer = Adam(learning_rate=0.0005)

    # Compile using binary_crossentropy for multi-label classification
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model
