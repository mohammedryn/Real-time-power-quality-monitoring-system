import numpy as np
import tensorflow as tf
import os

class PQMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sample_ids, labels_dict, data_dir, batch_size=32, 
                 img_dim=(128, 128, 1), phase_dim=10, n_classes=7, shuffle=True):
        """
        sample_ids: List of unique filenames or IDs for your samples (e.g., ['sample_001', 'sample_002'])
        labels_dict: Dictionary mapping sample IDs to their integer label (e.g., {'sample_001': 3})
        data_dir: The main folder where your .npy files are saved.
        """
        self.sample_ids = sample_ids
        self.labels_dict = labels_dict
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.phase_dim = phase_dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end() # Shuffle data at the start

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.sample_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # 1. Select the IDs for this specific batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.sample_ids[k] for k in indexes]

        # 2. Generate the data
        X, y = self.__data_generation(batch_ids)
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch to prevent the model from memorizing the order
        self.indexes = np.arange(len(self.sample_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        # Initialization of empty arrays for the batch
        X_v_fft = np.empty((self.batch_size, *self.img_dim))
        X_v_dwt = np.empty((self.batch_size, *self.img_dim))
        X_i_fft = np.empty((self.batch_size, *self.img_dim))
        X_i_dwt = np.empty((self.batch_size, *self.img_dim))
        X_phase = np.empty((self.batch_size, self.phase_dim))
        
        y = np.empty((self.batch_size), dtype=int)

        # Load data for each ID in the batch
        for i, ID in enumerate(batch_ids):
            file_path = os.path.join(self.data_dir, f"{ID}.npz")
            with np.load(file_path) as data:
                X_v_fft[i,] = data['v_fft']
                X_v_dwt[i,] = data['v_dwt']
                X_i_fft[i,] = data['i_fft']
                X_i_dwt[i,] = data['i_dwt']
                X_phase[i,] = data['phase']

            # Store the label
            y[i] = self.labels_dict[ID]

        # Convert labels to categorical (One-Hot Encoding)
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        # Return the list of 5 inputs exactly as the Keras model expects them
        X_dict = {
            "Voltage_FFT": X_v_fft,
            "Voltage_DWT": X_v_dwt,
            "Current_FFT": X_i_fft,
            "Current_DWT": X_i_dwt, 
            "Phase_difference": X_phase
        }
        return X_dict, y_categorical