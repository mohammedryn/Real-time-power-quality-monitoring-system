import os
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures

# Import your custom modules
import dsp
import signal_gen

def normalize_waveforms(v_wave: np.ndarray, i_wave: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales physical waveforms to [-1, 1] and strictly casts to float32.
    Ensures absolute compatibility with 32-bit microcontrollers.
    """
    v = np.asarray(v_wave, dtype=np.float32)
    i = np.asarray(i_wave, dtype=np.float32)

    v_scale = np.float32(max(float(np.max(np.abs(v))), eps))
    i_scale = np.float32(max(float(np.max(np.abs(i))), eps))

    return (v / v_scale).astype(np.float32), (i / i_scale).astype(np.float32)

def worker_task(args):
    """
    Isolated worker function for multiprocessing. 
    Generates one sample and extracts its features.
    """
    idx, class_id, seed = args
    
    # Each process MUST have its own independent random generator
    rng = np.random.default_rng(seed)
    
    # 1. Generate Physical Signals
    v_sig, i_sig = signal_gen.generate_sample(class_id, rng)
    
    # 2. Extract DSP Features (286 elements)
    X_full = dsp.extract_features(v_sig, i_sig)
    
    # 3. Normalize for CNN/LSTM (Float32 strict)
    v_norm, i_norm = normalize_waveforms(v_sig, i_sig)
    
    # 4. Slice into network branches
    X_wave = np.stack([v_norm, i_norm], axis=-1)
    X_mag = X_full[28:56]
    X_phase = np.concatenate([
        X_full[0:28],      # Time & Power stats
        X_full[56:214],    # Phase features
        X_full[214:298]    # Wavelet features
    ], axis=0)
    
    return idx, X_wave, X_mag, X_phase, class_id

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PQ dataset in parallel.")
    parser.add_argument("--samples_per_class", type=int, default=1000, 
                        help="Number of samples to generate per PQ class.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed for reproducibility.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU cores to use. Defaults to all available.")
    args = parser.parse_args()

    # Define dimensions
    num_classes = 7
    total_samples = args.samples_per_class * num_classes
    
    # Pre-allocate large contiguous memory blocks for massive speedups
    print(f"Allocating float32 memory for {total_samples} samples...")
    X_wave_arr = np.empty((total_samples, 500, 2), dtype=np.float32)
    X_mag_arr  = np.empty((total_samples, 28), dtype=np.float32)
    X_phase_arr = np.empty((total_samples, 270), dtype=np.float32)
    y_arr = np.empty((total_samples,), dtype=np.int32)
    
    # Build the task list
    # Seed incorporates both class and index to guarantee unique data streams
    tasks = []
    idx = 0
    for class_id in range(num_classes):
        for i in range(args.samples_per_class):
            unique_seed = args.seed + (class_id * 1000000) + i
            tasks.append((idx, class_id, unique_seed))
            idx += 1

    # Determine CPU core count
    max_workers = args.workers if args.workers else os.cpu_count()
    print(f"Firing up multiprocessing pool with {max_workers} CPU cores...")

    # Execute generation in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # chunksize groups tasks to reduce communication overhead between processes
        chunk_size = max(1, total_samples // (max_workers * 4))
        
        results = executor.map(worker_task, tasks, chunksize=chunk_size)
        
        with tqdm(total=total_samples, desc="Generating Dataset") as pbar:
            for res in results:
                returned_idx, X_wave, X_mag, X_phase, class_id = res
                
                # Directly map results into pre-allocated memory
                X_wave_arr[returned_idx] = X_wave
                X_mag_arr[returned_idx] = X_mag
                X_phase_arr[returned_idx] = X_phase
                y_arr[returned_idx] = class_id
                
                pbar.update(1)

    print("\nGeneration complete. Packing and compressing to disk...")
    
    # Create dataset directory
    output_dir = os.path.join(os.getcwd(), "dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as compressed NumPy archive
    output_file = os.path.join(output_dir, f"pq_dataset_{total_samples}.npz")
    np.savez_compressed(
        output_file, 
        X_wave=X_wave_arr, 
        X_mag=X_mag_arr, 
        X_phase=X_phase_arr, 
        y=y_arr
    )
    
    print(f"\nDataset saved successfully to: {output_file}")
    print(f"X_wave shape:  {X_wave_arr.shape}")
    print(f"X_mag shape:   {X_mag_arr.shape}")
    print(f"X_phase shape: {X_phase_arr.shape}")
    print(f"y shape:       {y_arr.shape}")

if __name__ == "__main__":
    main()