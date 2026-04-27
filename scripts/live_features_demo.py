import os
import sys
import time
import argparse
import numpy as np

# Bind local module paths flawlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.serial_receiver import SerialFrameReceiver
from src.dsp.preprocess import preprocess_frame, load_config
from src.dsp.features import extract_features

# ANSI Colour code bindings
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def main():
    parser = argparse.ArgumentParser(description="Real-Time PQ Feature Stream Demonstration")
    parser.add_argument("--port", required=True, help="Serial port of the Teensy 4.1 (e.g., /dev/ttyACM0)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument(
        "--receiver-mode",
        choices=["feature", "raw"],
        default="feature",
        help="feature: consume 282-feature frames from Teensy, raw: run host DSP fallback",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}⚡ Power Quality AI Data Ingestion Engine ⚡{RESET}")
    print(f"Connecting to hardware on: {CYAN}{args.port}{RESET}")

    receiver = SerialFrameReceiver(args.port, mode=args.receiver_mode)

    # Raw fallback path still needs calibration and preprocessing config.
    cfg = load_config(args.config) if args.receiver_mode == "raw" else None

    frames_processed = 0

    if args.receiver_mode == "feature":
        print(f"\n{GREEN}[*] Feature mode initialized. Listening for 282-element frames from Teensy...{RESET}\n")
    else:
        print(f"\n{GREEN}[*] Raw mode initialized. Running host-side preprocessing + feature extraction...{RESET}\n")

    try:
        while True:
            frame = receiver.read_frame(frame_timeout=0.1)
            if frame is None:
                continue

            t0 = time.time()

            if args.receiver_mode == "feature":
                # Direct low-latency path: Teensy has already extracted the full 282-feature vector.
                feature_vector = np.asarray(frame.features, dtype=np.float32)
                frame_descriptor = f"{frame.n_features} Features @ 5kHz window"
            else:
                # Fallback/debug path: compute features on host from raw ADC arrays.
                try:
                    processed = preprocess_frame(frame.v_raw, frame.i_raw, cfg)
                except Exception as e:
                    print(f"Preprocess Error: {e}")
                    continue
                feature_vector = extract_features(processed['v_phys'], processed['i_phys'])
                frame_descriptor = "500 Samples @ 5kHz"

            t_ms = (time.time() - t0) * 1000

            frames_processed += 1

            # --- Print Dynamic Ticker ---
            # Clear previous lines for a dashboard effect (moves cursor up)
            if frames_processed > 1:
                sys.stdout.write("\033[F" * 6)

            print(f"--------------------------------------------------")
            print(f"{BOLD}Data Frame #{frame.seq}{RESET}  ({frame_descriptor})")
            if args.receiver_mode == "feature":
                print(f"Host Parse Latency: {t_ms:.2f} ms")
            else:
                print(f"Host DSP Latency:   {t_ms:.2f} ms")
            print(f"Vector Shape:      {CYAN}({len(feature_vector)},){RESET} Elements -> [ {feature_vector[0]:.2f}, {feature_vector[1]:.2f}, {feature_vector[2]:.2f} ... ]")

            # Peek at Key Power Quality Variables derived from the vector:
            v_rms = feature_vector[2]
            i_rms = feature_vector[14]
            v_thd = feature_vector[24 + 13 + 13]  # THD V is located exactly at idx 50

            print(f"V_RMS: {YELLOW}{v_rms:6.2f} {RESET} | I_RMS: {CYAN}{i_rms:6.2f}{RESET} | V_THD: {YELLOW}{v_thd*100:5.2f}%{RESET}")
            print(f"--------------------------------------------------")

    except KeyboardInterrupt:
        print("\n[*] Stopping pipeline...")
    finally:
        receiver.close()
        print(f"Total Frames Passed to DSP: {frames_processed}")

if __name__ == "__main__":
    main()
