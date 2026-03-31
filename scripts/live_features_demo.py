import os
import sys
import time
import argparse
import numpy as np

# Bind local module paths flawlessly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.serial_receiver import SerialFrameReceiver
from src.dsp.preprocess import preprocess_frame
from src.dsp.features import extract_features

# ANSI Colour code bindings
GREEN = '\03.033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def main():
    parser = argparse.ArgumentParser(description="Real-Time PQ DSP Processing Demonstration")
    parser.add_argument("--port", required=True, help="Serial port of the Teensy 4.1 (e.g., /dev/ttyACM0)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    print(f"\n{BOLD}⚡ Power Quality AI Data Ingestion Engine ⚡{RESET}")
    print(f"Connecting to hardware on: {CYAN}{args.port}{RESET}")

    receiver = SerialFrameReceiver(args.port)

    frames_processed = 0
    t_start = time.time()

    print(f"\n{GREEN}[*] DSP Pipeline initialized. Listening for USB Frames...{RESET}\n")

    try:
        while True:
            frame = receiver.get_frame(timeout=0.1)
            if frame is None:
                continue
            
            # 1. Physics Preprocessing
            try:
                processed = preprocess_frame(frame, args.config)
            except Exception as e:
                print(f"Preprocess Error: {e}")
                continue

            v_phys = processed['v_phys']
            i_phys = processed['i_phys']

            # 2. Extract ML Features (The heavy lifting)
            t0 = time.time()
            feature_vector = extract_features(v_phys, i_phys)
            t_ms = (time.time() - t0) * 1000

            frames_processed += 1
            
            # --- Print Dynamic Ticker ---
            # Clear previous lines for a dashboard effect (moves cursor up)
            if frames_processed > 1:
                sys.stdout.write("\033[F" * 6)
            
            print(f"--------------------------------------------------")
            print(f"{BOLD}Data Frame #{frame.seq}{RESET}  (500 Samples @ 5kHz)")
            print(f"DSP Latency:       {t_ms:.2f} ms")
            print(f"Vector Shape:      {CYAN}({len(feature_vector)},){RESET} Elements -> [ {feature_vector[0]:.2f}, {feature_vector[1]:.2f}, {feature_vector[2]:.2f} ... ]")
            
            # Peek at Key Power Quality Variables derived from the vector:
            v_rms = feature_vector[2]
            i_rms = feature_vector[12+2]
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
