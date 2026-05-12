#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.io.serial_receiver import SerialFrameReceiver


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe ESP32-P4 raw PQ frames")
    parser.add_argument("--port", required=True)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    receiver = SerialFrameReceiver(args.port, mode="raw", timeout=1.0)
    receiver.open()
    try:
        for _ in range(args.frames):
            frame = receiver.read_frame(frame_timeout=2.0)
            if frame is None:
                print("frame=None")
                continue
            print(
                f"seq={frame.seq} "
                f"V[min={frame.v_raw.min()} max={frame.v_raw.max()} mean={frame.v_raw.mean():.1f}] "
                f"I[min={frame.i_raw.min()} max={frame.i_raw.max()} mean={frame.i_raw.mean():.1f}]"
            )
    finally:
        receiver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
