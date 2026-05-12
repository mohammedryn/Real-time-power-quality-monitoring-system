#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from src.io.serial_receiver import SerialFrameReceiver


def estimate_lag_samples(v: np.ndarray, i: np.ndarray) -> int:
    v0 = v.astype(np.float64) - float(np.mean(v))
    i0 = i.astype(np.float64) - float(np.mean(i))
    corr = np.correlate(v0, i0, mode="full")
    return int(np.argmax(corr) - (len(v0) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ADC pair skew from matching signals on both channels")
    parser.add_argument("--port", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fs-hz", type=float, default=5000.0)
    args = parser.parse_args()

    receiver = SerialFrameReceiver(args.port, mode="raw", timeout=1.0)
    lags: list[int] = []
    receiver.open()
    try:
        while len(lags) < args.frames:
            frame = receiver.read_frame(frame_timeout=2.0)
            if frame is None:
                continue
            lags.append(estimate_lag_samples(frame.v_raw, frame.i_raw))
    finally:
        receiver.close()

    lag_samples = np.asarray(lags, dtype=np.float64)
    lag_us = lag_samples * (1_000_000.0 / args.fs_hz)
    print(f"frames={len(lags)}")
    print(f"lag_samples_mean={lag_samples.mean():.3f}")
    print(f"lag_samples_max_abs={np.max(np.abs(lag_samples)):.3f}")
    print(f"lag_us_mean={lag_us.mean():.3f}")
    print(f"lag_us_max_abs={np.max(np.abs(lag_us)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
