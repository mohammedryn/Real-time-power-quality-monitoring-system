from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import serial

from src.io.frame_protocol import (
    FRAME_SIZE, FEATURE_FRAME_SIZE, MODEL_READY_FRAME_SIZE, MAGIC_BYTES,
    ParsedFrame, FeatureFrame, ModelReadyFrame,
    parse_frame, parse_feature_frame, parse_model_ready_frame,
)
from src.dsp.preprocess import load_config, preprocess_frame


@dataclass
class ReceiverStats:
    accepted_frames: int = 0
    crc_failures: int = 0
    parse_failures: int = 0
    timeouts: int = 0
    reconnects: int = 0


class SerialFrameReceiver:
    def __init__(
        self,
        port: str,
        baud: int = 115200,
        timeout: float = 1.0,
        reconnect_delay: float = 0.2,
        max_reconnect_attempts: int = 3,
        mode: str = "raw",
    ) -> None:
        if mode not in ("raw", "feature", "model4"):
            raise ValueError(f"mode must be 'raw', 'feature', or 'model4', got {mode!r}")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.mode = mode
        if mode == "model4":
            self._frame_size = MODEL_READY_FRAME_SIZE
        elif mode == "feature":
            self._frame_size = FEATURE_FRAME_SIZE
        else:
            self._frame_size = FRAME_SIZE
        self.ser: serial.Serial | None = None
        self.stats = ReceiverStats()

    def open(self) -> None:
        if self.ser is not None and self.ser.is_open:
            return
        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _reconnect(self) -> bool:
        self.close()
        for _ in range(self.max_reconnect_attempts):
            try:
                self.open()
                self.stats.reconnects += 1
                return True
            except serial.SerialException:
                time.sleep(self.reconnect_delay)
        return False

    def _safe_read(self, n: int) -> bytes:
        if self.ser is None:
            if not self._reconnect():
                return b""
        if self.ser is None:
            return b""

        try:
            return self.ser.read(n)
        except (serial.SerialException, OSError):
            self._reconnect()
            return b""

    def _read_exact(self, n: int, deadline: float) -> bytes | None:
        buf = bytearray()
        while len(buf) < n and time.monotonic() < deadline:
            chunk = self._safe_read(n - len(buf))
            if not chunk:
                continue
            buf.extend(chunk)
        if len(buf) != n:
            return None
        return bytes(buf)

    def _sync_to_magic(self, deadline: float) -> bool:
        rolling = bytearray()
        while time.monotonic() < deadline:
            b = self._safe_read(1)
            if not b:
                continue
            rolling.extend(b)
            if len(rolling) > 4:
                del rolling[0]
            if bytes(rolling) == MAGIC_BYTES:
                return True
        return False

    def read_frame(
        self,
        frame_timeout: float = 1.0,
        max_crc_failures: int = 8,
    ) -> ParsedFrame | FeatureFrame | ModelReadyFrame | None:
        if self.ser is None:
            self.open()

        deadline = time.monotonic() + frame_timeout
        crc_failures_in_call = 0

        while time.monotonic() < deadline:
            if not self._sync_to_magic(deadline):
                self.stats.timeouts += 1
                return None

            remaining = self._read_exact(self._frame_size - 4, deadline)
            if remaining is None:
                self.stats.timeouts += 1
                return None

            frame_bytes = MAGIC_BYTES + remaining
            try:
                if self.mode == "model4":
                    parsed: ParsedFrame | FeatureFrame | ModelReadyFrame = parse_model_ready_frame(frame_bytes)
                elif self.mode == "feature":
                    parsed = parse_feature_frame(frame_bytes)
                else:
                    parsed = parse_frame(frame_bytes)
            except ValueError:
                self.stats.parse_failures += 1
                continue

            if parsed.crc_ok:
                self.stats.accepted_frames += 1
                return parsed

            self.stats.crc_failures += 1
            crc_failures_in_call += 1
            if crc_failures_in_call >= max_crc_failures:
                return None

        self.stats.timeouts += 1
        return None

    def stream_frames(self, run_seconds: float | None = None, frame_timeout: float = 1.0):
        start = time.monotonic()
        while run_seconds is None or (time.monotonic() - start) < run_seconds:
            frame = self.read_frame(frame_timeout=frame_timeout)
            if frame is not None:
                yield frame


def record_raw_stream(
    port: str,
    output_path: str | Path,
    target_frames: int = 120,
    baud: int = 115200,
    timeout: float = 1.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    receiver = SerialFrameReceiver(port=port, baud=baud, timeout=timeout, mode="raw")
    receiver.open()

    written = 0
    with output.open("wb") as fp:
        try:
            while written < target_frames:
                frame = receiver.read_frame(frame_timeout=timeout)
                if frame is None:
                    continue
                # Re-pack frame bytes exactly as observed from serial stream.
                # Since read_frame returns parsed structures only, capture raw by reconstructing from fields.
                # For recording/validation workflows, this is sufficient because protocol is deterministic.
                from src.io.frame_protocol import pack_frame

                fp.write(pack_frame(frame.seq, frame.v_raw, frame.i_raw, n=frame.n))
                written += 1
        finally:
            receiver.close()

    return output


def record_feature_stream(
    port: str,
    output_path: str | Path,
    target_frames: int = 120,
    baud: int = 115200,
    timeout: float = 1.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    receiver = SerialFrameReceiver(port=port, baud=baud, timeout=timeout, mode="feature")
    receiver.open()

    written = 0
    with output.open("wb") as fp:
        try:
            while written < target_frames:
                frame = receiver.read_frame(frame_timeout=timeout)
                if frame is None:
                    continue
                # Re-pack feature frames as deterministic binary stream for later validation/replay.
                from src.io.frame_protocol import pack_feature_frame

                fp.write(pack_feature_frame(frame.seq, frame.features))
                written += 1
        finally:
            receiver.close()

    return output


def record_model4_stream(
    port: str,
    output_path: str | Path,
    target_frames: int = 120,
    baud: int = 115200,
    timeout: float = 1.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    receiver = SerialFrameReceiver(port=port, baud=baud, timeout=timeout, mode="model4")
    receiver.open()

    written = 0
    with output.open("wb") as fp:
        try:
            while written < target_frames:
                frame = receiver.read_frame(frame_timeout=timeout)
                if frame is None:
                    continue
                # Re-pack model-ready frames as deterministic binary stream for replay/validation.
                from src.io.frame_protocol import pack_model_ready_frame

                fp.write(pack_model_ready_frame(frame.seq, frame.X_wave, frame.X_mag, frame.X_phase))
                written += 1
        finally:
            receiver.close()

    return output


def record_frame_snapshots(
    port: str,
    output_path: str | Path,
    config_path: str | Path = "configs/default.yaml",
    target_frames: int = 120,
    baud: int = 115200,
    timeout: float = 1.0,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    expected_n = int(cfg["signal"]["samples_per_frame"])

    receiver = SerialFrameReceiver(port=port, baud=baud, timeout=timeout, mode="raw")
    receiver.open()

    written = 0
    with output.open("w", encoding="utf-8") as fp:
        try:
            while written < target_frames:
                frame = receiver.read_frame(frame_timeout=timeout)
                if frame is None:
                    continue

                processed = preprocess_frame(frame.v_raw, frame.i_raw, cfg, expected_n=expected_n)
                row = {
                    "timestamp": time.time(),
                    "seq": frame.seq,
                    "n": frame.n,
                    "v_raw": frame.v_raw.tolist(),
                    "i_raw": frame.i_raw.tolist(),
                    "v_phys": processed["v_phys"].tolist(),
                    "i_phys": processed["i_phys"].tolist(),
                    "v_norm": processed["v_norm"].tolist(),
                    "i_norm": processed["i_norm"].tolist(),
                }
                fp.write(json.dumps(row) + "\n")
                written += 1
        finally:
            receiver.close()

    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serial frame receiver for PQ monitor")
    parser.add_argument("--port", required=True, help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--output", required=True, help="Output path for recorded stream")
    parser.add_argument("--frames", type=int, default=120, help="How many valid frames to record")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=["raw", "snapshots", "feature", "model4"],
        default="model4",
        help=(
            "model4: model-ready frame stream (default), "
            "raw: binary raw-frame stream, "
            "feature: binary feature-frame stream, "
            "snapshots: jsonl with raw+processed arrays"
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config file for snapshot mode")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    start = time.time()
    if args.mode == "raw":
        out = record_raw_stream(
            port=args.port,
            output_path=args.output,
            target_frames=args.frames,
            baud=args.baud,
            timeout=args.timeout,
        )
    elif args.mode == "feature":
        out = record_feature_stream(
            port=args.port,
            output_path=args.output,
            target_frames=args.frames,
            baud=args.baud,
            timeout=args.timeout,
        )
    elif args.mode == "model4":
        out = record_model4_stream(
            port=args.port,
            output_path=args.output,
            target_frames=args.frames,
            baud=args.baud,
            timeout=args.timeout,
        )
    else:
        out = record_frame_snapshots(
            port=args.port,
            output_path=args.output,
            config_path=args.config,
            target_frames=args.frames,
            baud=args.baud,
            timeout=args.timeout,
        )
    elapsed = time.time() - start
    print(f"Recorded {args.frames} valid frames to {out} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
