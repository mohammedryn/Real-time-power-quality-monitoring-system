from __future__ import annotations

import argparse
import binascii
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = 0xDEADBEEF
MAGIC_BYTES = MAGIC.to_bytes(4, "big")
N_SAMPLES = 500
PAYLOAD_BYTES = 2 + 2 + (N_SAMPLES * 2) + (N_SAMPLES * 2)
FRAME_SIZE = 4 + PAYLOAD_BYTES + 4


@dataclass
class ParsedFrame:
    seq: int
    n: int
    v_raw: np.ndarray
    i_raw: np.ndarray
    rx_crc: int
    calc_crc: int

    @property
    def crc_ok(self) -> bool:
        return self.rx_crc == self.calc_crc


@dataclass
class ValidationReport:
    total_frames: int
    valid_frames: int
    crc_failures: int
    sequence_monotonic: bool
    first_seq: int | None
    last_seq: int | None


def compute_crc(payload: bytes) -> int:
    return binascii.crc32(payload) & 0xFFFFFFFF


def pack_frame(seq: int, v_raw: np.ndarray, i_raw: np.ndarray, n: int = N_SAMPLES) -> bytes:
    v = np.asarray(v_raw, dtype="<i2")
    i = np.asarray(i_raw, dtype="<i2")

    if len(v) != n or len(i) != n:
        raise ValueError(f"Expected {n} samples per channel, got len(v)={len(v)}, len(i)={len(i)}")

    payload = struct.pack("<HH", seq & 0xFFFF, n) + v.tobytes() + i.tobytes()
    crc = compute_crc(payload)
    return MAGIC_BYTES + payload + struct.pack("<I", crc)


def parse_frame(frame: bytes) -> ParsedFrame:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"Invalid frame length {len(frame)}; expected {FRAME_SIZE}")
    if frame[:4] != MAGIC_BYTES:
        raise ValueError("Invalid magic header")

    payload = frame[4:4 + PAYLOAD_BYTES]
    seq, n = struct.unpack_from("<HH", payload, 0)

    if n != N_SAMPLES:
        raise ValueError(f"Invalid sample count n={n}; expected {N_SAMPLES}")

    v_start = 4
    v_end = v_start + (N_SAMPLES * 2)
    i_end = v_end + (N_SAMPLES * 2)

    v_raw = np.frombuffer(payload[v_start:v_end], dtype="<i2").copy()
    i_raw = np.frombuffer(payload[v_end:i_end], dtype="<i2").copy()

    rx_crc = struct.unpack_from("<I", frame, 4 + PAYLOAD_BYTES)[0]
    calc_crc = compute_crc(payload)

    return ParsedFrame(seq=seq, n=n, v_raw=v_raw, i_raw=i_raw, rx_crc=rx_crc, calc_crc=calc_crc)


def iter_frames_from_bytes(blob: bytes) -> Iterator[bytes]:
    idx = 0
    blob_len = len(blob)

    while idx + 4 <= blob_len:
        magic_pos = blob.find(MAGIC_BYTES, idx)
        if magic_pos < 0:
            return

        end = magic_pos + FRAME_SIZE
        if end > blob_len:
            return

        yield blob[magic_pos:end]
        idx = end


def is_monotonic_modulo_u16(seqs: list[int]) -> bool:
    if len(seqs) < 2:
        return True
    for prev, cur in zip(seqs[:-1], seqs[1:]):
        if ((prev + 1) & 0xFFFF) != cur:
            return False
    return True


def validate_recorded_stream(path: str | Path, min_frames: int = 100) -> ValidationReport:
    raw = Path(path).read_bytes()

    total = 0
    valid = 0
    crc_failures = 0
    seqs: list[int] = []

    for frame_bytes in iter_frames_from_bytes(raw):
        total += 1
        try:
            parsed = parse_frame(frame_bytes)
        except ValueError:
            continue

        if parsed.crc_ok:
            valid += 1
            seqs.append(parsed.seq)
        else:
            crc_failures += 1

    monotonic = is_monotonic_modulo_u16(seqs)

    report = ValidationReport(
        total_frames=total,
        valid_frames=valid,
        crc_failures=crc_failures,
        sequence_monotonic=monotonic,
        first_seq=seqs[0] if seqs else None,
        last_seq=seqs[-1] if seqs else None,
    )

    if report.valid_frames < min_frames:
        raise ValueError(
            f"Validation failed: valid_frames={report.valid_frames} < min_frames={min_frames}"
        )
    if not report.sequence_monotonic:
        raise ValueError("Validation failed: sequence numbers are not monotonic modulo uint16")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frame protocol utilities for PQ monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate", help="Validate a recorded binary stream")
    validate_cmd.add_argument("--input", required=True, help="Path to recorded raw stream binary")
    validate_cmd.add_argument("--min-frames", type=int, default=100, help="Minimum valid frames required")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        report = validate_recorded_stream(args.input, min_frames=args.min_frames)
        print(json.dumps(asdict(report), indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
