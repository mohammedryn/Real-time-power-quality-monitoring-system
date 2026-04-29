from __future__ import annotations

import argparse
import binascii
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Union

import numpy as np

# ---- Constants ---------------------------------------------------------------

MAGIC         = 0xDEADBEEF
MAGIC_BYTES   = MAGIC.to_bytes(4, "big")

N_SAMPLES     = 500
N_FEATURES    = 282  # legacy feature frame — kept for backward compat

# Raw frame: [magic 4B BE][seq 2B LE][n 2B LE][v_raw 1000B][i_raw 1000B][crc32 4B LE]
_RAW_PAYLOAD  = 2 + 2 + (N_SAMPLES * 2) + (N_SAMPLES * 2)
FRAME_SIZE    = 4 + _RAW_PAYLOAD + 4        # 2012 bytes

# Feature frame: [magic 4B BE][seq 2B LE][n 2B LE][features 1128B LE float32][crc32 4B LE]
_FEAT_PAYLOAD = 2 + 2 + (N_FEATURES * 4)
FEATURE_FRAME_SIZE = 4 + _FEAT_PAYLOAD + 4  # 1140 bytes

# Model-ready frame type tag (occupies the same 2-byte "n" field; value 3
# does not collide with N_SAMPLES=500 or N_FEATURES=282).
MODEL_READY_FRAME_TYPE = 0x0003

# Model-ready frame layout:
#   [magic 4B BE][seq 2B LE][type 2B LE=0x0003]
#   [X_wave 4000B: v_norm(500 f32) + i_norm(500 f32)]
#   [X_mag  112B:  28 f32]
#   [X_phase 1080B: 270 f32]
#   [crc32 4B LE]
# X_phase = feat[0:28] ++ feat[56:214] ++ feat[214:298] (270 floats total)
_N_XWAVE_FLOATS  = 1000   # v_norm[500] + i_norm[500]
_N_XMAG_FLOATS   = 28
_N_XPHASE_FLOATS = 270

_MODEL_PAYLOAD = (
    2                          # seq
    + 2                        # type
    + _N_XWAVE_FLOATS  * 4    # X_wave
    + _N_XMAG_FLOATS   * 4    # X_mag
    + _N_XPHASE_FLOATS * 4    # X_phase
)                              # = 5196 bytes

MODEL_READY_FRAME_SIZE = 4 + _MODEL_PAYLOAD + 4   # 5204 bytes

# Valid "n" values and their corresponding total frame sizes.
# The model-ready frame uses type=3 in the "n" field; add it here so that
# iter_frames_from_bytes() can identify and size it correctly.
_FRAME_SIZE_FOR_N: dict[int, int] = {
    N_SAMPLES:              FRAME_SIZE,
    N_FEATURES:             FEATURE_FRAME_SIZE,
    MODEL_READY_FRAME_TYPE: MODEL_READY_FRAME_SIZE,
}

# ---- Data classes ------------------------------------------------------------

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
class FeatureFrame:
    seq: int
    n_features: int
    features: np.ndarray   # float32, shape (282,)
    rx_crc: int
    calc_crc: int

    @property
    def crc_ok(self) -> bool:
        return self.rx_crc == self.calc_crc


@dataclass
class ModelReadyFrame:
    """Frame produced by the Teensy model-ready DSP pipeline.

    Contains three arrays that are fed directly to model_4:
      X_wave  – peak-normalised waveform (500, 2) after reshape
      X_mag   – harmonic magnitude features (28,)
      X_phase – phase + wavelet features   (270,)
    """
    seq: int
    X_wave: np.ndarray   # float32, shape (1000,) = v_norm[500] + i_norm[500]
    X_mag: np.ndarray    # float32, shape (28,)
    X_phase: np.ndarray  # float32, shape (270,)
    rx_crc: int
    calc_crc: int

    @property
    def crc_ok(self) -> bool:
        return self.rx_crc == self.calc_crc

    @property
    def v_norm(self) -> np.ndarray:
        return self.X_wave[:500]

    @property
    def i_norm(self) -> np.ndarray:
        return self.X_wave[500:]


@dataclass
class ValidationReport:
    total_frames: int
    valid_frames: int
    crc_failures: int
    sequence_monotonic: bool
    first_seq: int | None
    last_seq: int | None


# ---- CRC --------------------------------------------------------------------

def compute_crc(payload: bytes) -> int:
    return binascii.crc32(payload) & 0xFFFFFFFF


# ---- Raw frame pack / parse --------------------------------------------------

def pack_frame(seq: int, v_raw: np.ndarray, i_raw: np.ndarray,
               n: int = N_SAMPLES) -> bytes:
    v = np.asarray(v_raw, dtype="<i2")
    i = np.asarray(i_raw, dtype="<i2")
    if len(v) != n or len(i) != n:
        raise ValueError(
            f"Expected {n} samples per channel, got len(v)={len(v)}, len(i)={len(i)}")
    payload = struct.pack("<HH", seq & 0xFFFF, n) + v.tobytes() + i.tobytes()
    crc = compute_crc(payload)
    return MAGIC_BYTES + payload + struct.pack("<I", crc)


def parse_frame(frame: bytes) -> ParsedFrame:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"Invalid frame length {len(frame)}; expected {FRAME_SIZE}")
    if frame[:4] != MAGIC_BYTES:
        raise ValueError("Invalid magic header")

    payload = frame[4:4 + _RAW_PAYLOAD]
    seq, n = struct.unpack_from("<HH", payload, 0)
    if n != N_SAMPLES:
        raise ValueError(f"Invalid sample count n={n}; expected {N_SAMPLES}")

    v_start = 4
    v_end   = v_start + N_SAMPLES * 2
    i_end   = v_end   + N_SAMPLES * 2
    v_raw   = np.frombuffer(payload[v_start:v_end], dtype="<i2").copy()
    i_raw   = np.frombuffer(payload[v_end:i_end],   dtype="<i2").copy()

    rx_crc   = struct.unpack_from("<I", frame, 4 + _RAW_PAYLOAD)[0]
    calc_crc = compute_crc(payload)
    return ParsedFrame(seq=seq, n=n, v_raw=v_raw, i_raw=i_raw,
                       rx_crc=rx_crc, calc_crc=calc_crc)


# ---- Feature frame pack / parse ---------------------------------------------

def pack_feature_frame(seq: int, features: np.ndarray) -> bytes:
    """Pack a 282-element float32 feature vector into a 1140-byte feature frame."""
    feat = np.asarray(features, dtype="<f4")
    if len(feat) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, got {len(feat)}")
    payload = struct.pack("<HH", seq & 0xFFFF, N_FEATURES) + feat.tobytes()
    crc = compute_crc(payload)
    return MAGIC_BYTES + payload + struct.pack("<I", crc)


def parse_feature_frame(frame: bytes) -> FeatureFrame:
    """Parse a 1140-byte feature frame."""
    if len(frame) != FEATURE_FRAME_SIZE:
        raise ValueError(
            f"Invalid feature frame length {len(frame)}; expected {FEATURE_FRAME_SIZE}")
    if frame[:4] != MAGIC_BYTES:
        raise ValueError("Invalid magic header")

    payload    = frame[4:4 + _FEAT_PAYLOAD]
    seq, n_feat = struct.unpack_from("<HH", payload, 0)
    if n_feat != N_FEATURES:
        raise ValueError(f"Invalid n_features={n_feat}; expected {N_FEATURES}")

    features = np.frombuffer(payload[4:4 + N_FEATURES * 4], dtype="<f4").copy()
    rx_crc   = struct.unpack_from("<I", frame, 4 + _FEAT_PAYLOAD)[0]
    calc_crc = compute_crc(payload)
    return FeatureFrame(seq=seq, n_features=n_feat, features=features,
                        rx_crc=rx_crc, calc_crc=calc_crc)


# ---- Model-ready frame pack / parse -----------------------------------------

def pack_model_ready_frame(
    seq: int,
    X_wave: np.ndarray,   # shape (1000,) = v_norm[500] + i_norm[500]
    X_mag: np.ndarray,    # shape (28,)
    X_phase: np.ndarray,  # shape (270,)
) -> bytes:
    """Pack a model-ready frame (5204 bytes)."""
    xw = np.asarray(X_wave,  dtype="<f4").reshape(-1)
    xm = np.asarray(X_mag,   dtype="<f4").reshape(-1)
    xp = np.asarray(X_phase, dtype="<f4").reshape(-1)
    if len(xw) != _N_XWAVE_FLOATS:
        raise ValueError(f"X_wave must have {_N_XWAVE_FLOATS} floats, got {len(xw)}")
    if len(xm) != _N_XMAG_FLOATS:
        raise ValueError(f"X_mag must have {_N_XMAG_FLOATS} floats, got {len(xm)}")
    if len(xp) != _N_XPHASE_FLOATS:
        raise ValueError(f"X_phase must have {_N_XPHASE_FLOATS} floats, got {len(xp)}")

    payload = (
        struct.pack("<HH", seq & 0xFFFF, MODEL_READY_FRAME_TYPE)
        + xw.tobytes()
        + xm.tobytes()
        + xp.tobytes()
    )
    crc = compute_crc(payload)
    return MAGIC_BYTES + payload + struct.pack("<I", crc)


def parse_model_ready_frame(frame: bytes) -> ModelReadyFrame:
    """Parse a 5204-byte model-ready frame."""
    if len(frame) != MODEL_READY_FRAME_SIZE:
        raise ValueError(
            f"Invalid model-ready frame length {len(frame)}; "
            f"expected {MODEL_READY_FRAME_SIZE}")
    if frame[:4] != MAGIC_BYTES:
        raise ValueError("Invalid magic header")

    payload = frame[4:4 + _MODEL_PAYLOAD]
    seq, ftype = struct.unpack_from("<HH", payload, 0)
    if ftype != MODEL_READY_FRAME_TYPE:
        raise ValueError(f"Invalid frame type {ftype:#06x}; "
                         f"expected {MODEL_READY_FRAME_TYPE:#06x}")

    offset = 4
    xw_bytes = offset + _N_XWAVE_FLOATS  * 4
    xm_bytes = xw_bytes + _N_XMAG_FLOATS  * 4
    xp_bytes = xm_bytes + _N_XPHASE_FLOATS * 4

    X_wave  = np.frombuffer(payload[offset:xw_bytes], dtype="<f4").copy()
    X_mag   = np.frombuffer(payload[xw_bytes:xm_bytes], dtype="<f4").copy()
    X_phase = np.frombuffer(payload[xm_bytes:xp_bytes], dtype="<f4").copy()

    rx_crc   = struct.unpack_from("<I", frame, 4 + _MODEL_PAYLOAD)[0]
    calc_crc = compute_crc(payload)

    return ModelReadyFrame(
        seq=seq,
        X_wave=X_wave,
        X_mag=X_mag,
        X_phase=X_phase,
        rx_crc=rx_crc,
        calc_crc=calc_crc,
    )


# ---- Variable-length iterator -----------------------------------------------
# Handles mixed raw (n=500), feature (n=282), and model-ready (type=3) frames.

def iter_frames_from_bytes(blob: bytes) -> Iterator[bytes]:
    idx      = 0
    blob_len = len(blob)

    while idx + 4 <= blob_len:
        magic_pos = blob.find(MAGIC_BYTES, idx)
        if magic_pos < 0:
            return

        # Need at least magic(4) + seq(2) + n/type(2) = 8 bytes
        if magic_pos + 8 > blob_len:
            return

        _seq, n = struct.unpack_from("<HH", blob, magic_pos + 4)

        expected_size = _FRAME_SIZE_FOR_N.get(n)
        if expected_size is None:
            idx = magic_pos + 1
            continue

        end = magic_pos + expected_size
        if end > blob_len:
            return  # truncated

        yield blob[magic_pos:end]
        idx = end


# ---- Sequence monotonicity --------------------------------------------------

def is_monotonic_modulo_u16(seqs: list[int]) -> bool:
    if len(seqs) < 2:
        return True
    for prev, cur in zip(seqs[:-1], seqs[1:]):
        if ((prev + 1) & 0xFFFF) != cur:
            return False
    return True


# ---- Stream validator -------------------------------------------------------

def validate_recorded_stream(path: str | Path,
                             min_frames: int = 100) -> ValidationReport:
    raw = Path(path).read_bytes()

    total = 0
    valid = 0
    crc_failures = 0
    seqs: list[int] = []

    for frame_bytes in iter_frames_from_bytes(raw):
        total += 1
        _seq, n = struct.unpack_from("<HH", frame_bytes, 4)
        try:
            if n == N_SAMPLES:
                parsed: Union[ParsedFrame, FeatureFrame, ModelReadyFrame] = parse_frame(frame_bytes)
            elif n == N_FEATURES:
                parsed = parse_feature_frame(frame_bytes)
            elif n == MODEL_READY_FRAME_TYPE:
                parsed = parse_model_ready_frame(frame_bytes)
            else:
                continue
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
            f"Validation failed: valid_frames={report.valid_frames} < "
            f"min_frames={min_frames}")
    if not report.sequence_monotonic:
        raise ValueError(
            "Validation failed: sequence numbers are not monotonic modulo uint16")

    return report


# ---- CLI --------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frame protocol utilities for PQ monitor")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate",
                                  help="Validate a recorded binary stream")
    validate_cmd.add_argument("--input", required=True,
                              help="Path to recorded raw stream binary")
    validate_cmd.add_argument("--min-frames", type=int, default=100,
                              help="Minimum valid frames required")
    return parser


def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "validate":
        report = validate_recorded_stream(args.input, min_frames=args.min_frames)
        print(json.dumps(asdict(report), indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
