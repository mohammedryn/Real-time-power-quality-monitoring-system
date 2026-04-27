import struct
import numpy as np
import pytest

from src.io.frame_protocol import (
    MAGIC_BYTES, N_FEATURES, FEATURE_FRAME_SIZE, FRAME_SIZE, N_SAMPLES,
    pack_feature_frame, parse_feature_frame, pack_frame,
    iter_frames_from_bytes, is_monotonic_modulo_u16,
    FeatureFrame,
)


def _random_features(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N_FEATURES).astype(np.float32)


# ---- Pack / parse roundtrip --------------------------------------------------

def test_feature_frame_roundtrip():
    feat  = _random_features()
    frame = pack_feature_frame(seq=42, features=feat)

    assert len(frame) == FEATURE_FRAME_SIZE
    assert frame[:4] == MAGIC_BYTES

    parsed = parse_feature_frame(frame)
    assert parsed.seq        == 42
    assert parsed.n_features == N_FEATURES
    assert parsed.crc_ok
    np.testing.assert_array_equal(parsed.features, feat)


def test_feature_frame_seq_wraps_u16():
    feat  = _random_features(1)
    frame = pack_feature_frame(seq=0xFFFF + 5, features=feat)
    parsed = parse_feature_frame(frame)
    assert parsed.seq == 4  # (0xFFFF + 5) & 0xFFFF


def test_feature_frame_crc_corruption_detected():
    frame     = bytearray(pack_feature_frame(seq=1, features=_random_features(2)))
    frame[-1] ^= 0xFF  # flip last CRC byte
    parsed = parse_feature_frame(bytes(frame))
    assert not parsed.crc_ok


def test_feature_frame_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        parse_feature_frame(b"\x00" * (FEATURE_FRAME_SIZE - 1))


def test_feature_frame_wrong_magic_raises():
    frame    = bytearray(pack_feature_frame(seq=1, features=_random_features(3)))
    frame[0] = 0x00  # corrupt magic
    with pytest.raises(ValueError, match="magic"):
        parse_feature_frame(bytes(frame))


def test_feature_frame_wrong_n_raises():
    frame = bytearray(pack_feature_frame(seq=1, features=_random_features(4)))
    # Overwrite n_features field (bytes 6-7) with wrong value
    struct.pack_into("<H", frame, 6, 500)
    with pytest.raises(ValueError, match="n_features"):
        parse_feature_frame(bytes(frame))


def test_pack_feature_frame_wrong_length_raises():
    with pytest.raises(ValueError, match="features"):
        pack_feature_frame(seq=0, features=np.zeros(100, dtype=np.float32))


# ---- Monotonic sequence check -----------------------------------------------

def test_feature_frame_seq_monotonic():
    seqs = [pack_feature_frame(s, _random_features(s)) for s in range(5)]
    frames = b"".join(seqs)
    parsed_seqs = []
    for fb in iter_frames_from_bytes(frames):
        pf = parse_feature_frame(fb)
        parsed_seqs.append(pf.seq)
    assert is_monotonic_modulo_u16(parsed_seqs)


# ---- Mixed raw + feature stream in iter_frames_from_bytes -------------------

def _raw_frame(seq: int) -> bytes:
    rng = np.random.default_rng(seq)
    v   = rng.integers(1800, 2300, N_SAMPLES, dtype=np.int16)
    i   = rng.integers(1900, 2200, N_SAMPLES, dtype=np.int16)
    return pack_frame(seq, v, i)


def test_mixed_stream_yields_both_frame_types():
    blob = (
        _raw_frame(0)
        + pack_feature_frame(1, _random_features(1))
        + _raw_frame(2)
        + pack_feature_frame(3, _random_features(3))
    )
    frames = list(iter_frames_from_bytes(blob))
    assert len(frames) == 4
    assert len(frames[0]) == FRAME_SIZE
    assert len(frames[1]) == FEATURE_FRAME_SIZE
    assert len(frames[2]) == FRAME_SIZE
    assert len(frames[3]) == FEATURE_FRAME_SIZE


def test_mixed_stream_leading_garbage_skipped():
    garbage = b"\xAB\xCD" * 64
    blob    = garbage + pack_feature_frame(7, _random_features(7))
    frames  = list(iter_frames_from_bytes(blob))
    assert len(frames) == 1
    assert len(frames[0]) == FEATURE_FRAME_SIZE
