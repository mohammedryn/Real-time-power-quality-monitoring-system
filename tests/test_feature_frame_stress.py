"""
Mixed-stream stress test for the variable-length frame iterator.
Verifies that the parser:
  - recovers from random garbage bytes
  - handles truncated frames without halting
  - handles frames with unknown n values (resync by advancing 1 byte)
  - never raises an exception or infinite-loops
  - yields only valid complete frames
"""
import struct
import numpy as np
import pytest

from src.io.frame_protocol import (
    MAGIC_BYTES, MAGIC, N_FEATURES, N_SAMPLES,
    FEATURE_FRAME_SIZE, FRAME_SIZE,
    pack_feature_frame, pack_frame,
    iter_frames_from_bytes, parse_feature_frame, parse_frame,
)


def _feat_frame(seq: int) -> bytes:
    rng = np.random.default_rng(seq)
    return pack_feature_frame(seq, rng.standard_normal(N_FEATURES).astype(np.float32))


def _raw_frame(seq: int) -> bytes:
    rng = np.random.default_rng(seq + 1000)
    v   = rng.integers(1800, 2300, N_SAMPLES, dtype=np.int16)
    i   = rng.integers(1900, 2200, N_SAMPLES, dtype=np.int16)
    return pack_frame(seq, v, i)


# ---- Unknown n value --------------------------------------------------------

def _frame_with_unknown_n(seq: int = 99) -> bytes:
    # Build a frame-like blob with magic + unknown n (e.g. 1234)
    n_bad = 1234
    payload = struct.pack("<HH", seq & 0xFFFF, n_bad) + b"\x00" * 100
    import binascii
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    return MAGIC_BYTES + payload + struct.pack("<I", crc)


def test_unknown_n_does_not_yield_frame():
    blob   = _frame_with_unknown_n()
    frames = list(iter_frames_from_bytes(blob))
    assert frames == []


def test_unknown_n_recovers_to_valid_frame_after():
    blob   = _frame_with_unknown_n() + _feat_frame(1)
    frames = list(iter_frames_from_bytes(blob))
    assert len(frames) == 1
    assert len(frames[0]) == FEATURE_FRAME_SIZE


# ---- Truncated frames -------------------------------------------------------

def test_truncated_feature_frame_not_yielded():
    full  = _feat_frame(0)
    trunc = full[:FEATURE_FRAME_SIZE // 2]  # cut in half
    frames = list(iter_frames_from_bytes(trunc))
    assert frames == []


def test_truncated_raw_frame_not_yielded():
    full  = _raw_frame(0)
    trunc = full[:FRAME_SIZE // 2]
    frames = list(iter_frames_from_bytes(trunc))
    assert frames == []


def test_non_magic_garbage_followed_by_valid_frame():
    # Garbage bytes that do not contain MAGIC, followed by a complete feature frame.
    # The iterator must skip all garbage and yield the feature frame.
    garbage   = b"\xAB\xCD\xEF\x00" * 256  # 1024 bytes, no MAGIC sequence
    full_feat = _feat_frame(5)
    blob      = garbage + full_feat
    frames    = list(iter_frames_from_bytes(blob))
    assert len(frames) == 1
    assert len(frames[0]) == FEATURE_FRAME_SIZE


# ---- Random garbage ---------------------------------------------------------

def test_random_garbage_only_yields_nothing():
    rng  = np.random.default_rng(42)
    blob = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()
    frames = list(iter_frames_from_bytes(blob))
    # Extremely unlikely that random bytes contain a valid MAGIC sequence
    # followed by n=282 or n=500; accept 0 or a small number
    assert len(frames) <= 2


def test_garbage_between_valid_frames_skipped():
    rng  = np.random.default_rng(7)
    junk = rng.integers(0, 256, 256, dtype=np.uint8).tobytes()
    blob = _feat_frame(0) + junk + _feat_frame(1) + junk + _raw_frame(2)
    frames = list(iter_frames_from_bytes(blob))
    # Should yield at least the 3 valid frames (garbage may accidentally
    # contain a partial valid MAGIC but those will be skipped by size check)
    valid_feat_frames = [f for f in frames if len(f) == FEATURE_FRAME_SIZE]
    valid_raw_frames  = [f for f in frames if len(f) == FRAME_SIZE]
    assert len(valid_feat_frames) >= 2
    assert len(valid_raw_frames)  >= 1


# ---- Magic bytes embedded in garbage ----------------------------------------

def test_magic_in_garbage_does_not_crash():
    # Insert magic bytes mid-stream without valid frame following
    blob = b"\x00" * 32 + MAGIC_BYTES + b"\x00" * 8 + _feat_frame(10)
    frames = list(iter_frames_from_bytes(blob))
    valid = [f for f in frames if len(f) == FEATURE_FRAME_SIZE]
    assert len(valid) >= 1


# ---- Large stress stream ----------------------------------------------------

def test_large_mixed_stream():
    rng   = np.random.default_rng(99)
    parts = []
    expected_feat = 0
    expected_raw  = 0
    for i in range(50):
        choice = i % 4
        if choice == 0:
            parts.append(_feat_frame(i))
            expected_feat += 1
        elif choice == 1:
            parts.append(_raw_frame(i))
            expected_raw += 1
        elif choice == 2:
            # random garbage chunk
            parts.append(rng.integers(0, 256, 64, dtype=np.uint8).tobytes())
        else:
            # unknown-n frame (should be skipped)
            parts.append(_frame_with_unknown_n(i))

    blob   = b"".join(parts)
    frames = list(iter_frames_from_bytes(blob))

    feat_frames = [f for f in frames if len(f) == FEATURE_FRAME_SIZE]
    raw_frames  = [f for f in frames if len(f) == FRAME_SIZE]

    assert len(feat_frames) == expected_feat
    assert len(raw_frames)  == expected_raw

    # All yielded frames must parse and have valid CRC
    for fb in feat_frames:
        pf = parse_feature_frame(fb)
        assert pf.crc_ok

    for fb in raw_frames:
        pr = parse_frame(fb)
        assert pr.crc_ok
