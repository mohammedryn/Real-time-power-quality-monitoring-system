"""Tests for the model-ready frame protocol (5204-byte frame, type=0x0003)."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from src.io.frame_protocol import (
    MAGIC_BYTES,
    MODEL_READY_FRAME_SIZE,
    MODEL_READY_FRAME_TYPE,
    ModelReadyFrame,
    compute_crc,
    iter_frames_from_bytes,
    pack_frame,
    pack_feature_frame,
    pack_model_ready_frame,
    parse_model_ready_frame,
    N_SAMPLES,
    N_FEATURES,
)


# ---- Helpers -----------------------------------------------------------------

def _make_arrays(rng: np.random.Generator):
    """Return (X_wave, X_mag, X_phase) filled with deterministic floats."""
    X_wave  = rng.standard_normal(1000).astype(np.float32)
    X_mag   = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)
    return X_wave, X_mag, X_phase


# ---- Frame size --------------------------------------------------------------

def test_model_ready_frame_size_constant():
    # magic(4) + payload(seq2+type2+wave4000+mag112+phase1080) + crc(4)
    assert MODEL_READY_FRAME_SIZE == 5204


# ---- Pack / parse roundtrip --------------------------------------------------

def test_pack_parse_roundtrip():
    rng = np.random.default_rng(0)
    X_wave, X_mag, X_phase = _make_arrays(rng)

    raw = pack_model_ready_frame(42, X_wave, X_mag, X_phase)
    assert len(raw) == MODEL_READY_FRAME_SIZE

    parsed = parse_model_ready_frame(raw)
    assert parsed.seq == 42
    assert parsed.crc_ok

    np.testing.assert_array_equal(parsed.X_wave,  X_wave)
    np.testing.assert_array_equal(parsed.X_mag,   X_mag)
    np.testing.assert_array_equal(parsed.X_phase, X_phase)


def test_seq_wraps_modulo_u16():
    rng = np.random.default_rng(1)
    X_wave, X_mag, X_phase = _make_arrays(rng)

    raw = pack_model_ready_frame(0x10001, X_wave, X_mag, X_phase)
    parsed = parse_model_ready_frame(raw)
    assert parsed.seq == 1   # 0x10001 & 0xFFFF


# ---- Magic header ------------------------------------------------------------

def test_magic_header_is_present():
    rng = np.random.default_rng(2)
    raw = pack_model_ready_frame(0, *_make_arrays(rng))
    assert raw[:4] == MAGIC_BYTES


# ---- Frame type tag ----------------------------------------------------------

def test_frame_type_tag_is_0x0003():
    rng = np.random.default_rng(3)
    raw = pack_model_ready_frame(0, *_make_arrays(rng))
    # Bytes 6-7 (after magic4 + seq2) carry the frame type
    ftype = struct.unpack_from("<H", raw, 6)[0]
    assert ftype == MODEL_READY_FRAME_TYPE
    assert ftype == 0x0003


# ---- CRC validation ----------------------------------------------------------

def test_crc_ok_on_intact_frame():
    rng = np.random.default_rng(4)
    raw = pack_model_ready_frame(7, *_make_arrays(rng))
    parsed = parse_model_ready_frame(raw)
    assert parsed.rx_crc == parsed.calc_crc
    assert parsed.crc_ok


def test_crc_fails_on_corrupted_frame():
    rng = np.random.default_rng(5)
    raw = bytearray(pack_model_ready_frame(8, *_make_arrays(rng)))
    # Flip a byte in the X_wave section
    raw[10] ^= 0xFF
    parsed = parse_model_ready_frame(bytes(raw))
    assert not parsed.crc_ok


# ---- Output shapes ----------------------------------------------------------

def test_output_shapes():
    rng = np.random.default_rng(6)
    X_wave, X_mag, X_phase = _make_arrays(rng)
    raw = pack_model_ready_frame(0, X_wave, X_mag, X_phase)
    parsed = parse_model_ready_frame(raw)

    assert parsed.X_wave.shape  == (1000,)
    assert parsed.X_mag.shape   == (28,)
    assert parsed.X_phase.shape == (270,)


# ---- v_norm / i_norm properties ---------------------------------------------

def test_v_norm_i_norm_split():
    rng = np.random.default_rng(7)
    X_wave  = rng.standard_normal(1000).astype(np.float32)
    X_mag   = rng.standard_normal(28).astype(np.float32)
    X_phase = rng.standard_normal(270).astype(np.float32)

    raw = pack_model_ready_frame(0, X_wave, X_mag, X_phase)
    parsed = parse_model_ready_frame(raw)

    np.testing.assert_array_equal(parsed.v_norm, X_wave[:500])
    np.testing.assert_array_equal(parsed.i_norm, X_wave[500:])


# ---- Input validation -------------------------------------------------------

def test_wrong_xwave_length_raises():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="X_wave"):
        pack_model_ready_frame(0, np.zeros(500, dtype=np.float32),
                               np.zeros(28, dtype=np.float32),
                               np.zeros(270, dtype=np.float32))


def test_wrong_xmag_length_raises():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="X_mag"):
        pack_model_ready_frame(0, np.zeros(1000, dtype=np.float32),
                               np.zeros(10, dtype=np.float32),
                               np.zeros(270, dtype=np.float32))


def test_wrong_xphase_length_raises():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="X_phase"):
        pack_model_ready_frame(0, np.zeros(1000, dtype=np.float32),
                               np.zeros(28, dtype=np.float32),
                               np.zeros(100, dtype=np.float32))


def test_parse_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        parse_model_ready_frame(b"\x00" * 100)


def test_parse_wrong_magic_raises():
    bad = b"\x00\x00\x00\x00" + b"\x00" * (MODEL_READY_FRAME_SIZE - 4)
    with pytest.raises(ValueError, match="magic"):
        parse_model_ready_frame(bad)


# ---- iter_frames_from_bytes with mixed stream --------------------------------

def test_iter_finds_model_ready_frames_in_mixed_stream():
    rng = np.random.default_rng(11)
    X_wave, X_mag, X_phase = _make_arrays(rng)

    model_frame = pack_model_ready_frame(10, X_wave, X_mag, X_phase)
    raw_v = rng.integers(1700, 2400, size=N_SAMPLES, dtype=np.int16)
    raw_i = rng.integers(1800, 2300, size=N_SAMPLES, dtype=np.int16)
    raw_frame = pack_frame(11, raw_v, raw_i)
    feat_frame = pack_feature_frame(12, rng.standard_normal(N_FEATURES).astype(np.float32))

    blob = raw_frame + model_frame + feat_frame

    found = list(iter_frames_from_bytes(blob))
    assert len(found) == 3
    assert len(found[0]) == len(raw_frame)
    assert len(found[1]) == MODEL_READY_FRAME_SIZE
    assert len(found[2]) == len(feat_frame)


def test_iter_skips_garbage_before_magic():
    rng = np.random.default_rng(12)
    X_wave, X_mag, X_phase = _make_arrays(rng)

    garbage = b"\xAA\xBB\xCC" * 50
    model_frame = pack_model_ready_frame(99, X_wave, X_mag, X_phase)

    blob = garbage + model_frame
    found = list(iter_frames_from_bytes(blob))
    assert len(found) == 1
    assert len(found[0]) == MODEL_READY_FRAME_SIZE


# ---- Feature reconstruction roundtrip ---------------------------------------

def test_feature_reconstruction_from_x_mag_x_phase():
    """Verify the Pi-side reconstruction: feat = X_phase[:28] ++ X_mag ++ X_phase[28:]."""
    rng = np.random.default_rng(13)
    # Simulate a 298-element feature vector
    feat298 = rng.standard_normal(298).astype(np.float32)

    # Slice as the Teensy would
    X_mag   = feat298[28:56].copy()
    X_phase = np.concatenate([feat298[0:28], feat298[56:214], feat298[214:298]])

    # Reconstruct as the Pi does
    reconstructed = np.concatenate([X_phase[:28], X_mag, X_phase[28:]])

    np.testing.assert_array_almost_equal(reconstructed, feat298, decimal=6)
