from __future__ import annotations

import numpy as np

from src.io.frame_protocol import (
    FRAME_SIZE,
    MAGIC_BYTES,
    ValidationReport,
    is_monotonic_modulo_u16,
    pack_frame,
    parse_frame,
    validate_recorded_stream,
)


def _rand_wave(seed: int, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    v = rng.integers(low=-2048, high=2047, size=n, dtype=np.int16)
    i = rng.integers(low=-2048, high=2047, size=n, dtype=np.int16)
    return v, i


def test_pack_parse_roundtrip() -> None:
    v, i = _rand_wave(seed=1)
    frame = pack_frame(seq=42, v_raw=v, i_raw=i)

    assert len(frame) == FRAME_SIZE
    assert frame[:4] == MAGIC_BYTES

    parsed = parse_frame(frame)
    assert parsed.seq == 42
    assert parsed.n == 500
    assert parsed.crc_ok
    np.testing.assert_array_equal(parsed.v_raw, v)
    np.testing.assert_array_equal(parsed.i_raw, i)


def test_monotonic_modulo_u16() -> None:
    seqs = [65534, 65535, 0, 1, 2]
    assert is_monotonic_modulo_u16(seqs)


def test_validate_recorded_stream_100_plus_frames(tmp_path) -> None:
    blob = bytearray(b"\x00\x11noise")
    expected_frames = 130

    for seq in range(expected_frames):
        v, i = _rand_wave(seed=1000 + seq)
        blob.extend(pack_frame(seq=seq, v_raw=v, i_raw=i))

    stream_path = tmp_path / "frames.bin"
    stream_path.write_bytes(bytes(blob))

    report = validate_recorded_stream(stream_path, min_frames=100)
    assert isinstance(report, ValidationReport)
    assert report.valid_frames >= 100
    assert report.crc_failures == 0
    assert report.sequence_monotonic
    assert report.first_seq == 0
    assert report.last_seq == expected_frames - 1
