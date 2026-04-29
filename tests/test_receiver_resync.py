from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import serial

from src.io.frame_protocol import N_SAMPLES, pack_frame
from src.io.serial_receiver import SerialFrameReceiver
import src.io.serial_receiver as serial_receiver_mod


class FakeSerial:
    def __init__(self, payload: bytes, timeout: float = 0.01, max_chunk: int = 11, fail_reads: int = 0) -> None:
        self._buf = bytearray(payload)
        self.timeout = timeout
        self.max_chunk = max_chunk
        self.fail_reads = fail_reads
        self.is_open = True

    def read(self, n: int) -> bytes:
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise serial.SerialException("simulated serial disconnect")

        if not self._buf:
            return b""

        chunk = min(n, self.max_chunk, len(self._buf))
        out = bytes(self._buf[:chunk])
        del self._buf[:chunk]
        return out

    def close(self) -> None:
        self.is_open = False


def _mk_frame(seq: int) -> bytes:
    v = np.arange(N_SAMPLES, dtype=np.int16) + seq
    i = (np.arange(N_SAMPLES, dtype=np.int16) * 2) - seq
    return pack_frame(seq=seq, v_raw=v, i_raw=i)


def test_receiver_resync_drops_corrupt_and_recovers(monkeypatch):
    good_1 = _mk_frame(100)
    bad = bytearray(_mk_frame(101))
    bad[-1] ^= 0xFF
    good_2 = _mk_frame(102)

    stream = b"\x01\x02garbage" + good_1 + b"junk" + bytes(bad) + b"noise" + good_2
    fake = FakeSerial(stream, max_chunk=7)

    monkeypatch.setattr("src.io.serial_receiver.serial.Serial", lambda *args, **kwargs: fake)

    rx = SerialFrameReceiver(port="/dev/null", timeout=0.01)
    rx.open()

    frame_1 = rx.read_frame(frame_timeout=0.3)
    frame_2 = rx.read_frame(frame_timeout=0.3)

    assert frame_1 is not None
    assert frame_2 is not None
    assert frame_1.seq == 100
    assert frame_2.seq == 102
    assert rx.stats.crc_failures >= 1


def test_receiver_reconnect_after_serial_exception(monkeypatch):
    factory_calls: list[FakeSerial] = []

    def fake_factory(*_args, **_kwargs):
        if not factory_calls:
            s = FakeSerial(payload=b"", fail_reads=1)
        else:
            s = FakeSerial(payload=_mk_frame(200), max_chunk=9)
        factory_calls.append(s)
        return s

    monkeypatch.setattr("src.io.serial_receiver.serial.Serial", fake_factory)

    rx = SerialFrameReceiver(port="/dev/null", timeout=0.01, reconnect_delay=0.0, max_reconnect_attempts=2)
    rx.open()

    frame = rx.read_frame(frame_timeout=0.5)

    assert frame is not None
    assert frame.seq == 200
    assert rx.stats.reconnects >= 1


def test_main_dispatches_model4_mode_to_model4_recorder(monkeypatch, tmp_path):
    calls = {"model4": 0, "raw": 0, "feature": 0, "snapshots": 0}

    args = SimpleNamespace(
        port="/dev/null",
        output=str(tmp_path / "out.bin"),
        frames=3,
        baud=115200,
        timeout=0.1,
        mode="model4",
        config="configs/default.yaml",
    )

    class _Parser:
        def parse_args(self):
            return args

    monkeypatch.setattr(serial_receiver_mod, "_build_parser", lambda: _Parser())

    def _mk(kind):
        def _inner(**_kwargs):
            calls[kind] += 1
            return tmp_path / "dummy.out"
        return _inner

    monkeypatch.setattr(serial_receiver_mod, "record_model4_stream", _mk("model4"))
    monkeypatch.setattr(serial_receiver_mod, "record_raw_stream", _mk("raw"))
    monkeypatch.setattr(serial_receiver_mod, "record_feature_stream", _mk("feature"))
    monkeypatch.setattr(serial_receiver_mod, "record_frame_snapshots", _mk("snapshots"))

    rc = serial_receiver_mod.main()
    assert rc == 0
    assert calls["model4"] == 1
    assert calls["raw"] == 0
    assert calls["feature"] == 0
    assert calls["snapshots"] == 0