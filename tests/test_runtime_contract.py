from __future__ import annotations

from pathlib import Path

import yaml


def test_config_voltage_calibration_matches_amc1301_tlv9001_frontend() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    calib = cfg["calibration"]

    assert calib["v_adc_midpoint"] == 1985
    assert calib["v_counts_to_volts"] == 3.3 / (
        4095.0 * ((560.0 / (2200000.0 + 560.0)) * 8.2 * 1.5)
    )


def test_config_defaults_to_tflite_receiver_mode() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["ml_inference"]["receiver_mode"] == "tflite"


def test_platformio_uses_teensy_cli_for_pi_headless_uploads() -> None:
    text = Path("firmware/teensy/pq_firmware/platformio.ini").read_text(encoding="utf-8")
    model4_block = text.split("[env:teensy41_model4]", 1)[1].split("[env:", 1)[0]
    raw_block = text.split("[env:teensy41_raw]", 1)[1].split("[env:", 1)[0]

    assert "upload_protocol = teensy-cli" in model4_block
    assert "upload_protocol = teensy-cli" in raw_block
    assert "-DPQ_FREE_RUN_FALLBACK=1" in model4_block
    assert "-DPQ_FREE_RUN_FALLBACK=1" in raw_block


def test_pq_firmware_uses_direct_analog_reads_for_frame_capture() -> None:
    text = Path("firmware/teensy/pq_firmware/src/main.cpp").read_text(encoding="utf-8")

    assert "analogRead(PIN_VOLTAGE_ADC0)" in text
    assert "analogRead(PIN_CURRENT_ADC1)" in text
    assert "readSynchronizedSingle" not in text
    assert "startSynchronizedSingleRead" not in text


def test_live_infer_defaults_to_tflite_mode() -> None:
    text = Path("src/infer/live_infer.py").read_text(encoding="utf-8")
    assert 'choices=["tflite", "raw"]' in text
    assert 'default="tflite"' in text


def test_ui_defaults_to_tflite_mode() -> None:
    text = Path("src/ui/app.py").read_text(encoding="utf-8")
    assert 'choices=["tflite", "raw"]' in text
    assert 'default="tflite"' in text


def test_dashboard_uses_qt_orientation_enum() -> None:
    text = Path("src/ui/views/dashboard.py").read_text(encoding="utf-8")
    assert "QtCore.Qt.Orientation.Vertical" in text
    assert "setOrientation(1)" not in text


def test_readme_mentions_tflite_and_298_contract() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Feature vector length: `298`" in text
    assert "pqm_multilabel_model.tflite" in text
