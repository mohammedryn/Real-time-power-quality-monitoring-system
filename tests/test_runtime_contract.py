from __future__ import annotations

from pathlib import Path

import yaml


def test_config_voltage_calibration_matches_amc1301_tlv9001_frontend() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    calib = cfg["calibration"]

    assert calib["v_adc_midpoint"] == 2007
    assert calib["v_counts_to_volts"] == 3.3 / (
        4095.0 * ((560.0 / (2200000.0 + 560.0)) * 8.2 * 1.5)
    )


def test_config_defaults_to_tflite_receiver_mode() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["ml_inference"]["receiver_mode"] == "tflite"


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
