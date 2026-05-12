from __future__ import annotations

from pathlib import Path


def test_teensy_firmware_is_preserved_under_legacy() -> None:
    legacy = Path("legacy/teensyfirmware")
    assert (legacy / "adc_probe" / "platformio.ini").exists()
    assert (legacy / "adc_probe" / "src" / "main.cpp").exists()
    assert (legacy / "pq_firmware" / "platformio.ini").exists()
    assert (legacy / "pq_firmware" / "src" / "main.cpp").exists()
    assert (legacy / "pq_firmware" / "src" / "dsp.cpp").exists()
    assert (legacy / "pq_firmware" / "src" / "dsp.h").exists()


def test_esp32p4_firmware_layout_exists() -> None:
    root = Path("firmware/esp32p4/pq_firmware")
    assert (root / "CMakeLists.txt").exists()
    assert (root / "main" / "CMakeLists.txt").exists()
    assert (root / "main" / "main.c").exists()


def test_esp32p4_build_scripts_exist() -> None:
    assert Path("scripts/compile_esp32p4_firmware.sh").exists()
    assert Path("scripts/flash_esp32p4_firmware.sh").exists()
