from __future__ import annotations

from pathlib import Path

from src.io.frame_protocol import FRAME_SIZE, INFERENCE_FRAME_SIZE, N_SAMPLES


HEADER = Path("firmware/esp32p4/pq_firmware/main/pq_frame_protocol.h")


def test_esp32p4_protocol_header_matches_host_constants() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "#define PQ_MAGIC 0xDEADBEEFu" in text
    assert f"#define PQ_FRAME_SAMPLES {N_SAMPLES}u" in text
    assert f"#define PQ_RAW_FRAME_BYTES {FRAME_SIZE}u" in text
    assert f"#define PQ_INFERENCE_FRAME_BYTES {INFERENCE_FRAME_SIZE}u" in text
    assert "#define PQ_INFERENCE_FRAME_TYPE 0x0003u" in text


def test_esp32p4_raw_protocol_uses_signed_16_bit_adc_payload() -> None:
    text = HEADER.read_text(encoding="utf-8")
    assert "int16_t v_raw[PQ_FRAME_SAMPLES]" in text
    assert "int16_t i_raw[PQ_FRAME_SAMPLES]" in text


def test_esp32p4_protocol_sources_exist() -> None:
    assert Path("firmware/esp32p4/pq_firmware/main/pq_frame_protocol.c").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_serial.c").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_serial.h").exists()


def test_esp32p4_adc_capture_sources_exist() -> None:
    assert Path("docs/esp32p4_migration_notes.md").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_adc.c").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_adc.h").exists()


def test_esp32p4_dsp_port_sources_exist() -> None:
    root = Path("firmware/esp32p4/pq_firmware/main/dsp")
    assert (root / "dsp.cpp").exists()
    assert (root / "dsp.h").exists()
    assert (root / "goertzel.h").exists()
    assert (root / "dwt.h").exists()


def test_esp32p4_model_ready_path_is_wired() -> None:
    main_text = Path("firmware/esp32p4/pq_firmware/main/main.c").read_text(encoding="utf-8")
    cmake_text = Path("firmware/esp32p4/pq_firmware/main/CMakeLists.txt").read_text(encoding="utf-8")
    assert "compute_model4_frame" in main_text
    assert "PQ_RAW_MODE" in main_text
    assert "target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=$ENV{PQ_RAW_MODE})" in cmake_text
    assert "target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=0)" in cmake_text
