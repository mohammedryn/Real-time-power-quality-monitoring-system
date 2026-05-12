# ESP32-P4 Firmware Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the damaged Teensy USB acquisition path with an ESP32-P4 acquisition firmware while preserving the current Raspberry Pi runtime, serial frame protocol, UI, and the full Teensy firmware history under `legacy/teensyfirmware/`.

**Architecture:** The Raspberry Pi remains the system owner for live inference, TFLite execution, session logging, and UI. ESP32-P4 first becomes a protocol-compatible raw ADC frame source, then gains the existing Teensy DSP/model-ready `tflite` frame path after raw capture is verified. The Teensy firmware is archived before migration work starts, and the host parser continues to accept the same `0xDEADBEEF` raw and inference frame formats throughout the migration.

**Tech Stack:** ESP32-P4-WIFI6, ESP-IDF v5.3.1 or newer, ESP-IDF ADC continuous mode, USB Serial/JTAG or UART serial transport, C/C++ firmware, Python 3 runtime, NumPy, pytest, existing `src.io.frame_protocol` parser, existing `src.runtime.pipeline`, PlatformIO retained for legacy Teensy builds.

---

## Source References

- Espressif ESP32-P4 ADC overview: https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/adc/index.html
- Espressif ESP32-P4 ADC continuous mode: https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/adc/adc_continuous.html
- ESP32-P4 datasheet: https://documentation.espressif.com/esp32-p4_datasheet_en.html
- Waveshare ESP32-P4-WIFI6 wiki: https://www.waveshare.com/wiki/ESP32-P4-WIFI6
- Current raw frame parser: `src/io/frame_protocol.py`
- Current serial receiver: `src/io/serial_receiver.py`
- Current runtime pipeline: `src/runtime/pipeline.py`
- Current Teensy firmware: `firmware/teensy/`

## Important Migration Constraints

- Do not change the Pi serial protocol for Phase 1.
- Do not remove `firmware/teensy/` until ESP32-P4 raw and `tflite` paths are both verified.
- Preserve a full Teensy copy at `legacy/teensyfirmware/` before touching active firmware layout.
- Start with `raw` mode because it tests acquisition, framing, CRC, and host DSP independently from ESP32-P4 DSP.
- Treat ESP32-P4 ADC continuous multi-channel sampling as sequential group sampling unless a hardware skew test proves a tighter guarantee. For two channels, configure total conversion rate at `10000 Hz` to get `5000 samples/s/channel`.
- Keep voltage/current analog signals inside ESP32-P4 ADC limits. The current frontend was designed around Teensy 3.3 V ADC behavior; ESP32-P4 ADC attenuation and calibration must be verified before live mains testing.

## File Structure

### New Directories

- `legacy/teensyfirmware/`
  - Full preserved copy of `firmware/teensy/`.
  - Contains `adc_probe/` and `pq_firmware/`.

- `firmware/esp32p4/pq_firmware/`
  - ESP-IDF project for ESP32-P4 production firmware.
  - First supports raw frame mode, then model-ready `tflite` frame mode.

- `firmware/esp32p4/pq_firmware/main/`
  - ESP-IDF application source.

- `firmware/esp32p4/pq_firmware/main/dsp/`
  - ESP32-P4 port of Teensy DSP files after raw mode is verified.

### New Files

- `docs/esp32p4_migration_notes.md`
  - Pin mapping, ADC attenuation choice, measured channel skew, calibration notes, and live validation results.

- `scripts/compile_esp32p4_firmware.sh`
  - Builds the ESP-IDF project in `firmware/esp32p4/pq_firmware`.

- `scripts/flash_esp32p4_firmware.sh`
  - Flashes the ESP32-P4 board through the configured serial port.

- `scripts/probe_esp32p4_raw.py`
  - Reads ESP32-P4 raw frames through the existing Python parser and prints min/max/mean diagnostics.

- `tests/test_firmware_layout.py`
  - Verifies Teensy firmware is archived and ESP32-P4 firmware layout exists.

- `tests/test_esp32p4_protocol_contract.py`
  - Verifies ESP32-P4 protocol constants match `src.io.frame_protocol`.

### Modified Files

- `README.md`
  - Documents ESP32-P4 as active MCU path and Teensy as legacy fallback.

- `docs/pi_deployment_runbook.md`
  - Adds ESP32-P4 flash/test commands.

- `docs/demo_runbook.md`
  - Adds ESP32-P4 raw and `tflite` validation steps.

- `scripts/compile_teensy_firmware.sh`
  - Updates description to say Teensy is legacy after migration; command remains usable.

- `src/io/frame_protocol.py`
  - No wire-format changes. Only add comments if needed to state ESP32-P4 shares the same constants.

---

## Task 1: Preserve Teensy Firmware and Lock Firmware Layout

**Files:**
- Create: `legacy/teensyfirmware/`
- Create: `tests/test_firmware_layout.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing firmware layout test**

```python
# tests/test_firmware_layout.py
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
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python3 -m pytest -q tests/test_firmware_layout.py
```

Expected:

```text
FAILED tests/test_firmware_layout.py::test_teensy_firmware_is_preserved_under_legacy
FAILED tests/test_firmware_layout.py::test_esp32p4_firmware_layout_exists
```

- [ ] **Step 3: Preserve the current Teensy firmware**

Run:

```bash
mkdir -p legacy
cp -a firmware/teensy legacy/teensyfirmware
```

Do not delete `firmware/teensy/` in this task. It remains active until ESP32-P4 raw and `tflite` modes are both verified.

- [ ] **Step 4: Create the ESP32-P4 project skeleton**

Create:

```text
firmware/esp32p4/pq_firmware/CMakeLists.txt
firmware/esp32p4/pq_firmware/sdkconfig.defaults
firmware/esp32p4/pq_firmware/main/CMakeLists.txt
firmware/esp32p4/pq_firmware/main/main.c
```

Use this root CMake file:

```cmake
# firmware/esp32p4/pq_firmware/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(pq_monitor_esp32p4)
```

Use this main CMake file:

```cmake
# firmware/esp32p4/pq_firmware/main/CMakeLists.txt
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES esp_adc driver esp_timer
)
```

Use this temporary main file:

```c
// firmware/esp32p4/pq_firmware/main/main.c
#include "esp_log.h"

static const char *TAG = "pq_esp32p4";

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-P4 PQ firmware skeleton booted");
}
```

Use this sdkconfig defaults file:

```text
# firmware/esp32p4/pq_firmware/sdkconfig.defaults
CONFIG_IDF_TARGET="esp32p4"
CONFIG_LOG_DEFAULT_LEVEL_INFO=y
```

- [ ] **Step 5: Run the layout test again**

Run:

```bash
python3 -m pytest -q tests/test_firmware_layout.py
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add legacy/teensyfirmware firmware/esp32p4 tests/test_firmware_layout.py README.md
git commit -m "chore: preserve teensy firmware and scaffold esp32p4 firmware"
```

---

## Task 2: Add ESP32-P4 Build and Flash Scripts

**Files:**
- Create: `scripts/compile_esp32p4_firmware.sh`
- Create: `scripts/flash_esp32p4_firmware.sh`
- Modify: `tests/test_firmware_layout.py`

- [ ] **Step 1: Add failing script checks**

Append to `tests/test_firmware_layout.py`:

```python
def test_esp32p4_build_scripts_exist() -> None:
    assert Path("scripts/compile_esp32p4_firmware.sh").exists()
    assert Path("scripts/flash_esp32p4_firmware.sh").exists()
```

- [ ] **Step 2: Run the failing check**

Run:

```bash
python3 -m pytest -q tests/test_firmware_layout.py::test_esp32p4_build_scripts_exist
```

Expected:

```text
FAILED tests/test_firmware_layout.py::test_esp32p4_build_scripts_exist
```

- [ ] **Step 3: Create compile script**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$REPO_ROOT/firmware/esp32p4/pq_firmware"

if ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF idf.py is not on PATH."
  echo "Run: . \$IDF_PATH/export.sh"
  exit 1
fi

idf.py -C "$FIRMWARE_DIR" set-target esp32p4
idf.py -C "$FIRMWARE_DIR" build

echo "ESP32-P4 firmware compile succeeded."
```

- [ ] **Step 4: Create flash script**

```bash
#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyACM0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$REPO_ROOT/firmware/esp32p4/pq_firmware"

if ! command -v idf.py >/dev/null 2>&1; then
  echo "ESP-IDF idf.py is not on PATH."
  echo "Run: . \$IDF_PATH/export.sh"
  exit 1
fi

idf.py -C "$FIRMWARE_DIR" -p "$PORT" flash
```

- [ ] **Step 5: Mark scripts executable**

Run:

```bash
chmod +x scripts/compile_esp32p4_firmware.sh scripts/flash_esp32p4_firmware.sh
```

- [ ] **Step 6: Run script checks**

Run:

```bash
python3 -m pytest -q tests/test_firmware_layout.py
```

Expected:

```text
3 passed
```

- [ ] **Step 7: Commit**

```bash
git add scripts/compile_esp32p4_firmware.sh scripts/flash_esp32p4_firmware.sh tests/test_firmware_layout.py
git commit -m "chore: add esp32p4 firmware build scripts"
```

---

## Task 3: Define the ESP32-P4 Raw Frame Protocol Contract

**Files:**
- Create: `firmware/esp32p4/pq_firmware/main/pq_frame_protocol.h`
- Create: `tests/test_esp32p4_protocol_contract.py`
- Modify: `firmware/esp32p4/pq_firmware/main/CMakeLists.txt`

- [ ] **Step 1: Write protocol contract tests**

```python
# tests/test_esp32p4_protocol_contract.py
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
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
python3 -m pytest -q tests/test_esp32p4_protocol_contract.py
```

Expected:

```text
FAILED tests/test_esp32p4_protocol_contract.py::test_esp32p4_protocol_header_matches_host_constants
```

- [ ] **Step 3: Add protocol header**

```c
// firmware/esp32p4/pq_firmware/main/pq_frame_protocol.h
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define PQ_MAGIC 0xDEADBEEFu
#define PQ_FRAME_SAMPLES 500u
#define PQ_RAW_PAYLOAD_BYTES (2u + 2u + (PQ_FRAME_SAMPLES * 2u) + (PQ_FRAME_SAMPLES * 2u))
#define PQ_RAW_FRAME_BYTES (4u + PQ_RAW_PAYLOAD_BYTES + 4u)

#define PQ_INFERENCE_FRAME_TYPE 0x0003u
#define PQ_XWAVE_FLOATS 1000u
#define PQ_XMAG_FLOATS 28u
#define PQ_XPHASE_FLOATS 270u
#define PQ_INFERENCE_PAYLOAD_BYTES (2u + 2u + (PQ_XWAVE_FLOATS * 4u) + (PQ_XMAG_FLOATS * 4u) + (PQ_XPHASE_FLOATS * 4u))
#define PQ_INFERENCE_FRAME_BYTES (4u + PQ_INFERENCE_PAYLOAD_BYTES + 4u)

typedef struct {
    uint16_t seq;
    uint16_t n;
    int16_t v_raw[PQ_FRAME_SAMPLES];
    int16_t i_raw[PQ_FRAME_SAMPLES];
} pq_raw_payload_t;

uint32_t pq_crc32_le(const uint8_t *data, uint32_t len);
void pq_write_u32_be(uint8_t *dst, uint32_t value);
```

- [ ] **Step 4: Run protocol tests**

Run:

```bash
python3 -m pytest -q tests/test_esp32p4_protocol_contract.py
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main/pq_frame_protocol.h tests/test_esp32p4_protocol_contract.py
git commit -m "feat: define esp32p4 frame protocol contract"
```

---

## Task 4: Implement CRC32 and Binary Serial Transport

**Files:**
- Create: `firmware/esp32p4/pq_firmware/main/pq_frame_protocol.c`
- Create: `firmware/esp32p4/pq_firmware/main/pq_serial.c`
- Create: `firmware/esp32p4/pq_firmware/main/pq_serial.h`
- Modify: `firmware/esp32p4/pq_firmware/main/CMakeLists.txt`
- Modify: `tests/test_esp32p4_protocol_contract.py`

- [ ] **Step 1: Add source presence contract test**

Append to `tests/test_esp32p4_protocol_contract.py`:

```python
def test_esp32p4_protocol_sources_exist() -> None:
    assert Path("firmware/esp32p4/pq_firmware/main/pq_frame_protocol.c").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_serial.c").exists()
    assert Path("firmware/esp32p4/pq_firmware/main/pq_serial.h").exists()
```

- [ ] **Step 2: Run failing source presence test**

Run:

```bash
python3 -m pytest -q tests/test_esp32p4_protocol_contract.py::test_esp32p4_protocol_sources_exist
```

Expected:

```text
FAILED tests/test_esp32p4_protocol_contract.py::test_esp32p4_protocol_sources_exist
```

- [ ] **Step 3: Add CRC implementation**

```c
// firmware/esp32p4/pq_firmware/main/pq_frame_protocol.c
#include "pq_frame_protocol.h"

uint32_t pq_crc32_le(const uint8_t *data, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

void pq_write_u32_be(uint8_t *dst, uint32_t value)
{
    dst[0] = (uint8_t)((value >> 24) & 0xFFu);
    dst[1] = (uint8_t)((value >> 16) & 0xFFu);
    dst[2] = (uint8_t)((value >> 8) & 0xFFu);
    dst[3] = (uint8_t)(value & 0xFFu);
}
```

- [ ] **Step 4: Add serial transport wrapper**

```c
// firmware/esp32p4/pq_firmware/main/pq_serial.h
#pragma once

#include <stddef.h>
#include <stdint.h>

void pq_serial_init(void);
void pq_serial_write(const uint8_t *data, size_t len);
```

```c
// firmware/esp32p4/pq_firmware/main/pq_serial.c
#include "pq_serial.h"

#include "driver/usb_serial_jtag.h"
#include "esp_check.h"
#include "esp_log.h"

static const char *TAG = "pq_serial";

void pq_serial_init(void)
{
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 8192,
        .rx_buffer_size = 1024,
    };
    esp_err_t err = usb_serial_jtag_driver_install(&cfg);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "usb_serial_jtag_driver_install failed: %s", esp_err_to_name(err));
    }
}

void pq_serial_write(const uint8_t *data, size_t len)
{
    size_t written = 0;
    while (written < len) {
        int chunk = usb_serial_jtag_write_bytes(
            (const char *)(data + written),
            len - written,
            pdMS_TO_TICKS(100)
        );
        if (chunk > 0) {
            written += (size_t)chunk;
        }
    }
}
```

- [ ] **Step 5: Update component sources**

Replace `firmware/esp32p4/pq_firmware/main/CMakeLists.txt` with:

```cmake
idf_component_register(
    SRCS
        "main.c"
        "pq_frame_protocol.c"
        "pq_serial.c"
    INCLUDE_DIRS "."
    REQUIRES esp_adc driver esp_timer esp_driver_usb_serial_jtag
)
```

- [ ] **Step 6: Run tests and compile**

Run:

```bash
python3 -m pytest -q tests/test_esp32p4_protocol_contract.py
./scripts/compile_esp32p4_firmware.sh
```

Expected:

```text
3 passed
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 7: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main tests/test_esp32p4_protocol_contract.py
git commit -m "feat: add esp32p4 frame transport primitives"
```

---

## Task 5: Implement ESP32-P4 Raw ADC Capture

**Files:**
- Create: `firmware/esp32p4/pq_firmware/main/pq_adc.c`
- Create: `firmware/esp32p4/pq_firmware/main/pq_adc.h`
- Modify: `firmware/esp32p4/pq_firmware/main/main.c`
- Modify: `firmware/esp32p4/pq_firmware/main/CMakeLists.txt`
- Modify: `docs/esp32p4_migration_notes.md`

- [ ] **Step 1: Create migration notes with fixed initial pin choices**

```markdown
# ESP32-P4 Migration Notes

## Board

- Board: Waveshare ESP32-P4-WIFI6
- Firmware target: `esp32p4`
- Initial voltage ADC pin: GPIO16 / ADC1 channel 0
- Initial current ADC pin: GPIO17 / ADC1 channel 1
- ADC mode: ESP-IDF continuous mode
- Per-channel sample rate: 5000 samples/s
- Total conversion rate: 10000 conversions/s because ESP-IDF continuous mode sequentially samples the configured channel group
- Frame length: 500 samples/channel

## Bench Validation Required Before Mains

- GPIO16 and GPIO17 must be confirmed on the board header with continuity/pinout inspection.
- With frontend disconnected, both ADC pins must measure 0 V to GND.
- With bias connected and mains disconnected, both ADC pins must be within 0.0 V to 3.3 V.
- With frontend connected, neither channel may clip at raw count 0 or 4095.
```

- [ ] **Step 2: Add ADC capture header**

```c
// firmware/esp32p4/pq_firmware/main/pq_adc.h
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "pq_frame_protocol.h"

void pq_adc_init(void);
bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES]);
```

- [ ] **Step 3: Add ADC continuous capture implementation**

```c
// firmware/esp32p4/pq_firmware/main/pq_adc.c
#include "pq_adc.h"

#include <string.h>

#include "esp_adc/adc_continuous.h"
#include "esp_check.h"
#include "esp_log.h"

static const char *TAG = "pq_adc";

static adc_continuous_handle_t s_adc = NULL;

void pq_adc_init(void)
{
    adc_continuous_handle_cfg_t handle_cfg = {
        .max_store_buf_size = 8192,
        .conv_frame_size = 512,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &s_adc));

    adc_digi_pattern_config_t pattern[2] = {
        {
            .atten = ADC_ATTEN_DB_12,
            .channel = ADC_CHANNEL_0,
            .unit = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
        {
            .atten = ADC_ATTEN_DB_12,
            .channel = ADC_CHANNEL_1,
            .unit = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
    };

    adc_continuous_config_t config = {
        .pattern_num = 2,
        .adc_pattern = pattern,
        .sample_freq_hz = 10000,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &config));
    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
    ESP_LOGI(TAG, "ADC continuous capture started: ADC1 channels 0/1 at 10000 conversions/s");
}

bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES])
{
    uint32_t v_count = 0;
    uint32_t i_count = 0;
    uint8_t buf[512];

    while (v_count < PQ_FRAME_SAMPLES || i_count < PQ_FRAME_SAMPLES) {
        uint32_t out_len = 0;
        esp_err_t err = adc_continuous_read(s_adc, buf, sizeof(buf), &out_len, pdMS_TO_TICKS(1000));
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "adc_continuous_read failed: %s", esp_err_to_name(err));
            return false;
        }

        for (uint32_t offset = 0; offset + SOC_ADC_DIGI_RESULT_BYTES <= out_len; offset += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t *)&buf[offset];
            uint32_t channel = sample->type2.channel;
            int16_t raw = (int16_t)(sample->type2.data & 0x0FFFu);

            if (channel == ADC_CHANNEL_0 && v_count < PQ_FRAME_SAMPLES) {
                v_raw[v_count++] = raw;
            } else if (channel == ADC_CHANNEL_1 && i_count < PQ_FRAME_SAMPLES) {
                i_raw[i_count++] = raw;
            }

            if (v_count >= PQ_FRAME_SAMPLES && i_count >= PQ_FRAME_SAMPLES) {
                return true;
            }
        }
    }
    return true;
}
```

- [ ] **Step 4: Update component sources**

Add `pq_adc.c` to `firmware/esp32p4/pq_firmware/main/CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS
        "main.c"
        "pq_adc.c"
        "pq_frame_protocol.c"
        "pq_serial.c"
    INCLUDE_DIRS "."
    REQUIRES esp_adc driver esp_timer esp_driver_usb_serial_jtag
)
```

- [ ] **Step 5: Compile**

Run:

```bash
./scripts/compile_esp32p4_firmware.sh
```

Expected:

```text
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 6: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main docs/esp32p4_migration_notes.md
git commit -m "feat: add esp32p4 continuous adc capture"
```

---

## Task 6: Emit Host-Compatible Raw Frames

**Files:**
- Modify: `firmware/esp32p4/pq_firmware/main/main.c`
- Create: `scripts/probe_esp32p4_raw.py`

- [ ] **Step 1: Replace ESP32-P4 main with raw frame emitter**

```c
// firmware/esp32p4/pq_firmware/main/main.c
#include <string.h>

#include "esp_log.h"

#include "pq_adc.h"
#include "pq_frame_protocol.h"
#include "pq_serial.h"

static const char *TAG = "pq_main";

static uint16_t s_seq = 0;
static int16_t s_v_raw[PQ_FRAME_SAMPLES];
static int16_t s_i_raw[PQ_FRAME_SAMPLES];
static uint8_t s_payload[PQ_RAW_PAYLOAD_BYTES];
static uint8_t s_frame[PQ_RAW_FRAME_BYTES];

static void put_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static void send_raw_frame(void)
{
    uint8_t *p = s_payload;
    put_u16_le(p, s_seq);
    p += 2;
    put_u16_le(p, (uint16_t)PQ_FRAME_SAMPLES);
    p += 2;
    memcpy(p, s_v_raw, PQ_FRAME_SAMPLES * sizeof(int16_t));
    p += PQ_FRAME_SAMPLES * sizeof(int16_t);
    memcpy(p, s_i_raw, PQ_FRAME_SAMPLES * sizeof(int16_t));

    pq_write_u32_be(s_frame, PQ_MAGIC);
    memcpy(s_frame + 4, s_payload, PQ_RAW_PAYLOAD_BYTES);

    uint32_t crc = pq_crc32_le(s_payload, PQ_RAW_PAYLOAD_BYTES);
    uint32_t crc_off = 4u + PQ_RAW_PAYLOAD_BYTES;
    s_frame[crc_off + 0u] = (uint8_t)(crc & 0xFFu);
    s_frame[crc_off + 1u] = (uint8_t)((crc >> 8) & 0xFFu);
    s_frame[crc_off + 2u] = (uint8_t)((crc >> 16) & 0xFFu);
    s_frame[crc_off + 3u] = (uint8_t)((crc >> 24) & 0xFFu);

    pq_serial_write(s_frame, PQ_RAW_FRAME_BYTES);
    s_seq++;
}

void app_main(void)
{
    ESP_LOGI(TAG, "Booting ESP32-P4 raw PQ firmware");
    pq_serial_init();
    pq_adc_init();

    while (true) {
        if (pq_adc_read_frame(s_v_raw, s_i_raw)) {
            send_raw_frame();
        }
    }
}
```

- [ ] **Step 2: Create raw probe script**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.io.serial_receiver import SerialFrameReceiver


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe ESP32-P4 raw PQ frames")
    parser.add_argument("--port", required=True)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    receiver = SerialFrameReceiver(args.port, mode="raw", timeout=1.0)
    receiver.open()
    try:
        for _ in range(args.frames):
            frame = receiver.read_frame(frame_timeout=2.0)
            if frame is None:
                print("frame=None")
                continue
            print(
                f"seq={frame.seq} "
                f"V[min={frame.v_raw.min()} max={frame.v_raw.max()} mean={frame.v_raw.mean():.1f}] "
                f"I[min={frame.i_raw.min()} max={frame.i_raw.max()} mean={frame.i_raw.mean():.1f}]"
            )
    finally:
        receiver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Compile**

Run:

```bash
./scripts/compile_esp32p4_firmware.sh
```

Expected:

```text
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 4: Flash and smoke-test raw frames**

Run on the Raspberry Pi with the ESP32-P4 connected:

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python scripts/probe_esp32p4_raw.py --port /dev/ttyACM0 --frames 5
```

Expected with bias connected and mains disconnected:

```text
seq=0 V[min=... max=... mean=...] I[min=... max=... mean=...]
```

Acceptance:

- No `frame=None`.
- Sequence increments.
- V and I arrays are not all zero.
- CRC failures remain `0`.

- [ ] **Step 5: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main scripts/probe_esp32p4_raw.py
git commit -m "feat: stream esp32p4 raw pq frames"
```

---

## Task 7: Verify Host Runtime Works Unchanged in Raw Mode

**Files:**
- Modify: `docs/esp32p4_migration_notes.md`

- [ ] **Step 1: Run raw live inference**

Run on the Pi:

```bash
python -m src.infer.live_infer \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode raw \
  --max-frames 10
```

Expected:

```text
[live] receiver_mode=raw port=/dev/ttyACM0
seq=... RMS_V=...
[live] scored_frames=10
```

- [ ] **Step 2: Record raw validation results**

Append to `docs/esp32p4_migration_notes.md`:

```markdown
## Raw Mode Validation

- Date: 2026-05-12
- Command: `python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode raw --max-frames 10`
- Result: 10 scored frames
- Receiver CRC failures: 0
- Voltage clipping status: record observed min/max from `scripts/probe_esp32p4_raw.py`
- Current clipping status: record observed min/max from `scripts/probe_esp32p4_raw.py`
```

Replace the two clipping-status lines with actual observed values before committing.

- [ ] **Step 3: Commit**

```bash
git add docs/esp32p4_migration_notes.md
git commit -m "docs: record esp32p4 raw runtime validation"
```

---

## Task 8: Measure ESP32-P4 Voltage/Current Channel Skew

**Files:**
- Create: `scripts/analyze_adc_pair_skew.py`
- Modify: `docs/esp32p4_migration_notes.md`

- [ ] **Step 1: Create skew analysis script**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from src.io.serial_receiver import SerialFrameReceiver


def estimate_lag_samples(v: np.ndarray, i: np.ndarray) -> int:
    v0 = v.astype(np.float64) - float(np.mean(v))
    i0 = i.astype(np.float64) - float(np.mean(i))
    corr = np.correlate(v0, i0, mode="full")
    return int(np.argmax(corr) - (len(v0) - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate ADC pair skew from matching signals on both channels")
    parser.add_argument("--port", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--fs-hz", type=float, default=5000.0)
    args = parser.parse_args()

    receiver = SerialFrameReceiver(args.port, mode="raw", timeout=1.0)
    lags: list[int] = []
    receiver.open()
    try:
        while len(lags) < args.frames:
            frame = receiver.read_frame(frame_timeout=2.0)
            if frame is None:
                continue
            lags.append(estimate_lag_samples(frame.v_raw, frame.i_raw))
    finally:
        receiver.close()

    lag_samples = np.asarray(lags, dtype=np.float64)
    lag_us = lag_samples * (1_000_000.0 / args.fs_hz)
    print(f"frames={len(lags)}")
    print(f"lag_samples_mean={lag_samples.mean():.3f}")
    print(f"lag_samples_max_abs={np.max(np.abs(lag_samples)):.3f}")
    print(f"lag_us_mean={lag_us.mean():.3f}")
    print(f"lag_us_max_abs={np.max(np.abs(lag_us)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run skew test with the same safe low-voltage waveform on both ADC inputs**

Run:

```bash
python scripts/analyze_adc_pair_skew.py --port /dev/ttyACM0 --frames 20
```

Acceptance:

- `lag_samples_max_abs <= 1.000` for classification/RMS/THD work.
- If phase or PF accuracy is a grading requirement, record the measured lag in microseconds and compensate in host DSP or move to an external simultaneous-sampling ADC.

- [ ] **Step 3: Record skew result**

Append to `docs/esp32p4_migration_notes.md`:

```markdown
## ADC Pair Skew Validation

- Test input: same low-voltage waveform fed into voltage and current ADC inputs
- Frames: 20
- Maximum absolute lag: record `lag_us_max_abs` from the script
- Decision: accepted for project classification path when max absolute lag is no more than one 5 kHz sample
```

Replace the lag line with the actual measured value before committing.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_adc_pair_skew.py docs/esp32p4_migration_notes.md
git commit -m "test: add esp32p4 adc channel skew check"
```

---

## Task 9: Port Teensy DSP to ESP32-P4 Model-Ready Firmware

**Files:**
- Create: `firmware/esp32p4/pq_firmware/main/dsp/dsp.cpp`
- Create: `firmware/esp32p4/pq_firmware/main/dsp/dsp.h`
- Create: `firmware/esp32p4/pq_firmware/main/dsp/goertzel.h`
- Create: `firmware/esp32p4/pq_firmware/main/dsp/dwt.h`
- Modify: `firmware/esp32p4/pq_firmware/main/CMakeLists.txt`
- Modify: `firmware/esp32p4/pq_firmware/main/main.c`

- [ ] **Step 1: Copy DSP sources from archived Teensy firmware**

Run:

```bash
mkdir -p firmware/esp32p4/pq_firmware/main/dsp
cp legacy/teensyfirmware/pq_firmware/src/dsp.cpp firmware/esp32p4/pq_firmware/main/dsp/dsp.cpp
cp legacy/teensyfirmware/pq_firmware/src/dsp.h firmware/esp32p4/pq_firmware/main/dsp/dsp.h
cp legacy/teensyfirmware/pq_firmware/src/goertzel.h firmware/esp32p4/pq_firmware/main/dsp/goertzel.h
cp legacy/teensyfirmware/pq_firmware/src/dwt.h firmware/esp32p4/pq_firmware/main/dsp/dwt.h
```

- [ ] **Step 2: Update include paths in ESP32-P4 CMake**

```cmake
idf_component_register(
    SRCS
        "main.c"
        "pq_adc.c"
        "pq_frame_protocol.c"
        "pq_serial.c"
        "dsp/dsp.cpp"
    INCLUDE_DIRS
        "."
        "dsp"
    REQUIRES esp_adc driver esp_timer esp_driver_usb_serial_jtag
)
```

- [ ] **Step 3: Compile and fix C++ portability errors only**

Run:

```bash
./scripts/compile_esp32p4_firmware.sh
```

Allowed fixes:

- include `<math.h>` or `<string.h>` where missing
- replace Teensy-only attributes with standard C/C++ equivalents
- keep all feature indices and calibration constants unchanged until raw parity proves a calibration update is needed

Expected:

```text
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 4: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main
git commit -m "feat: port pq dsp to esp32p4 firmware"
```

---

## Task 10: Emit ESP32-P4 Model-Ready `tflite` Frames

**Files:**
- Modify: `firmware/esp32p4/pq_firmware/main/main.c`
- Modify: `firmware/esp32p4/pq_firmware/main/pq_frame_protocol.h`

- [ ] **Step 1: Add model-ready buffers and frame packing**

Modify `main.c` to:

- keep raw buffers `s_v_raw` and `s_i_raw`
- add `float s_feat[298]`
- add `float s_v_norm[500]`
- add `float s_i_norm[500]`
- call `compute_model4_frame(s_v_raw, s_i_raw, s_feat, s_v_norm, s_i_norm)`
- pack the inference frame exactly as `src.io.frame_protocol.pack_inference_frame` does:
  - magic big-endian
  - `seq` little-endian
  - frame type `0x0003` little-endian
  - `v_norm[500]`
  - `i_norm[500]`
  - `feat[28:56]`
  - `feat[0:28]`
  - `feat[56:214]`
  - `feat[214:298]`
  - CRC32 little-endian over payload

Use the existing Teensy packing order in `legacy/teensyfirmware/pq_firmware/src/main.cpp` as the byte-for-byte reference.

- [ ] **Step 2: Add compile-time firmware mode selection**

Use CMake compile definitions:

```cmake
target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=0)
```

Then support raw builds by changing the definition to `PQ_RAW_MODE=1` during raw validation builds.

- [ ] **Step 3: Compile**

Run:

```bash
./scripts/compile_esp32p4_firmware.sh
```

Expected:

```text
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 4: Flash and run `tflite` receiver**

Run:

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python -m src.infer.live_infer \
  --port /dev/ttyACM0 \
  --config configs/default.yaml \
  --receiver-mode tflite \
  --max-frames 10
```

Expected:

```text
[live] receiver_mode=tflite port=/dev/ttyACM0
seq=... RMS_V=...
[live] scored_frames=10
```

Acceptance:

- 10 scored frames.
- `CRC failures: 0` in UI/system health or receiver stats.
- `RMS_V` and `THD_V` are within 5% of raw-mode host DSP for the same stable input after calibration is corrected.

- [ ] **Step 5: Commit**

```bash
git add firmware/esp32p4/pq_firmware/main
git commit -m "feat: stream esp32p4 tflite inference frames"
```

---

## Task 11: Add Hardware-in-Loop Raw-vs-TFLite Parity Check

**Files:**
- Create: `scripts/hil_compare_esp32p4_raw_tflite.md`
- Modify: `docs/esp32p4_migration_notes.md`

- [ ] **Step 1: Create HIL parity runbook**

```markdown
# ESP32-P4 Raw vs TFLite HIL Parity

## Purpose

Prove that ESP32-P4 onboard DSP produces metrics consistent with Pi-side host DSP before making `tflite` the default demo path.

## Step 1: Flash raw mode

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode raw --max-frames 20
```

Record:

- RMS_V mean
- RMS_I mean
- THD_V mean
- THD_I mean

## Step 2: Flash tflite mode

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite --max-frames 20
```

Record:

- RMS_V mean
- RMS_I mean
- THD_V mean
- THD_I mean

## Acceptance

- RMS_V difference <= 5%
- RMS_I difference <= 5%
- THD_V absolute difference <= 0.05
- THD_I absolute difference <= 0.05
- No CRC failures
- No serial parse failures
```
```

- [ ] **Step 2: Execute the runbook and record results**

Append measured results to `docs/esp32p4_migration_notes.md` under:

```markdown
## Raw vs Tflite Parity

- Raw RMS_V mean: measured value
- Tflite RMS_V mean: measured value
- Raw THD_V mean: measured value
- Tflite THD_V mean: measured value
- Decision: accepted when the differences meet the runbook acceptance limits
```

Replace each measured line with actual values before committing.

- [ ] **Step 3: Commit**

```bash
git add scripts/hil_compare_esp32p4_raw_tflite.md docs/esp32p4_migration_notes.md
git commit -m "docs: add esp32p4 raw tflite parity runbook"
```

---

## Task 12: Cut Over Documentation Without Breaking Legacy Teensy

**Files:**
- Modify: `README.md`
- Modify: `docs/demo_runbook.md`
- Modify: `docs/pi_deployment_runbook.md`
- Modify: `scripts/compile_teensy_firmware.sh`

- [ ] **Step 1: Update README firmware section**

Add this firmware section:

```markdown
## Firmware Targets

Active MCU target:

```bash
./scripts/compile_esp32p4_firmware.sh
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
```

Legacy Teensy firmware is preserved under:

```text
legacy/teensyfirmware/
```

The old Teensy build remains available for reference and emergency fallback:

```bash
./scripts/compile_teensy_firmware.sh
```
```
```

- [ ] **Step 2: Update Pi deployment runbook**

Add a pre-live validation block:

```markdown
## ESP32-P4 Pre-Live Validation

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python scripts/probe_esp32p4_raw.py --port /dev/ttyACM0 --frames 5
python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite --max-frames 10
```

Proceed to the UI only after the CLI reports scored frames and zero CRC failures.
```
```

- [ ] **Step 3: Update Teensy compile script message**

Change the final echo in `scripts/compile_teensy_firmware.sh` to:

```bash
echo "Legacy Teensy firmware compile succeeded. Active migration target is ESP32-P4."
```

- [ ] **Step 4: Run docs contract tests**

Run:

```bash
python3 -m pytest -q tests/test_runtime_contract.py tests/test_firmware_layout.py
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/demo_runbook.md docs/pi_deployment_runbook.md scripts/compile_teensy_firmware.sh
git commit -m "docs: document esp32p4 firmware cutover"
```

---

## Task 13: Final System Verification

**Files:**
- Modify: `docs/esp32p4_migration_notes.md`

- [ ] **Step 1: Run Python contract tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Compile ESP32-P4 firmware**

Run:

```bash
./scripts/compile_esp32p4_firmware.sh
```

Expected:

```text
ESP32-P4 firmware compile succeeded.
```

- [ ] **Step 3: Flash ESP32-P4 and verify CLI**

Run:

```bash
./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0
python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite --max-frames 20
```

Expected:

```text
[live] scored_frames=20
```

- [ ] **Step 4: Verify UI in windowed mode first**

Run:

```bash
python -m src.ui.app --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite --windowed
```

Acceptance:

- Active faults update.
- Waveforms update.
- Harmonic spectrum updates.
- Live metrics update.
- System health shows connected serial and zero CRC failures.

- [ ] **Step 5: Verify UI fullscreen**

Run:

```bash
python -m src.ui.app --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode tflite
```

Acceptance:

- Same as windowed mode.
- Dashboard remains readable at Pi display resolution.

- [ ] **Step 6: Record final verification**

Append to `docs/esp32p4_migration_notes.md`:

```markdown
## Final Verification

- Python tests: passed
- ESP32-P4 compile: passed
- Tflite CLI scored frames: 20
- UI windowed: passed
- UI fullscreen: passed
- CRC failures: 0
- Parse failures: 0
```

- [ ] **Step 7: Commit**

```bash
git add docs/esp32p4_migration_notes.md
git commit -m "docs: record esp32p4 final verification"
```

---

## Rollback Plan

If ESP32-P4 raw frames do not validate:

1. Keep Pi software unchanged.
2. Use `legacy/teensyfirmware/` to inspect the last working Teensy implementation.
3. Use `firmware/teensy/` only if a replacement Teensy or repaired USB path becomes available.
4. Continue demos with recorded replay logs using:

```bash
python -m src.infer.offline_replay --input artifacts/protocol_test_frames.bin --config configs/default.yaml
```

If ESP32-P4 raw mode works but `tflite` mode fails:

1. Ship with `--receiver-mode raw` temporarily.
2. Keep ESP32-P4 as the acquisition board.
3. Continue DSP parity work in Task 9 through Task 11.

## Self-Review

- Spec coverage: The plan preserves Teensy firmware, introduces ESP32-P4 raw capture, keeps the Pi runtime unchanged, ports model-ready frames later, validates channel skew, and documents cutover.
- Placeholder scan: The plan uses fixed initial ADC pins, exact commands, exact file paths, and explicit acceptance criteria. Measured values are deliberately recorded during hardware validation steps because they cannot be known before the ESP32-P4 board is wired and flashed.
- Type consistency: Raw frame constants match `src.io.frame_protocol`; model-ready frame packing order matches current Teensy packing and Pi reconstruction; runtime modes remain `raw` and `tflite`.
