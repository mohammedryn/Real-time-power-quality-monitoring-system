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

## Raw Mode Validation

- Date: PENDING
- Command: `PQ_RAW_MODE=1 ./scripts/compile_esp32p4_firmware.sh && ./scripts/flash_esp32p4_firmware.sh /dev/ttyACM0 && python -m src.infer.live_infer --port /dev/ttyACM0 --config configs/default.yaml --receiver-mode raw --max-frames 10`
- Result: PENDING
- Receiver CRC failures: PENDING
- Voltage clipping status: replace with observed min/max from `scripts/probe_esp32p4_raw.py`
- Current clipping status: replace with observed min/max from `scripts/probe_esp32p4_raw.py`

## ADC Pair Skew Validation

- Test input: same low-voltage waveform fed into voltage and current ADC inputs
- Frames: 20
- Maximum absolute lag: replace with `lag_us_max_abs` from `scripts/analyze_adc_pair_skew.py`
- Decision: accepted for project classification path when max absolute lag is no more than one 5 kHz sample

## Raw vs Tflite Parity

- Raw RMS_V mean: PENDING
- Tflite RMS_V mean: PENDING
- Raw THD_V mean: PENDING
- Tflite THD_V mean: PENDING
- Decision: accepted when the differences meet the runbook acceptance limits

## Final Verification

- Python tests: PENDING
- ESP32-P4 compile: PENDING
- Tflite CLI scored frames: PENDING
- UI windowed: PENDING
- UI fullscreen: PENDING
- CRC failures: PENDING
- Parse failures: PENDING
