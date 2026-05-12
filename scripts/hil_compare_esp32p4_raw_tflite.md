# ESP32-P4 Raw vs Tflite HIL Parity

## Purpose

Prove that ESP32-P4 onboard DSP produces metrics consistent with Pi-side host DSP before making `tflite` the default demo path.

## Step 1: Flash raw mode

```bash
PQ_RAW_MODE=1 ./scripts/compile_esp32p4_firmware.sh
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
PQ_RAW_MODE=0 ./scripts/compile_esp32p4_firmware.sh
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
