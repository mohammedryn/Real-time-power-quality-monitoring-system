#include <Arduino.h>
#include <ADC.h>
#include <FastCRC.h>

// ---- Compile-time hardware constants (Teensy 4.1) ----
// ADC pins for synchronized dual-channel acquisition
static constexpr uint8_t PIN_VOLTAGE_ADC0 = A0;   // Pin 14 -> ADC0
static constexpr uint8_t PIN_CURRENT_ADC1 = A10;  // Pin 24 -> ADC1

// Sampling and framing constants
static constexpr uint32_t SAMPLE_RATE_HZ = 5000;
static constexpr uint32_t SAMPLE_PERIOD_US = 200;  // 1e6 / 5000
static constexpr uint16_t FRAME_SAMPLES = 500;     // Exactly one 50 Hz cycle at 5 kHz

// Zero-crossing settings (voltage channel)
static constexpr int16_t ADC_MIDPOINT = 2071;      // 1.668 V midpoint in 12-bit counts
static constexpr int16_t ZC_HYSTERESIS = 20;       // Noise margin for robust ZC detection

// Frame protocol constants
static constexpr uint32_t MAGIC = 0xDEADBEEF;
static constexpr uint16_t PAYLOAD_BYTES = 2 + 2 + (FRAME_SAMPLES * 2) + (FRAME_SAMPLES * 2);  // seq + N + v + i
static constexpr uint16_t FRAME_BYTES = 4 + PAYLOAD_BYTES + 4;  // magic + payload + crc32

ADC* adc = new ADC();
FastCRC32 crc32;
IntervalTimer sampleTimer;

volatile int16_t v_buf[FRAME_SAMPLES];
volatile int16_t i_buf[FRAME_SAMPLES];
volatile bool windowReady = false;
volatile bool collecting = false;
volatile uint16_t sampleCount = 0;
volatile int16_t prevV = ADC_MIDPOINT;

static uint16_t frameSeq = 0;

static inline void write_u32_be(uint32_t value) {
  uint8_t b[4];
  b[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
  b[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
  b[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
  b[3] = static_cast<uint8_t>(value & 0xFF);
  Serial.write(b, 4);
}

void setupADC() {
  // ADC0: voltage channel
  adc->adc0->setAveraging(4);
  adc->adc0->setResolution(12);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
  adc->adc0->setReference(ADC_REFERENCE::REF_3V3);

  // ADC1: current channel
  adc->adc1->setAveraging(4);
  adc->adc1->setResolution(12);
  adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
  adc->adc1->setReference(ADC_REFERENCE::REF_3V3);
}

void FASTRUN sampleISR() {
  // Synchronized ADC read ensures zero inter-channel sample offset
  ADC::Sync_result result = adc->readSynchronizedSingle();
  int16_t v = static_cast<int16_t>(result.result_adc0);
  int16_t i = static_cast<int16_t>(result.result_adc1);

  bool risingZC =
      (prevV < (ADC_MIDPOINT - ZC_HYSTERESIS)) &&
      (v >= (ADC_MIDPOINT + ZC_HYSTERESIS));

  if (!collecting && !windowReady && risingZC) {
    collecting = true;
    sampleCount = 0;
  }

  if (collecting) {
    v_buf[sampleCount] = v;
    i_buf[sampleCount] = i;
    sampleCount++;

    if (sampleCount >= FRAME_SAMPLES) {
      collecting = false;
      windowReady = true;
    }
  }

  prevV = v;
}

void sendFrame() {
  int16_t v_local[FRAME_SAMPLES];
  int16_t i_local[FRAME_SAMPLES];

  noInterrupts();
  for (uint16_t idx = 0; idx < FRAME_SAMPLES; idx++) {
    v_local[idx] = v_buf[idx];
    i_local[idx] = i_buf[idx];
  }
  interrupts();

  const uint16_t txSeq = frameSeq;
  const uint16_t n = FRAME_SAMPLES;

  uint8_t payload[PAYLOAD_BYTES];
  memcpy(payload, &txSeq, 2);
  memcpy(payload + 2, &n, 2);
  memcpy(payload + 4, v_local, FRAME_SAMPLES * 2);
  memcpy(payload + 4 + (FRAME_SAMPLES * 2), i_local, FRAME_SAMPLES * 2);

  const uint32_t crc = crc32.crc32(payload, PAYLOAD_BYTES);

  write_u32_be(MAGIC);
  Serial.write(payload, PAYLOAD_BYTES);
  Serial.write(reinterpret_cast<const uint8_t*>(&crc), 4);  // little-endian CRC
  Serial.send_now();

  frameSeq++;
}

void setup() {
  Serial.begin(0);  // USB CDC, baud ignored
  setupADC();

  adc->startSynchronizedSingleRead(PIN_VOLTAGE_ADC0, PIN_CURRENT_ADC1);
  sampleTimer.begin(sampleISR, SAMPLE_PERIOD_US);
}

void loop() {
  if (windowReady) {
    sendFrame();
    windowReady = false;
  }
}
