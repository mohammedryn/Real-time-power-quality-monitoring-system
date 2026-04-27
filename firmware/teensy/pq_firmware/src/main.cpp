#include <Arduino.h>
#include <ADC.h>
#include <FastCRC.h>
#include "dsp.h"

#ifndef PQ_BENCH_MODE
#define PQ_BENCH_MODE 0
#endif

// When defined, reverts to raw ADC frame (2012 bytes) instead of feature frame.
// Useful for recording training data and hardware-in-the-loop parity validation.
#ifndef PQ_RAW_MODE
#define PQ_RAW_MODE 0
#endif

// When defined, emits DSP and total latency over Serial after each frame.
#ifndef PQ_DEBUG_TIMING
#define PQ_DEBUG_TIMING 0
#endif

// ---- Compile-time hardware constants (Teensy 4.1) ----
static constexpr uint8_t PIN_VOLTAGE_ADC0 = A0;   // Pin 14 -> ADC0
static constexpr uint8_t PIN_CURRENT_ADC1 = A10;  // Pin 24 -> ADC1

static constexpr uint32_t SAMPLE_RATE_HZ  = 5000;
static constexpr uint32_t SAMPLE_PERIOD_US = 200;  // 1e6 / 5000
static constexpr uint16_t FRAME_SAMPLES   = 500;   // one 50 Hz cycle at 5 kHz

// Zero-crossing settings (voltage channel)
static constexpr int16_t ADC_MIDPOINT  = 2071;
static constexpr int16_t ZC_HYSTERESIS = 20;

// ---- Frame protocol constants ----
static constexpr uint32_t MAGIC = 0xDEADBEEF;

#if PQ_RAW_MODE
// Raw frame: [magic BE][seq LE][N LE][v_raw 1000B][i_raw 1000B][crc32 LE]
static constexpr uint16_t PAYLOAD_BYTES =
    2 + 2 + (FRAME_SAMPLES * 2) + (FRAME_SAMPLES * 2);
static constexpr uint16_t FRAME_BYTES = 4 + PAYLOAD_BYTES + 4;
#else
// Feature frame: [magic BE][seq LE][n_feat LE][features 1128B][crc32 LE]
static constexpr uint16_t FEAT_PAYLOAD_BYTES = 2 + 2 + (N_FEATURES * 4);
static constexpr uint16_t FEAT_FRAME_BYTES   = 4 + FEAT_PAYLOAD_BYTES + 4;
#endif

ADC*        adc = new ADC();
FastCRC32   crc32;
IntervalTimer sampleTimer;

volatile int16_t v_buf[FRAME_SAMPLES];
volatile int16_t i_buf[FRAME_SAMPLES];
volatile bool    windowReady = false;
volatile bool    collecting  = false;
volatile uint16_t sampleCount = 0;
volatile int16_t prevV = ADC_MIDPOINT;

static uint16_t frameSeq = 0;

static inline void write_u32_be(uint32_t value) {
    uint8_t b[4];
    b[0] = (value >> 24) & 0xFF;
    b[1] = (value >> 16) & 0xFF;
    b[2] = (value >>  8) & 0xFF;
    b[3] =  value        & 0xFF;
    Serial.write(b, 4);
}

void setupADC() {
    adc->adc0->setAveraging(4);
    adc->adc0->setResolution(12);
    adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
    adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
    adc->adc0->setReference(ADC_REFERENCE::REF_3V3);

    adc->adc1->setAveraging(4);
    adc->adc1->setResolution(12);
    adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
    adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
    adc->adc1->setReference(ADC_REFERENCE::REF_3V3);
}

void FASTRUN sampleISR() {
    ADC::Sync_result result = adc->readSynchronizedSingle();
    int16_t v = static_cast<int16_t>(result.result_adc0);
    int16_t i = static_cast<int16_t>(result.result_adc1);

    adc->startSynchronizedSingleRead(PIN_VOLTAGE_ADC0, PIN_CURRENT_ADC1);

#if PQ_BENCH_MODE
    if (!collecting && !windowReady) {
        collecting  = true;
        sampleCount = 0;
    }
#else
    bool risingZC =
        (prevV <  (ADC_MIDPOINT - ZC_HYSTERESIS)) &&
        (v     >= (ADC_MIDPOINT + ZC_HYSTERESIS));

    if (!collecting && !windowReady && risingZC) {
        collecting  = true;
        sampleCount = 0;
    }
#endif

    if (collecting) {
        v_buf[sampleCount] = v;
        i_buf[sampleCount] = i;
        sampleCount++;
        if (sampleCount >= FRAME_SAMPLES) {
            collecting   = false;
            windowReady  = true;
        }
    }

    prevV = v;
}

// ---- Frame transmission -----------------------------------------------------

#if PQ_RAW_MODE
static void sendRawFrame() {
    int16_t v_local[FRAME_SAMPLES];
    int16_t i_local[FRAME_SAMPLES];

    noInterrupts();
    for (uint16_t k = 0; k < FRAME_SAMPLES; k++) {
        v_local[k] = v_buf[k];
        i_local[k] = i_buf[k];
    }
    interrupts();

    const uint16_t txSeq = frameSeq;
    const uint16_t n     = FRAME_SAMPLES;

    uint8_t payload[PAYLOAD_BYTES];
    memcpy(payload,                            &txSeq,   2);
    memcpy(payload + 2,                        &n,       2);
    memcpy(payload + 4,                        v_local,  FRAME_SAMPLES * 2);
    memcpy(payload + 4 + FRAME_SAMPLES * 2,    i_local,  FRAME_SAMPLES * 2);

    const uint32_t crc = crc32.crc32(payload, PAYLOAD_BYTES);

    write_u32_be(MAGIC);
    Serial.write(payload, PAYLOAD_BYTES);
    Serial.write(reinterpret_cast<const uint8_t*>(&crc), 4);
    Serial.send_now();

    frameSeq++;
}
#else
static float feat_buf[N_FEATURES];

static void sendFeatureFrame() {
    // Copy volatile ADC buffers under interrupt lock
    static int16_t v_local[FRAME_SAMPLES];
    static int16_t i_local[FRAME_SAMPLES];

    noInterrupts();
    for (uint16_t k = 0; k < FRAME_SAMPLES; k++) {
        v_local[k] = v_buf[k];
        i_local[k] = i_buf[k];
    }
    interrupts();

#if PQ_DEBUG_TIMING
    uint32_t t0 = micros();
#endif

    compute_features(v_local, i_local, feat_buf);

#if PQ_DEBUG_TIMING
    uint32_t t_dsp = micros() - t0;
#endif

    // Build payload: [seq LE 2B][n_feat LE 2B][features 1128B]
    const uint16_t txSeq   = frameSeq;
    const uint16_t n_feat  = (uint16_t)N_FEATURES;

    uint8_t payload[FEAT_PAYLOAD_BYTES];
    memcpy(payload,     &txSeq,    2);
    memcpy(payload + 2, &n_feat,   2);
    memcpy(payload + 4, feat_buf,  N_FEATURES * 4);

    const uint32_t crc = crc32.crc32(payload, FEAT_PAYLOAD_BYTES);

    write_u32_be(MAGIC);
    Serial.write(payload, FEAT_PAYLOAD_BYTES);
    Serial.write(reinterpret_cast<const uint8_t*>(&crc), 4);
    Serial.send_now();

#if PQ_DEBUG_TIMING
    uint32_t t_total = micros() - t0;
    // Emit timing over serial as a simple CSV line prefixed with '#'
    // so the host parser ignores it (magic-byte sync skips non-frame bytes).
    Serial.print("#TIMING dsp_us=");
    Serial.print(t_dsp);
    Serial.print(" total_us=");
    Serial.println(t_total);
    // Hard requirement: t_total must be well under 100,000 us (one frame period).
#endif

    frameSeq++;
}
#endif

// ---- Arduino entry points ---------------------------------------------------

void setup() {
    Serial.begin(0);  // USB CDC; baud rate ignored
    setupADC();
    adc->startSynchronizedSingleRead(PIN_VOLTAGE_ADC0, PIN_CURRENT_ADC1);
    sampleTimer.begin(sampleISR, SAMPLE_PERIOD_US);
}

void loop() {
    if (windowReady) {
#if PQ_RAW_MODE
        sendRawFrame();
#else
        sendFeatureFrame();
#endif
        windowReady = false;
    }
}
