#include <Arduino.h>
#include <ADC.h>
#include <FastCRC.h>
#include "dsp.h"

#ifndef PQ_BENCH_MODE
#define PQ_BENCH_MODE 0
#endif

// When defined, reverts to raw ADC frame (2012 bytes).
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
static constexpr uint16_t FRAME_SAMPLES   = 500;

// Zero-crossing settings (voltage channel)
static constexpr int16_t ADC_MIDPOINT  = 2071;
static constexpr int16_t ZC_HYSTERESIS = 20;

// ---- Frame protocol constants ----
static constexpr uint32_t MAGIC = 0xDEADBEEF;

// Model-ready frame type identifier (occupies the same 2-byte field as n/n_feat
// in legacy frames, but uses a value that doesn't collide with N_SAMPLES=500
// or N_FEATURES=282).
static constexpr uint16_t MODEL_READY_FRAME_TYPE = 0x0003;

// X_phase = feat[0:28] ++ feat[56:214] ++ feat[214:298]  = 270 floats
static constexpr int XWAVE_FLOATS  = 1000;   // v_norm[500] + i_norm[500]
static constexpr int XMAG_FLOATS   = 28;
static constexpr int XPHASE_FLOATS = 270;

// Model-ready payload (after magic):
//   seq(2) + type(2) + X_wave(4000) + X_mag(112) + X_phase(1080) = 5196 bytes
static constexpr uint16_t MODEL_PAYLOAD_BYTES =
    2 + 2 + (XWAVE_FLOATS * 4) + (XMAG_FLOATS * 4) + (XPHASE_FLOATS * 4);
// Full frame = magic(4) + payload(5196) + CRC(4) = 5204 bytes
static constexpr uint16_t MODEL_FRAME_BYTES = 4 + MODEL_PAYLOAD_BYTES + 4;

#if PQ_RAW_MODE
static constexpr uint16_t PAYLOAD_BYTES =
    2 + 2 + (FRAME_SAMPLES * 2) + (FRAME_SAMPLES * 2);
static constexpr uint16_t FRAME_BYTES = 4 + PAYLOAD_BYTES + 4;
#endif

ADC*          adc = new ADC();
FastCRC32     crc32;
IntervalTimer sampleTimer;

volatile int16_t  v_buf[FRAME_SAMPLES];
volatile int16_t  i_buf[FRAME_SAMPLES];
volatile bool     windowReady = false;
volatile bool     collecting  = false;
volatile uint16_t sampleCount = 0;
volatile int16_t  prevV = ADC_MIDPOINT;

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
            collecting  = false;
            windowReady = true;
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
    memcpy(payload,                            &txSeq,  2);
    memcpy(payload + 2,                        &n,      2);
    memcpy(payload + 4,                        v_local, FRAME_SAMPLES * 2);
    memcpy(payload + 4 + FRAME_SAMPLES * 2,    i_local, FRAME_SAMPLES * 2);

    const uint32_t crc = crc32.crc32(payload, PAYLOAD_BYTES);

    write_u32_be(MAGIC);
    Serial.write(payload, PAYLOAD_BYTES);
    Serial.write(reinterpret_cast<const uint8_t*>(&crc), 4);
    Serial.send_now();

    frameSeq++;
}

#else  // default: model-ready frame

// Static output buffers (avoid stack pressure on M7)
static float feat_buf[N_FEATURES];        // 298 features
static float v_norm_buf[N_WAVE_SAMPLES];  // 500 peak-normalised V
static float i_norm_buf[N_WAVE_SAMPLES];  // 500 peak-normalised I

// X_phase = feat[0:28] ++ feat[56:214] ++ feat[214:298]
// Section A: feat[0:28]    -> 28 floats  (time-domain stats)
// Section B: feat[56:214]  -> 158 floats (phase_self + phase_cross + phase_rel + power_harm + circ)
// Section C: feat[214:298] -> 84 floats  (wavelet)
// Total X_phase: 28 + 158 + 84 = 270 floats
static constexpr int XPHASE_A_OFF = 0,   XPHASE_A_LEN = 28;
static constexpr int XPHASE_B_OFF = 56,  XPHASE_B_LEN = 158;
static constexpr int XPHASE_C_OFF = 214, XPHASE_C_LEN = 84;
// X_mag  = feat[28:56] = 28 floats
static constexpr int XMAG_OFF = 28;

static void sendModelReadyFrame() {
    static int16_t v_local[FRAME_SAMPLES];
    static int16_t i_local[FRAME_SAMPLES];

    // Copy volatile ADC buffers under interrupt lock
    noInterrupts();
    for (uint16_t k = 0; k < FRAME_SAMPLES; k++) {
        v_local[k] = v_buf[k];
        i_local[k] = i_buf[k];
    }
    interrupts();

#if PQ_DEBUG_TIMING
    uint32_t t0 = micros();
#endif

    compute_model4_frame(v_local, i_local, feat_buf, v_norm_buf, i_norm_buf);

#if PQ_DEBUG_TIMING
    uint32_t t_dsp = micros() - t0;
#endif

    // Build payload:
    //   [seq 2B LE][type 2B LE = 0x0003]
    //   [v_norm 2000B][i_norm 2000B]   <- X_wave (1000 floats)
    //   [X_mag 112B]                   <- feat[28:56] (28 floats)
    //   [X_phase 1080B]                <- feat[0:28]++feat[56:214]++feat[214:298] (270 floats)
    const uint16_t txSeq = frameSeq;
    const uint16_t ftype = MODEL_READY_FRAME_TYPE;

    static uint8_t payload[MODEL_PAYLOAD_BYTES];
    uint8_t* p = payload;

    memcpy(p, &txSeq,          2); p += 2;
    memcpy(p, &ftype,          2); p += 2;
    memcpy(p, v_norm_buf,      N_WAVE_SAMPLES * 4); p += N_WAVE_SAMPLES * 4;
    memcpy(p, i_norm_buf,      N_WAVE_SAMPLES * 4); p += N_WAVE_SAMPLES * 4;
    memcpy(p, feat_buf + XMAG_OFF, XMAG_FLOATS * 4); p += XMAG_FLOATS * 4;
    memcpy(p, feat_buf + XPHASE_A_OFF, XPHASE_A_LEN * 4); p += XPHASE_A_LEN * 4;
    memcpy(p, feat_buf + XPHASE_B_OFF, XPHASE_B_LEN * 4); p += XPHASE_B_LEN * 4;
    memcpy(p, feat_buf + XPHASE_C_OFF, XPHASE_C_LEN * 4); p += XPHASE_C_LEN * 4;
    // p should now be payload + MODEL_PAYLOAD_BYTES

    const uint32_t crc = crc32.crc32(payload, MODEL_PAYLOAD_BYTES);

    write_u32_be(MAGIC);
    Serial.write(payload, MODEL_PAYLOAD_BYTES);
    Serial.write(reinterpret_cast<const uint8_t*>(&crc), 4);
    Serial.send_now();

#if PQ_DEBUG_TIMING
    uint32_t t_total = micros() - t0;
    Serial.print("#TIMING dsp_us=");
    Serial.print(t_dsp);
    Serial.print(" total_us=");
    Serial.println(t_total);
#endif

    frameSeq++;
}
#endif  // !PQ_RAW_MODE

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
        sendModelReadyFrame();
#endif
        windowReady = false;
    }
}
