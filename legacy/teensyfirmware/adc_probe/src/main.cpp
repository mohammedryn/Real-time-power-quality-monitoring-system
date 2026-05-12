#include <Arduino.h>

// ADC pins
static constexpr uint8_t PIN_V = A0;   // Voltage channel (pin 14)
static constexpr uint8_t PIN_I = A10;  // Current channel (pin 24)

// Calibration (matches dsp.h)
static constexpr float ADC_REF       = 3.3f;
static constexpr float ADC_MAX       = 4095.0f;
static constexpr float V_MIDPOINT    = 1985.0f;
static constexpr float V_SCALE       = 0.25745f;  // V_mains per ADC count
static constexpr float I_MIDPOINT    = 2048.0f;
static constexpr float I_SCALE       = 0.030518f; // Amps per ADC count

static constexpr int   N_SAMPLES     = 500;        // one 50 Hz frame at 5 kHz
static constexpr int   SAMPLE_US     = 200;        // 5 kHz

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogReadAveraging(4);
    delay(1000);
    Serial.println("=== ADC Probe (A0=Voltage  A10=Current) ===");
    Serial.println("Columns: v_min  v_max  v_mean  v_rms_V  i_min  i_max  i_mean  i_rms_A");
}

void loop() {
    int32_t v_sum = 0, i_sum = 0;
    int32_t v_sq  = 0, i_sq  = 0;
    int16_t v_min = 4095, v_max = 0;
    int16_t i_min = 4095, i_max = 0;

    // Collect N_SAMPLES at 5 kHz
    for (int k = 0; k < N_SAMPLES; k++) {
        uint32_t t0 = micros();

        int16_t v = (int16_t)analogRead(PIN_V);
        int16_t i = (int16_t)analogRead(PIN_I);

        v_sum += v;  i_sum += i;

        int32_t dv = v - (int32_t)V_MIDPOINT;
        int32_t di = i - (int32_t)I_MIDPOINT;
        v_sq += dv * dv;
        i_sq += di * di;

        if (v < v_min) v_min = v;
        if (v > v_max) v_max = v;
        if (i < i_min) i_min = i;
        if (i > i_max) i_max = i;

        // Busy-wait to hit 5 kHz
        while ((micros() - t0) < SAMPLE_US) {}
    }

    float v_mean_cts = (float)v_sum / N_SAMPLES;
    float i_mean_cts = (float)i_sum / N_SAMPLES;

    // RMS of AC component (after midpoint removal)
    float v_rms_cts = sqrtf((float)v_sq / N_SAMPLES);
    float i_rms_cts = sqrtf((float)i_sq / N_SAMPLES);

    float v_rms_V = v_rms_cts * V_SCALE;
    float i_rms_A = i_rms_cts * I_SCALE;

    // Print raw counts + physical values
    Serial.print("V: min="); Serial.print(v_min);
    Serial.print("  max="); Serial.print(v_max);
    Serial.print("  mean="); Serial.print(v_mean_cts, 1);
    Serial.print("  RMS="); Serial.print(v_rms_V, 2); Serial.print(" V");

    Serial.print("   |   I: min="); Serial.print(i_min);
    Serial.print("  max="); Serial.print(i_max);
    Serial.print("  mean="); Serial.print(i_mean_cts, 1);
    Serial.print("  RMS="); Serial.print(i_rms_A, 4); Serial.print(" A");

    Serial.println();
    delay(1000);
}
