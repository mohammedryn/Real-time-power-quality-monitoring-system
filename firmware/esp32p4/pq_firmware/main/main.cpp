#include <string.h>

#include "esp_log.h"

#include "dsp.h"
#include "pq_adc.h"
#include "pq_frame_protocol.h"
#include "pq_serial.h"

#ifndef PQ_RAW_MODE
#define PQ_RAW_MODE 0
#endif

static const char *TAG = "pq_main";

static uint16_t s_seq = 0;
static int16_t s_v_raw[PQ_FRAME_SAMPLES];
static int16_t s_i_raw[PQ_FRAME_SAMPLES];

#if PQ_RAW_MODE
static uint8_t s_payload[PQ_RAW_PAYLOAD_BYTES];
static uint8_t s_frame[PQ_RAW_FRAME_BYTES];
#else
static float s_feat[N_FEATURES];
static float s_v_norm[N_WAVE_SAMPLES];
static float s_i_norm[N_WAVE_SAMPLES];
static uint8_t s_payload[PQ_INFERENCE_PAYLOAD_BYTES];
static uint8_t s_frame[PQ_INFERENCE_FRAME_BYTES];

static constexpr int XMAG_OFF = 28;
static constexpr int XPHASE_A_OFF = 0;
static constexpr int XPHASE_A_LEN = 28;
static constexpr int XPHASE_B_OFF = 56;
static constexpr int XPHASE_B_LEN = 158;
static constexpr int XPHASE_C_OFF = 214;
static constexpr int XPHASE_C_LEN = 84;
#endif

static void put_u16_le(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
}

#if PQ_RAW_MODE
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
#else
static void send_model_ready_frame(void)
{
    uint8_t *p = s_payload;

    compute_model4_frame(s_v_raw, s_i_raw, s_feat, s_v_norm, s_i_norm);

    put_u16_le(p, s_seq);
    p += 2;
    put_u16_le(p, (uint16_t)PQ_INFERENCE_FRAME_TYPE);
    p += 2;

    memcpy(p, s_v_norm, N_WAVE_SAMPLES * sizeof(float));
    p += N_WAVE_SAMPLES * sizeof(float);
    memcpy(p, s_i_norm, N_WAVE_SAMPLES * sizeof(float));
    p += N_WAVE_SAMPLES * sizeof(float);
    memcpy(p, s_feat + XMAG_OFF, PQ_XMAG_FLOATS * sizeof(float));
    p += PQ_XMAG_FLOATS * sizeof(float);
    memcpy(p, s_feat + XPHASE_A_OFF, XPHASE_A_LEN * sizeof(float));
    p += XPHASE_A_LEN * sizeof(float);
    memcpy(p, s_feat + XPHASE_B_OFF, XPHASE_B_LEN * sizeof(float));
    p += XPHASE_B_LEN * sizeof(float);
    memcpy(p, s_feat + XPHASE_C_OFF, XPHASE_C_LEN * sizeof(float));

    pq_write_u32_be(s_frame, PQ_MAGIC);
    memcpy(s_frame + 4, s_payload, PQ_INFERENCE_PAYLOAD_BYTES);

    uint32_t crc = pq_crc32_le(s_payload, PQ_INFERENCE_PAYLOAD_BYTES);
    uint32_t crc_off = 4u + PQ_INFERENCE_PAYLOAD_BYTES;
    s_frame[crc_off + 0u] = (uint8_t)(crc & 0xFFu);
    s_frame[crc_off + 1u] = (uint8_t)((crc >> 8) & 0xFFu);
    s_frame[crc_off + 2u] = (uint8_t)((crc >> 16) & 0xFFu);
    s_frame[crc_off + 3u] = (uint8_t)((crc >> 24) & 0xFFu);

    pq_serial_write(s_frame, PQ_INFERENCE_FRAME_BYTES);
    s_seq++;
}
#endif

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Booting ESP32-P4 PQ firmware");
    pq_serial_init();
    pq_adc_init();

    while (true) {
        if (pq_adc_read_frame(s_v_raw, s_i_raw)) {
#if PQ_RAW_MODE
            send_raw_frame();
#else
            send_model_ready_frame();
#endif
        }
    }
}
