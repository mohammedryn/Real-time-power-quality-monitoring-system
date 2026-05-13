#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PQ_MAGIC 0xDEADBEEFu
#define PQ_FRAME_SAMPLES 500u
#define PQ_RAW_PAYLOAD_BYTES 2004u
#define PQ_RAW_FRAME_BYTES 2012u

#define PQ_INFERENCE_FRAME_TYPE 0x0003u
#define PQ_XWAVE_FLOATS 1000u
#define PQ_XMAG_FLOATS 28u
#define PQ_XPHASE_FLOATS 270u
#define PQ_INFERENCE_PAYLOAD_BYTES 5196u
#define PQ_INFERENCE_FRAME_BYTES 5204u

typedef struct {
    uint16_t seq;
    uint16_t n;
    int16_t v_raw[PQ_FRAME_SAMPLES];
    int16_t i_raw[PQ_FRAME_SAMPLES];
} pq_raw_payload_t;

uint32_t pq_crc32_le(const uint8_t *data, uint32_t len);
void pq_write_u32_be(uint8_t *dst, uint32_t value);

#ifdef __cplusplus
}
#endif
