#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "pq_frame_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

void pq_adc_init(void);
bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES]);

#ifdef __cplusplus
}
#endif
