#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pq_serial_init(void);
void pq_serial_write(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif
