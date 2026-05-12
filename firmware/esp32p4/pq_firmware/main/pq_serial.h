#pragma once

#include <stddef.h>
#include <stdint.h>

void pq_serial_init(void);
void pq_serial_write(const uint8_t *data, size_t len);
