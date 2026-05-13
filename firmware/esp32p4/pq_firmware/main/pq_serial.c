#include "pq_serial.h"

#include "driver/usb_serial_jtag.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "pq_serial";

void pq_serial_init(void)
{
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 8192,
        .rx_buffer_size = 1024,
    };
    esp_err_t err = usb_serial_jtag_driver_install(&cfg);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "usb_serial_jtag_driver_install failed: %s", esp_err_to_name(err));
    }
}

void pq_serial_write(const uint8_t *data, size_t len)
{
    size_t written = 0;
    while (written < len) {
        int chunk = usb_serial_jtag_write_bytes(
            (const char *)(data + written),
            len - written,
            pdMS_TO_TICKS(100)
        );
        if (chunk > 0) {
            written += (size_t)chunk;
        } else {
            break; /* host not reading — drop remainder of frame */
        }
    }
}
