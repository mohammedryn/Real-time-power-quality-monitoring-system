#include "pq_adc.h"

#include "esp_adc/adc_continuous.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "soc/soc_caps.h"

static const char *TAG = "pq_adc";

static adc_continuous_handle_t s_adc = NULL;

void pq_adc_init(void)
{
    adc_continuous_handle_cfg_t handle_cfg = {
        .max_store_buf_size = 8192,
        .conv_frame_size = 512,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &s_adc));

    adc_digi_pattern_config_t pattern[2] = {
        {
            .atten = ADC_ATTEN_DB_12,
            .channel = ADC_CHANNEL_0,
            .unit = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
        {
            .atten = ADC_ATTEN_DB_12,
            .channel = ADC_CHANNEL_1,
            .unit = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
    };

    adc_continuous_config_t config = {
        .pattern_num = 2,
        .adc_pattern = pattern,
        .sample_freq_hz = 10000,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &config));
    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
    ESP_LOGI(TAG, "ADC continuous capture started: ADC1 channels 0/1 at 10000 conversions/s");
}

bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES])
{
    uint32_t v_count = 0;
    uint32_t i_count = 0;
    uint8_t buf[512];

    while (v_count < PQ_FRAME_SAMPLES || i_count < PQ_FRAME_SAMPLES) {
        uint32_t out_len = 0;
        esp_err_t err = adc_continuous_read(s_adc, buf, sizeof(buf), &out_len, pdMS_TO_TICKS(1000));
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "adc_continuous_read failed: %s", esp_err_to_name(err));
            return false;
        }

        for (uint32_t offset = 0; offset + SOC_ADC_DIGI_RESULT_BYTES <= out_len; offset += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t *)&buf[offset];
            uint32_t channel = sample->type2.channel;
            int16_t raw = (int16_t)(sample->type2.data & 0x0FFFu);

            if (channel == ADC_CHANNEL_0 && v_count < PQ_FRAME_SAMPLES) {
                v_raw[v_count++] = raw;
            } else if (channel == ADC_CHANNEL_1 && i_count < PQ_FRAME_SAMPLES) {
                i_raw[i_count++] = raw;
            }

            if (v_count >= PQ_FRAME_SAMPLES && i_count >= PQ_FRAME_SAMPLES) {
                return true;
            }
        }
    }

    return true;
}
