#include "pq_adc.h"

#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"

static const char *TAG = "pq_adc";

static adc_continuous_handle_t s_adc = NULL;
static TaskHandle_t s_task_handle = NULL;
static volatile uint32_t s_conv_done_count = 0;

static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle,
                                      const adc_continuous_evt_data_t *edata,
                                      void *user_data)
{
    s_conv_done_count++;
    BaseType_t must_yield = pdFALSE;
    vTaskNotifyGiveFromISR(s_task_handle, &must_yield);
    return must_yield == pdTRUE;
}

/* Read a few oneshot samples to verify GPIO20/21 ADC hardware works at all. */
static void adc_oneshot_sanity_check(void)
{
    adc_oneshot_unit_handle_t adc_os = NULL;
    adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = ADC_UNIT_1 };
    if (adc_oneshot_new_unit(&unit_cfg, &adc_os) != ESP_OK) {
        ESP_LOGE(TAG, "oneshot: failed to create unit");
        return;
    }
    adc_oneshot_chan_cfg_t ch_cfg = {
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    adc_oneshot_config_channel(adc_os, ADC_CHANNEL_4, &ch_cfg);
    adc_oneshot_config_channel(adc_os, ADC_CHANNEL_5, &ch_cfg);

    for (int i = 0; i < 5; i++) {
        int raw4 = -1, raw5 = -1;
        adc_oneshot_read(adc_os, ADC_CHANNEL_4, &raw4);
        adc_oneshot_read(adc_os, ADC_CHANNEL_5, &raw5);
        ESP_LOGI(TAG, "oneshot[%d]: ch4=%d ch5=%d", i, raw4, raw5);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    adc_oneshot_del_unit(adc_os);
}

void pq_adc_init(void)
{
    s_task_handle = xTaskGetCurrentTaskHandle();

    adc_oneshot_sanity_check();

    adc_continuous_handle_cfg_t handle_cfg = {
        .max_store_buf_size = 8192,
        .conv_frame_size = 256,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &s_adc));

    adc_digi_pattern_config_t pattern[2] = {
        {
            .atten     = ADC_ATTEN_DB_12,
            .channel   = ADC_CHANNEL_4,
            .unit      = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
        {
            .atten     = ADC_ATTEN_DB_12,
            .channel   = ADC_CHANNEL_5,
            .unit      = ADC_UNIT_1,
            .bit_width = ADC_BITWIDTH_12,
        },
    };

    adc_continuous_config_t config = {
        .pattern_num  = 2,
        .adc_pattern  = pattern,
        .sample_freq_hz = 20000,
        .conv_mode    = ADC_CONV_SINGLE_UNIT_1,
        .format       = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };
    ESP_ERROR_CHECK(adc_continuous_config(s_adc, &config));

    adc_continuous_evt_cbs_t cbs = {
        .on_conv_done = s_conv_done_cb,
    };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(s_adc, &cbs, NULL));

    ESP_ERROR_CHECK(adc_continuous_start(s_adc));
    ESP_LOGI(TAG, "ADC continuous started: ch4/5, 20000 sps, conv_frame=256");
}

bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES])
{
    uint32_t v_count = 0;
    uint32_t i_count = 0;
    uint8_t buf[256];

    while (v_count < PQ_FRAME_SAMPLES || i_count < PQ_FRAME_SAMPLES) {
        uint32_t before = s_conv_done_count;
        uint32_t notified = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(2000));
        if (!notified) {
            ESP_LOGW(TAG, "no DMA notify (callback fires so far: %lu)", s_conv_done_count);
            return false;
        }

        uint32_t out_len = 0;
        esp_err_t err = adc_continuous_read(s_adc, buf, sizeof(buf), &out_len, 0);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "adc_continuous_read err: %s (cb count=%lu, before=%lu)",
                     esp_err_to_name(err), s_conv_done_count, before);
            return false;
        }

        for (uint32_t offset = 0; offset + SOC_ADC_DIGI_RESULT_BYTES <= out_len; offset += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t *)&buf[offset];
            uint32_t channel = sample->type2.channel;
            int16_t raw = (int16_t)(sample->type2.data & 0x0FFFu);

            if (channel == ADC_CHANNEL_4 && v_count < PQ_FRAME_SAMPLES) {
                v_raw[v_count++] = raw;
            } else if (channel == ADC_CHANNEL_5 && i_count < PQ_FRAME_SAMPLES) {
                i_raw[i_count++] = raw;
            }

            if (v_count >= PQ_FRAME_SAMPLES && i_count >= PQ_FRAME_SAMPLES) {
                return true;
            }
        }
    }

    return true;
}
