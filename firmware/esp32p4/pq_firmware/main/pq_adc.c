#include "pq_adc.h"

#include <string.h>
#include "esp_adc/adc_oneshot.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "pq_adc";

#define SAMPLE_RATE_HZ   5000
#define SAMPLE_PERIOD_US (1000000 / SAMPLE_RATE_HZ)

static adc_oneshot_unit_handle_t s_adc1        = NULL;
static esp_timer_handle_t        s_timer        = NULL;
static TaskHandle_t              s_sample_task  = NULL;
static TaskHandle_t              s_frame_task   = NULL;

static int16_t           s_v_buf[PQ_FRAME_SAMPLES];
static int16_t           s_i_buf[PQ_FRAME_SAMPLES];
static volatile uint32_t s_count        = 0;
static volatile bool     s_collecting   = false;
static volatile uint32_t s_timer_ticks  = 0;

static void timer_cb(void *arg)
{
    s_timer_ticks++;
    xTaskNotifyGive(s_sample_task);
}

static void sample_task_fn(void *arg)
{
    ESP_LOGI(TAG, "sample_task started");
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (!s_collecting) continue;

        uint32_t idx = s_count;
        if (idx >= PQ_FRAME_SAMPLES) continue;

        int v = 0, i_val = 0;
        adc_oneshot_read(s_adc1, ADC_CHANNEL_4, &v);
        adc_oneshot_read(s_adc1, ADC_CHANNEL_5, &i_val);
        s_v_buf[idx] = (int16_t)v;
        s_i_buf[idx] = (int16_t)i_val;

        if (++s_count >= PQ_FRAME_SAMPLES) {
            s_collecting = false;
            xTaskNotifyGive(s_frame_task);
        }
    }
}

void pq_adc_init(void)
{
    adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = ADC_UNIT_1 };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &s_adc1));

    adc_oneshot_chan_cfg_t ch_cfg = {
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(s_adc1, ADC_CHANNEL_4, &ch_cfg));
    ESP_ERROR_CHECK(adc_oneshot_config_channel(s_adc1, ADC_CHANNEL_5, &ch_cfg));

    BaseType_t rc = xTaskCreatePinnedToCore(sample_task_fn, "adc_samp", 4096, NULL,
                                            configMAX_PRIORITIES - 1, &s_sample_task, 0);
    if (rc != pdPASS) {
        ESP_LOGE(TAG, "sample_task create failed (%d)", rc);
        return;
    }

    esp_timer_create_args_t timer_args = {
        .callback        = timer_cb,
        .arg             = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name            = "adc_tmr",
    };
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &s_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(s_timer, SAMPLE_PERIOD_US));

    ESP_LOGI(TAG, "ADC timer-oneshot: ch4/5 @ %d Hz (GPIO20/21) — 100ms/frame", SAMPLE_RATE_HZ);
}

bool pq_adc_read_frame(int16_t v_raw[PQ_FRAME_SAMPLES], int16_t i_raw[PQ_FRAME_SAMPLES])
{
    s_frame_task = xTaskGetCurrentTaskHandle();
    s_count      = 0;
    s_collecting = true;

    uint32_t notified = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(5000));
    if (!notified) {
        s_collecting = false;
        ESP_LOGW(TAG, "frame timeout — timer ticks so far: %lu", s_timer_ticks);
        return false;
    }

    ESP_LOGI(TAG, "frame ready (timer ticks: %lu, v[0]=%d i[0]=%d)",
             s_timer_ticks, s_v_buf[0], s_i_buf[0]);

    memcpy(v_raw, s_v_buf, PQ_FRAME_SAMPLES * sizeof(int16_t));
    memcpy(i_raw, s_i_buf, PQ_FRAME_SAMPLES * sizeof(int16_t));
    return true;
}
