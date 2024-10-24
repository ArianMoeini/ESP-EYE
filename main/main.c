#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "inference_handler.h"
#include "model/images.h"

static const char* TAG = "Main";

void app_main(void)
{
    ESP_LOGI(TAG, "Initializing Inference Handler...");

    // Initialize the inference handler
    if (setup_inference() != 0) {
        ESP_LOGE(TAG, "Failed to setup inference handler");
        return;
    }

    ESP_LOGI(TAG, "Inference Handler initialized successfully.");

    // Iterate over predefined images and run inference
    for (size_t i = 0; i < num_images; i++) {
        ESP_LOGI(TAG, "Running inference on Image %zu...", i);
        run_inference(images[i]);
        vTaskDelay(pdMS_TO_TICKS(2000));  // Delay between inferences (2 seconds)
    }

    ESP_LOGI(TAG, "All inferences completed.");
}
