#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "inference_handler.h"
#include "test_hippo_image.h"
#include <inttypes.h>
#include "esp_cpu.h"
#include "esp_system.h"  // Add this for esp_get_cpu_freq_mhz()


static const char* TAG = "Main";

void app_main(void)
{
    ESP_LOGI(TAG, "Initializing Inference Handler...");
    
    if (setup_inference() != 0) {
        ESP_LOGE(TAG, "Failed to setup inference");
        return;
    }
    
    ESP_LOGI(TAG, "Inference Handler initialized successfully.");

    // Run continuously
    int count = 0;
    //while (1) {
        ESP_LOGI(TAG, "Running inference batch %d...", count++);
        
        uint32_t start_cycles = esp_cpu_get_cycle_count();
        run_inference(test_hippo_data);
        uint32_t end_cycles = esp_cpu_get_cycle_count();
        
        ESP_LOGI(TAG, "Cycles taken: %" PRIu32, end_cycles - start_cycles);
        // calculate to sedonds based on CPU frequency = 240 MHz
        double cycles_to_seconds = (end_cycles - start_cycles) / 240000000.0;
        ESP_LOGI(TAG, "Time taken: %.6f seconds", cycles_to_seconds);
        //vTaskDelay(pdMS_TO_TICKS(2000));
        ESP_LOGI(TAG, "----------------------------------------");
    //}
}
