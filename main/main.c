#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "inference_handler.h"
#include "esp_cpu.h"
#include <string.h>
#include "esp_task_wdt.h"
#include "esp_heap_caps.h"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#define UART_NUM UART_NUM_0
#define BUF_SIZE 1024
#define IMAGE_SIZE (64 * 64 * 3) // 12,288 bytes
#define START_MARKER "START:"
#define END_MARKER "END\n"
#define BUFFER_SIZE 1024

static const char* TAG = "Main";

#define UART_BUF_SIZE (1024)
#define UART_PORT_NUM UART_NUM_0
#define UART_BAUD_RATE 115200

static void configure_uart(void) {
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT_NUM, UART_BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_PORT_NUM, &uart_config));
    ESP_LOGI(TAG, "UART configured with baud rate %d", UART_BAUD_RATE);
}

static void receive_task(void *pvParameters) {
    ESP_LOGI(TAG, "Receive task started");

    // Register this task with the watchdog
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));

    // Allocate buffer with error checking
    uint8_t* image_buffer = (uint8_t*)heap_caps_malloc(IMAGE_SIZE, MALLOC_CAP_SPIRAM);
    if (!image_buffer) {
        ESP_LOGE(TAG, "Failed to allocate image buffer");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Image buffer allocated in PSRAM");

    // Initialize variables
    size_t total_received = 0;
    uint8_t* temp_buffer = (uint8_t*)heap_caps_malloc(UART_BUF_SIZE, MALLOC_CAP_INTERNAL);
    if (!temp_buffer) {
        ESP_LOGE(TAG, "Failed to allocate temp buffer");
        heap_caps_free(image_buffer);
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Temp buffer allocated in internal memory");

    while (1) {
        esp_task_wdt_reset();  // Reset watchdog in the main loop
        
        if (total_received >= IMAGE_SIZE) {
            ESP_LOGI(TAG, "Image received, running inference");
            run_inference(image_buffer);
            total_received = 0;
        }
        
        if (total_received < IMAGE_SIZE) {
            // Add timeout to prevent infinite blocking
            int len = uart_read_bytes(UART_PORT_NUM, 
                                    temp_buffer, 
                                    MIN(UART_BUF_SIZE, IMAGE_SIZE - total_received), 
                                    pdMS_TO_TICKS(100));
            
            if (len > 0) {
                if (total_received + len <= IMAGE_SIZE) {
                    memcpy(image_buffer + total_received, temp_buffer, len);
                    total_received += len;
                    ESP_LOGI(TAG, "Received %d/%d bytes", total_received, IMAGE_SIZE);
                } else {
                    ESP_LOGW(TAG, "Received bytes exceed IMAGE_SIZE");
                    total_received = IMAGE_SIZE; // Force inference
                }
            } else {
                ESP_LOGD(TAG, "No bytes received in this iteration");
            }

            // Yield to prevent watchdog triggers
            vTaskDelay(pdMS_TO_TICKS(10));
        }

        // Periodically log free heap
        static TickType_t last_log_time = 0;
        if (xTaskGetTickCount() - last_log_time > pdMS_TO_TICKS(5000)) {
            size_t free_heap_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
            size_t free_heap_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
            ESP_LOGI(TAG, "Free PSRAM: %d bytes, Free Internal: %d bytes", free_heap_psram, free_heap_internal);
            last_log_time = xTaskGetTickCount();
        }
    }

    // Cleanup (unreachable in this context)
    heap_caps_free(temp_buffer);
    heap_caps_free(image_buffer);
}

void app_main(void) {
    ESP_LOGI(TAG, "Initializing...");

    // Define and initialize the watchdog configuration
    esp_task_wdt_config_t wdt_config = {
        .timeout_ms = 20000,  // 20 seconds timeout
        .idle_core_mask = 0,  // Apply to all cores
        .trigger_panic = true // Trigger panic on timeout
    };

    // Initialize Task Watchdog with the configuration
    esp_err_t wdt_init_result = esp_task_wdt_init(&wdt_config);
    if (wdt_init_result != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize task watchdog: %s", esp_err_to_name(wdt_init_result));
    } else {
        esp_task_wdt_add(NULL); // Add current task (app_main) to WDT
    }

    // Configure UART
    configure_uart();

    // Initialize inference handler
    if (setup_inference() != 0) {
        ESP_LOGE(TAG, "Failed to setup inference");
        cleanup_inference();
        return;
    }
    ESP_LOGI(TAG, "Inference handler initialized");

    // Create and start the receive task
    BaseType_t result = xTaskCreatePinnedToCore(
        receive_task,          // Task function
        "receive_task",        // Task name
        8192,                  // Stack size (8 KB)
        NULL,                  // Task parameters
        5,                     // Priority
        NULL,                  // Task handle
        0                      // Run on core 0
    );

    if (result == pdPASS) {
        ESP_LOGI(TAG, "Receive task created successfully");
    } else {
        ESP_LOGE(TAG, "Failed to create receive task");
        cleanup_inference();
    }

    // Let app_main finish to avoid blocking
    ESP_LOGI(TAG, "app_main completed");
}
