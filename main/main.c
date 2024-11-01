#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "inference_handler.h"
#include "esp_cpu.h"
#include <string.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#define UART_NUM UART_NUM_0
#define BUF_SIZE 1024
#define IMAGE_SIZE (224 * 224 * 3)
#define START_MARKER "START:"
#define END_MARKER "END\n"
#define BUFFER_SIZE 1024

static const char* TAG = "Main";

static void receive_image(void) {
    uint8_t* image_buffer = (uint8_t*)heap_caps_malloc(IMAGE_SIZE, MALLOC_CAP_SPIRAM);
    if (!image_buffer) {
        ESP_LOGE(TAG, "Failed to allocate image buffer");
        return;
    }

    uint8_t temp_buffer[BUFFER_SIZE];
    int received = 0;
    bool started = false;
    int len;
    int expected_size = IMAGE_SIZE;

    // Wait for START message
    ESP_LOGI(TAG, "Waiting for START message...");
    while (!started) {
        len = uart_read_bytes(UART_NUM, temp_buffer, BUFFER_SIZE, pdMS_TO_TICKS(1000));
        if (len > 0) {
            temp_buffer[len] = '\0';
            if (strstr((char*)temp_buffer, "START:")) {
                started = true;
                ESP_LOGI(TAG, "Start marker received");
            }
        }
    }

    // Receive image data
    while (received < expected_size) {
        len = uart_read_bytes(UART_NUM, temp_buffer, 
            MIN(BUFFER_SIZE, expected_size - received), 
            pdMS_TO_TICKS(1000));
        
        if (len > 0) {
            // Check for END marker in this chunk
            bool end_found = false;
            int data_len = len;
            for (int i = 0; i < len - 3; i++) {
                if (temp_buffer[i] == 'E' && temp_buffer[i+1] == 'N' && 
                    temp_buffer[i+2] == 'D' && temp_buffer[i+3] == '\n') {
                    data_len = i;  // Only copy data before END marker
                    end_found = true;
                    break;
                }
            }

            // Copy valid data to image buffer
            if (received + data_len <= expected_size) {
                memcpy(image_buffer + received, temp_buffer, data_len);
                received += data_len;
                ESP_LOGI(TAG, "Received %d/%d bytes", received, expected_size);
            }

            if (end_found) {
                break;
            }
        } else {
            ESP_LOGE(TAG, "Timeout waiting for data");
            free(image_buffer);
            return;
        }
    }

    if (received == expected_size) {
        ESP_LOGI(TAG, "Image received successfully, running inference");
        run_inference(image_buffer);
        uart_write_bytes(UART_NUM, "DONE\n", 5);
    } else {
        ESP_LOGE(TAG, "Received incomplete image: %d/%d bytes", received, expected_size);
        uart_write_bytes(UART_NUM, "ERROR:INCOMPLETE\n", 16);
    }

    free(image_buffer);
}

void app_main(void) {
    ESP_LOGI(TAG, "Initializing...");
    
    // Initialize UART with larger buffers
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };
    
    uart_param_config(UART_NUM, &uart_config);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, BUF_SIZE * 2, 0, NULL, 0);
    uart_flush(UART_NUM);
    
    // Initialize inference handler
    if (setup_inference() != 0) {
        ESP_LOGE(TAG, "Failed to setup inference");
        cleanup_inference();
        return;
    }
    
    ESP_LOGI(TAG, "Initialization complete. Waiting for images...");
    
    // Continuously receive images
    while (1) {
        receive_image();
        vTaskDelay(pdMS_TO_TICKS(500));  // Delay between attempts
    }
}
