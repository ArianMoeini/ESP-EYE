#include "inference_handler.h"
#include "model/model.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "esp_heap_caps.h"
#include "test_hippo_image.h"
#include "esp_system.h"
#include <sys/time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/portmacro.h"


static const char* TAG = "InferenceHandler";

// TensorFlow Lite variables
static const tflite::Model* model_ptr = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 800 * 1024;  // 800KB
static uint8_t* tensor_arena = nullptr;  // Will be allocated from PSRAM

// Update class names for binary classification
static const char* CLASS_NAMES[] = {
    "hippo", "other"
};

// Add timing statistics structure
typedef struct {
    int64_t total_time_us;
    int64_t copy_time_us;
    int64_t inference_time_us;
    int64_t post_process_time_us;
    int count;
} inference_stats_t;

static inference_stats_t stats = {0};

static inline int64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000L + (int64_t)tv.tv_usec;
}

int setup_inference() {
    // Allocate tensor arena from PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        return -1;
    }

    // Load the model
    model_ptr = tflite::GetModel(cifar10_model_quant_tflite);
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch!");
        return -1;
    }

    // Define the resolver with required ops
    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddPad();

    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model_ptr, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return -1;
    }

    // Obtain input tensor
    input = interpreter->input(0);
    ESP_LOGI(TAG, "Inference handler initialized successfully");
    ESP_LOGI(TAG, "Tensor Arena size: %d bytes", kTensorArenaSize);
    ESP_LOGI(TAG, "Input tensor size: %d bytes", input->bytes);
    
    return 0;
}

void run_inference(const uint8_t* image_data) {
    if (!interpreter || !input) {
        ESP_LOGE(TAG, "Interpreter not initialized");
        return;
    }

    // int64_t start_time = get_time_us();
    // int64_t stage_time;

    // // Measure copy time
    // stage_time = get_time_us();
    // memcpy(input->data.uint8, image_data, 224 * 224 * 3);
    // stats.copy_time_us += get_time_us() - stage_time;

    // // Measure inference time
    // stage_time = get_time_us();
    // if (interpreter->Invoke() != kTfLiteOk) {
    //     ESP_LOGE(TAG, "Inference failed");
    //     return;
    // }
    // stats.inference_time_us += get_time_us() - stage_time;

    // // Measure post-processing time
    // stage_time = get_time_us();
    
    TfLiteTensor* output = interpreter->output(0);
    float confidence;
    int predicted_class;
    
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    
    if (output->type == kTfLiteUInt8) {
        confidence = (output->data.uint8[0] - zero_point) * scale;
        predicted_class = confidence >= 0.5f ? 0 : 1;
    } else {
        confidence = output->data.f[0];
        predicted_class = confidence >= 0.5f ? 0 : 1;
    }

    float confidence_percent = predicted_class == 0 ? confidence * 100 : (1 - confidence) * 100;
    
    // stats.post_process_time_us += get_time_us() - stage_time;

    // // Calculate total time
    // stats.total_time_us += get_time_us() - start_time;
    // stats.count++;

    // Log the results with timing information
    ESP_LOGI(TAG, "Prediction: %s (%.2f%% confidence)", 
             CLASS_NAMES[predicted_class], confidence_percent);
    // ESP_LOGI(TAG, "Timing for this inference:");
    // ESP_LOGI(TAG, "  Copy time: %lld us", (get_time_us() - start_time) - stats.inference_time_us);
    // ESP_LOGI(TAG, "  Inference time: %lld us", stats.inference_time_us);
    // ESP_LOGI(TAG, "  Total time: %lld us", get_time_us() - start_time);
}

void cleanup_inference() {
    if (tensor_arena != nullptr) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
}

// Add this function to print average statistics
void print_inference_stats() {
    if (stats.count == 0) {
        ESP_LOGI(TAG, "No inference statistics available");
        return;
    }

    float avg_total = stats.total_time_us / (float)stats.count;
    float avg_copy = stats.copy_time_us / (float)stats.count;
    float avg_inference = stats.inference_time_us / (float)stats.count;
    float avg_post = stats.post_process_time_us / (float)stats.count;
    float fps = 1000000.0f / avg_total;  // Convert microseconds to FPS

    ESP_LOGI(TAG, "Inference Statistics over %d runs:", stats.count);
    ESP_LOGI(TAG, "  Average total time: %.2f ms", avg_total / 1000.0f);
    ESP_LOGI(TAG, "  Average copy time: %.2f ms", avg_copy / 1000.0f);
    ESP_LOGI(TAG, "  Average inference time: %.2f ms", avg_inference / 1000.0f);
    ESP_LOGI(TAG, "  Average post-processing time: %.2f ms", avg_post / 1000.0f);
    ESP_LOGI(TAG, "  Theoretical FPS: %.2f", fps);
}

// Update run_test_inference to run multiple times
void run_test_inference() {
    const int NUM_RUNS = 10;  // Number of times to run the inference
    
    ESP_LOGI(TAG, "Running inference test %d times...", NUM_RUNS);
    
    // Reset statistics
    memset(&stats, 0, sizeof(stats));
    
    // Run inference multiple times
    for (int i = 0; i < NUM_RUNS; i++) {
        ESP_LOGI(TAG, "Test run %d/%d", i + 1, NUM_RUNS);
        run_inference(test_hippo_data);
        vTaskDelay(pdMS_TO_TICKS(100));  // Simple 100ms delay
    }
    
    // Print average statistics
    print_inference_stats();
}
