#include "inference_handler.h"
#include "model.h"  // This should match your model header file name
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <sys/time.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static const char* TAG = "InferenceHandler";

// Declare the op resolver before other TFLite variables
static tflite::MicroMutableOpResolver<9> micro_op_resolver;

// TensorFlow Lite variables
static const tflite::Model* model_ptr = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;

// Tensor arena - allocate from PSRAM
constexpr int kTensorArenaSize = 2 * 1024 * 1024;  // Changed from 1MB to 2MB
static uint8_t* tensor_arena = nullptr;

// Binary classification names
static const char* CLASS_NAMES[] = {
    "hippo", "other"
};

// Timing statistics structure
typedef struct {
    int64_t total_time_us;
    int64_t copy_time_us;
    int64_t inference_time_us;
    int count;
} inference_stats_t;

static inference_stats_t stats = {
    .total_time_us = 0,
    .copy_time_us = 0,
    .inference_time_us = 0,
    .count = 0
};

static inline int64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000L + (int64_t)tv.tv_usec;
}

int setup_inference() {
    // Clean up any previous allocations
    cleanup_inference();

    // Allocate new tensor arena
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, 
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT | MALLOC_CAP_32BIT);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM");
        return -1;
    }

    // Log memory information
    ESP_LOGI(TAG, "Tensor Arena allocated: %d bytes", kTensorArenaSize);
    ESP_LOGI(TAG, "Tensor Arena address: %p", tensor_arena);

    // Load the model
    model_ptr = tflite::GetModel(cifar10_model_quant_tflite);  // Use your model name
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch!");
        return -1;
    }

    // Register only the operations we need
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddMul();  // For rescaling
    micro_op_resolver.AddAdd();  // For batch normalization
    micro_op_resolver.AddMean(); // For global average pooling
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddLogistic();

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model_ptr, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;  // Assign to global pointer

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed with status: %d", allocate_status);
        return -1;
    }

    // Get input tensor
    input = interpreter->input(0);
    
    // Log tensor information
    ESP_LOGI(TAG, "Inference handler initialized successfully");
    ESP_LOGI(TAG, "Tensor Arena size: %d bytes", kTensorArenaSize);
    ESP_LOGI(TAG, "Input tensor dims: %dx%dx%d", 
             input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Input tensor type: %d", input->type);
    ESP_LOGI(TAG, "Input tensor bytes: %d", input->bytes);
    
    return 0;
}

void run_inference(const uint8_t* image_data) {
    if (!interpreter || !input) {
        ESP_LOGE(TAG, "Interpreter not initialized");
        return;
    }

    int64_t start_time = get_time_us();
    int64_t stage_time;

    // Copy data
    stage_time = get_time_us();
    memcpy(input->data.uint8, image_data, 224 * 224 * 3);
    stats.copy_time_us += get_time_us() - stage_time;

    // Run inference
    stage_time = get_time_us();
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed with status: %d", invoke_status);
        return;
    }
    stats.inference_time_us += get_time_us() - stage_time;

    // Process output - binary classification
    TfLiteTensor* output = interpreter->output(0);
    
    // Handle quantized output
    float confidence;
    if (output->type == kTfLiteUInt8) {
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        confidence = (output->data.uint8[0] - zero_point) * scale;
    } else {
        confidence = output->data.f[0];
    }

    // Binary classification threshold
    bool is_hippo = confidence >= 0.5f;
    float confidence_percent = is_hippo ? confidence * 100 : (1 - confidence) * 100;

    ESP_LOGI(TAG, "Prediction: %s (%.2f%% confidence)", 
             is_hippo ? "hippo" : "other", confidence_percent);

    // Update timing stats
    stats.total_time_us += get_time_us() - start_time;
    stats.count++;
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

    ESP_LOGI(TAG, "Inference Statistics:");
    ESP_LOGI(TAG, "  Average total time: %.2f ms", avg_total / 1000.0f);
    ESP_LOGI(TAG, "  Average copy time: %.2f ms", avg_copy / 1000.0f);
    ESP_LOGI(TAG, "  Average inference time: %.2f ms", avg_inference / 1000.0f);
    ESP_LOGI(TAG, "  Total inferences: %d", stats.count);
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
       // run_inference(test_hippo_image.h);
    }
    
    // Print average statistics
    print_inference_stats();
}