#include "inference_handler.h"
#include "model.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"
#include <inttypes.h>
#include "esp_task_wdt.h"  // Add this at the top with other includes


// TFLite includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace tflite;

static const char* TAG = "InferenceHandler";

// TensorFlow Lite globals
static const tflite::Model* model = nullptr;
static MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Arena memory
constexpr int kTensorArenaSize = 512 * 1024;  // 512KB should be enough for the smaller model
static uint8_t *tensor_arena = nullptr;

int setup_inference(void) {
    // Allocate tensor arena from PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return -1;
    }
    ESP_LOGI(TAG, "Tensor arena allocated at %p", tensor_arena);

    // Load model
    model = tflite::GetModel(cifar10_model_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version does not match Schema");
        return -1;
    }

    // Create an op resolver with enough capacity
    static MicroMutableOpResolver<20> micro_op_resolver;  // Adjust size as needed
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddLogistic();
    // Add other operators here

    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model,
        micro_op_resolver,
        tensor_arena,
        kTensorArenaSize,
        nullptr
    );
    interpreter = &static_interpreter;

    // Allocate tensor buffers
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return -1;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Inference engine initialized successfully");
    return 0;
}

void run_inference(const uint8_t* image_buffer) {
    if (!interpreter || !image_buffer) {
        ESP_LOGE(TAG, "Invalid interpreter or buffer");
        return;
    }

    // Clear the entire tensor first
    memset(input->data.uint8, 0, IMAGE_SIZE);
    
    // Copy new data
    memcpy(input->data.uint8, image_buffer, IMAGE_SIZE);
    
    // Verify data distribution
    int zeros = 0, non_zeros = 0;
    uint8_t min_val = 255, max_val = 0;
    
    for (int i = 0; i < IMAGE_SIZE; i++) {
        if (input->data.uint8[i] == 0) zeros++;
        else non_zeros++;
        if (input->data.uint8[i] < min_val) min_val = input->data.uint8[i];
        if (input->data.uint8[i] > max_val) max_val = input->data.uint8[i];
    }
    
    ESP_LOGI(TAG, "Tensor statistics:");
    ESP_LOGI(TAG, "Min value: %d", min_val);
    ESP_LOGI(TAG, "Max value: %d", max_val);
    ESP_LOGI(TAG, "Zero pixels: %d", zeros);
    ESP_LOGI(TAG, "Non-zero pixels: %d", non_zeros);
    
    // Sample values in a grid pattern
    ESP_LOGI(TAG, "Sampling 3x3 grid across image:");
    for (int y = 0; y < 64; y += 32) {
        for (int x = 0; x < 64; x += 32) {
            int idx = (y * 64 + x) * 3;  // RGB data
            ESP_LOGI(TAG, "Position (%d,%d): R=%d G=%d B=%d", 
                     x, y,
                     input->data.uint8[idx],
                     input->data.uint8[idx + 1],
                     input->data.uint8[idx + 2]);
        }
    }
    
    // Reset watchdog before inference
    esp_task_wdt_reset();
    
    // Start timing
    int64_t start = esp_timer_get_time();
    
    // Run inference
    ESP_LOGI(TAG, "Starting inference...");
    TfLiteStatus invoke_status = interpreter->Invoke();
    
    // Reset watchdog after inference
    esp_task_wdt_reset();
    
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return;
    }

    // Calculate inference time
    int64_t end = esp_timer_get_time();
    int inference_time = (end - start) / 1000;
    ESP_LOGI(TAG, "Inference took %d ms", inference_time);

    // Get output
    TfLiteTensor* output = interpreter->output(0);
    
    // Log results
    if (output->type == kTfLiteFloat32) {
        float* results = output->data.f;
        ESP_LOGI(TAG, "Inference results (float):");
        ESP_LOGI(TAG, "Class 0: %.2f%%", results[0] * 100);
    } else if (output->type == kTfLiteUInt8) {
        uint8_t* results = output->data.uint8;
        float scale = 0.00390625f;  // From quantization parameters
        int zero_point = 0;         // From quantization parameters
        float probability = (results[0] - zero_point) * scale;
        
        ESP_LOGI(TAG, "Inference results:");
        ESP_LOGI(TAG, "Probability of PNEUMONIA: %.2f%%", probability * 100);
        ESP_LOGI(TAG, "Classification: %s", 
                (probability >= 0.5) ? "PNEUMONIA" : "NORMAL");
    }
}

void cleanup_inference(void) {
    if (tensor_arena) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
    interpreter = nullptr;
    input = nullptr;
    output = nullptr;
}

void initialize_inference() {
    // ... existing code ...
    
    // Log quantization parameters
    ESP_LOGI(TAG, "Model Quantization Parameters:");
    ESP_LOGI(TAG, "Input - Zero point: %" PRId32 ", Scale: %f", 
             input->params.zero_point, input->params.scale);
}