#include "inference_handler.h"
#include "model/model.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static const char* TAG = "InferenceHandler";

// TensorFlow Lite variables
static const tflite::Model* model_ptr = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 150 * 1024;  // Adjust as needed
static uint8_t tensor_arena[kTensorArenaSize];

// CIFAR-10 class names
static const char* CIFAR10_CLASSES[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

int setup_inference() {
    // Load the model
    model_ptr = tflite::GetModel(cifar10_model_quant_tflite);
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch!");
        return -1;
    }

    // Define the resolver with required ops
    // Increase the number to accommodate additional ops
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddQuantize();  // Added QUANTIZE operation

    // Alternatively, if you have more operations, increase the template parameter accordingly
    // e.g., MicroMutableOpResolver<10>

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
    return 0;
}

void run_inference(const uint8_t* image_data) {
    if (!interpreter || !input) {
        ESP_LOGE(TAG, "Interpreter not initialized");
        return;
    }

    // Assuming image_data is 32x32x3 uint8_t
    memcpy(input->data.uint8, image_data, 32 * 32 * 3);

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return;
    }

    // Retrieve output tensor
    TfLiteTensor* output = interpreter->output(0);

    // Find the class with the highest probability
    int max_index = 0;
    uint8_t max_value = output->data.uint8[0];
    for (int i = 1; i < 10; i++) {
        if (output->data.uint8[i] > max_value) {
            max_value = output->data.uint8[i];
            max_index = i;
        }
    }

    // Convert quantized output to float probability
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    float confidence = (max_value - zero_point) * scale;

    ESP_LOGI(TAG, "Predicted: %s (%.2f%% confidence)", 
             CIFAR10_CLASSES[max_index], confidence * 100);
}
