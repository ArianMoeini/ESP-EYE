#ifndef INFERENCE_HANDLER_H
#define INFERENCE_HANDLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the inference system
int setup_inference(void);

// Run inference on a single image
void run_inference(const uint8_t* image_data);

// Cleanup resources
void cleanup_inference(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_HANDLER_H
