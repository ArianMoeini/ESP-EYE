#ifndef INFERENCE_HANDLER_H_
#define INFERENCE_HANDLER_H_

#include <stdint.h>
#include "model/model.h"  // Ensure correct path

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the inference system
int setup_inference();

// Run one inference cycle
void run_inference(const uint8_t* image_data);

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_HANDLER_H_
