#ifndef INFERENCE_HANDLER_H
#define INFERENCE_HANDLER_H

#include <stdint.h>
#include <stddef.h>
#include "esp_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define IMAGE_SIZE (224 * 224 * 3)

// Function declarations
int setup_inference(void);
void run_inference(const uint8_t* image_data);
void cleanup_inference(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_HANDLER_H
