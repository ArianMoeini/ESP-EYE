#ifndef IMAGES_H_
#define IMAGES_H_

#include <stdint.h>

// Include the generated image header
#include "rgb_output.h"

// Add more images as needed
// #include "image_1.h"
// #include "image_2.h"

const uint8_t* images[] = {image_0 /*, image_1, image_2 */};
const size_t num_images = sizeof(images) / sizeof(images[0]);

#endif  // IMAGES_H_
