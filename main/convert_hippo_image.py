"""
This script converts a test hippo image to a C header file containing the image data.
It resizes the image to 224x224 and formats it for use with the ESP-EYE TensorFlow Lite model.
"""

import cv2
import numpy as np
import os

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input image path
IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "/Users/arianmoeini/Desktop/hardware/esp-idf/examples/arian/esp-eye-image-classification/trainingModel/dataset/test/hippo/seq000057-img000071-non-local.JPG"
)

# Output header file path
OUTPUT_HEADER = os.path.join(PROJECT_ROOT, "main/test_hippo_image.h")

def convert_image_to_header():
    """Convert a test hippo image to a C header file for ESP-EYE testing."""
    
    print(f"Converting image: {IMAGE_PATH}")
    print(f"Output header: {OUTPUT_HEADER}")
    
    # Read and resize image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {IMAGE_PATH}")
    
    print(f"Original image size: {img.shape}")
    img = cv2.resize(img, (224, 224))
    print(f"Resized to: {img.shape}")
    
    # Convert to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create header file content
    header_content = """// Auto-generated test hippo image header
#ifndef TEST_HIPPO_IMAGE_H
#define TEST_HIPPO_IMAGE_H

#include <stdint.h>

// Image dimensions
#define TEST_IMAGE_WIDTH 224
#define TEST_IMAGE_HEIGHT 224
#define TEST_IMAGE_CHANNELS 3

// Image data (RGB format, 224x224x3)
const uint8_t test_hippo_data[] = {
"""
    
    # Add image data
    for i, value in enumerate(img.flatten()):
        if i % 12 == 0:
            header_content += "\n    "
        header_content += f"{value}, "

    # Close the array and header guards
    header_content = header_content.rstrip(", ")
    header_content += "\n};\n\n#endif // TEST_HIPPO_IMAGE_H\n"

    # Write header file
    with open(OUTPUT_HEADER, "w") as f:
        f.write(header_content)
    
    print("Conversion completed successfully!")
    print(f"Header file size: {os.path.getsize(OUTPUT_HEADER)} bytes")

if __name__ == "__main__":
    try:
        convert_image_to_header()
    except Exception as e:
        print(f"Error: {e}")