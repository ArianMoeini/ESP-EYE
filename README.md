# ESP-EYE Image Classification Project

## Overview

This project demonstrates image classification on the ESP-EYE development board using TensorFlow Lite for Microcontrollers. Unlike real-time camera input, this implementation uses a static image converted to a C binary as input, which is then processed through a pre-trained neural network to output classification results.

## Hardware Components

- ESP-EYE development board
- ESP32-S3 microcontroller
- 8MB PSRAM
- 4MB Flash memory

## Software Components

- ESP-IDF (Espressif IoT Development Framework)
- TensorFlow Lite for Microcontrollers
- Custom image classification model (CIFAR-10 based)
- Static image data converted to C binary

## Features

- Static image processing using pre-converted image data
- Low-latency inference using TensorFlow Lite
- Classification of images into 10 categories (CIFAR-10 classes)
- Serial output of classification results

## Setup Instructions

### Prerequisites

- ESP-IDF v4.4 or later
- Python 3.7-3.10
- Git

### Environment Setup

1. Clone the repository:
   ```
   git clone https://github.com/ArianMoeini/ESP-EYE.git
   cd ESP-EYE
   ```

2. Set up ESP-IDF:
   Follow the [official ESP-IDF setup guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Building and Flashing

1. Configure the project:
   ```
   idf.py set-target esp32s3
   idf.py menuconfig
   ```

2. Build the project:
   ```
   idf.py build
   ```

3. Flash to the ESP-EYE:
   ```
   idf.py -p [PORT] flash
   ```
   Replace [PORT] with your device's serial port (e.g., /dev/ttyUSB0 on Linux or COM3 on Windows)

4. Monitor the output:
   ```
   idf.py -p [PORT] monitor
   ```

## Usage

Once flashed, the ESP-EYE will process the static image data and perform classification. The results will be output via the serial monitor.

To interpret the results:
- The classification will show the predicted class and confidence level for the static image
- Unlike a real-time system, this will be a one-time classification of the pre-loaded image

## Static Image Data

The project uses a static image converted to a C binary. This image data is included in the project files and is processed by the TensorFlow Lite model.

To use a different image:
1. Convert your desired image to the appropriate format and size
2. Update the C binary data in the project
3. Rebuild and flash the project

## Model Information

The image classification model is based on the CIFAR-10 dataset, capable of classifying images into 10 categories. The model is quantized and optimized for running on the ESP32-S3 microcontroller.

## Troubleshooting

- Ensure all connections are secure
- Verify that the serial monitor is set to the correct baud rate (typically 115200)
- Check that the static image data is correctly included in the build
