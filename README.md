# ESP-EYE Image Classification Project

## Overview

This project implements real-time image classification on the ESP-EYE development board using TensorFlow Lite for Microcontrollers. The system captures images using the onboard camera, processes them through a pre-trained neural network, and outputs the classification results.

## Hardware Components

- ESP-EYE development board
- Onboard OV2640 camera
- ESP32-S3 microcontroller
- 8MB PSRAM
- 4MB Flash memory

## Software Components

- ESP-IDF (Espressif IoT Development Framework)
- TensorFlow Lite for Microcontrollers
- Custom image classification model (CIFAR-10 based)

## Features

- Real-time image capture and processing
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

Once flashed, the ESP-EYE will automatically start capturing images and performing classifications. The results will be output via the serial monitor.

To interpret the results:
- Each classification will show the predicted class and confidence level
- The system updates classifications in real-time as the camera captures new images

## Model Training

The image classification model was trained on the CIFAR-10 dataset. For details on training your own model or using a different dataset, refer to the `trainingModel/` directory.

## Troubleshooting

- Ensure all connections are secure
- Check that the ESP-EYE's camera is unobstructed and properly aligned
- Verify that the serial monitor is set to the correct baud rate (typically 115200)

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

Not open source.


