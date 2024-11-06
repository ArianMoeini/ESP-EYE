import os
import re
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Run inference on a directory of images using a TFLite model reconstructed from a C array.')
parser.add_argument('source_file', type=str, help='Path to the C source file containing the model array.')
parser.add_argument('image_dir', type=str, help='Directory containing images for inference.')
parser.add_argument('output_file', type=str, help='File to save inference results.')
parser.add_argument('--max_images', type=int, default=100, help='Maximum number of images to process.')
args = parser.parse_args()

# Configuration
SOURCE_FILE = args.source_file
IMAGE_DIR = args.image_dir
OUTPUT_FILE = args.output_file
MAX_IMAGES = args.max_images

# Function to extract byte array from C source file
def extract_model_from_source(source_path):
    with open(source_path, 'r') as file:
        content = file.read()
    # Extract the array content between the curly braces
    array_match = re.search(r'{([^}]*)}', content, re.DOTALL)
    if not array_match:
        raise ValueError("Could not find the model array in the source file.")
    array_content = array_match.group(1)
    # Extract all hexadecimal byte values
    hex_values = re.findall(r'0x[0-9a-fA-F]+', array_content)
    # Convert hex values to bytes
    model_bytes = bytes(int(hv, 16) for hv in hex_values)
    return model_bytes

# Reconstruct the TFLite model
model_bytes = extract_model_from_source(SOURCE_FILE)

# Load the model into the TFLite interpreter
interpreter = tf.lite.Interpreter(model_content=model_bytes)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Input details
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]
input_dtype = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization']

# Output details
output_dtype = output_details[0]['dtype']
output_scale, output_zero_point = output_details[0]['quantization']

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_width, input_height))
    image_array = np.array(image, dtype=input_dtype)
    if input_dtype == np.uint8 and input_scale and input_zero_point:
        image_array = (image_array / 255.0 / input_scale + input_zero_point).astype(np.uint8)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to run inference
def run_inference(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # In milliseconds
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_dtype == np.uint8:
        output_data = output_scale * (output_data.astype(np.int32) - output_zero_point)
    return output_data, inference_time

# Collect image files from directory and subdirectories
image_files = []
for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Process images
results = []
correct_predictions = 0
total_images = min(MAX_IMAGES, len(image_files))

for i, image_path in enumerate(image_files[:total_images]):
    image_array = preprocess_image(image_path)
    output_data, inference_time = run_inference(image_array)
    probability = output_data[0][0] * 100  # Convert to percentage
    predicted_class = 'PNEUMONIA' if probability > 50 else 'NORMAL'
    actual_class = 'PNEUMONIA' if 'PNEUMONIA' in image_path.upper() else 'NORMAL'
    is_correct = predicted_class == actual_class
    if is_correct:
        correct_predictions += 1
    results.append({
        'image': image_path,
        'output_percentage': probability,
        'predicted_class': predicted_class,
        'actual_class': actual_class,
        'is_correct': is_correct,
        'inference_time_ms': inference_time
    })
    print(f"Processed {i+1}/{total_images}: {image_path}")

# Save results to file
with open(OUTPUT_FILE, 'w') as f:
    for result in results:
        f.write(f"Image: {result['image']}\n")
        f.write(f"Output: {result['output_percentage']:.2f}%\n")
        f.write(f"Predicted Class: {result['predicted_class']}\n")
        f.write(f"Actual Class: {result['actual_class']}\n")
        f.write(f"Correct Prediction: {result['is_correct']}\n")
        f.write(f"Inference Time: {result['inference_time_ms']:.2f} ms\n")
        f.write("\n")
    accuracy = (correct_predictions / total_images) * 100
    f.write(f"Total Images Processed: {total_images}\n")
    f.write(f"Correct Predictions: {correct_predictions}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")

print(f"Inference completed. Results saved to {OUTPUT_FILE}.")

# After loading the model and getting input/output details
print("\nQuantization Parameters:")
print(f"Input - Scale: {input_details[0]['quantization'][0]}, Zero point: {input_details[0]['quantization'][1]}")
print(f"Output - Scale: {output_details[0]['quantization'][0]}, Zero point: {output_details[0]['quantization'][1]}")

# Or add this right after interpreter setup:
details = interpreter.get_input_details()[0]
print(f"\nInput Tensor Details:")
print(f"Shape: {details['shape']}")
print(f"Dtype: {details['dtype']}")
print(f"Quantization: scale={details['quantization'][0]}, zero_point={details['quantization'][1]}")
