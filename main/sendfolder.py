import serial
import time
import os
import cv2
import numpy as np
import serial.tools.list_ports
from datetime import datetime

# Add these quantization parameters at the top of the file
input_scale = 1.0  # From the model's quantization parameters
input_zero_point = 0  # From the model's quantization parameters
output_scale = 0.00390625  # From the model's quantization parameters
output_zero_point = 0  # From the model's quantization parameters

def log_message(message, log_file, print_to_console=True):
    """Helper function to log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_line)
    
    if print_to_console:
        print(message)

def preprocess_image(img, log_file):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    
    # Since input_scale is 1.0 and zero_point is 0, we only need to ensure uint8 type
    # and proper range [0, 255]
    img = img.astype(np.uint8)
    
    return img

def wait_for_memory_free(ser, log_file, timeout=5):
    """Wait until ESP32 reports memory is freed or timeout"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            return True
            
        line = ser.readline().decode().strip()
        if "After image processing" in line:
            time.sleep(1)
            return True

def send_image(ser, image_path, log_file):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to load image: {image_path}")
            
        img = preprocess_image(img, log_file)
        img_bytes = img.tobytes()
        total_size = len(img_bytes)
        
        chunk_size = 1024
        sent = 0
        while sent < total_size:
            current_chunk = min(chunk_size, total_size - sent)
            chunk = img_bytes[sent:sent + current_chunk]
            ser.write(chunk)
            sent += current_chunk
            time.sleep(0.001)
        
        start_time = time.time()
        inference_time = None
        probability = None
        classification = None
        
        while True:
            line = ser.readline().decode().strip()
            if line:
                if "Inference took" in line:
                    inference_time = line.split("took")[1].strip().split()[0]
                elif "Probability of PNEUMONIA:" in line:
                    probability = line.split("PNEUMONIA:")[1].strip().split()[0]
                elif "Classification:" in line:
                    classification = line.split("Classification:")[1].strip()
                    actual_class = 'NORMAL' if 'NORMAL' in image_path.upper() else 'PNEUMONIA'
                    is_correct = (classification == actual_class)
                    
                    results = (
                        f"Image: {image_path}\n"
                        f"Output: {probability}%\n"
                        f"Predicted Class: {classification}\n"
                        f"Actual Class: {actual_class}\n"
                        f"Correct Prediction: {is_correct}\n"
                        f"Inference Time: {inference_time} ms\n"
                        f"{'=' * 50}"
                    )
                    log_message(results, log_file, print_to_console=True)
                    
                    if wait_for_memory_free(ser, log_file):
                        return True, classification, is_correct
                        
            if time.time() - start_time > 15:
                return False, None, None
                
    except Exception as e:
        log_message(f"Error processing {image_path}: {e}", log_file)
        return False, None, None

def is_port_available(port_name):
    """Check if the specified port is available."""
    available_ports = [port.device for port in serial.tools.list_ports.comports()]
    return port_name in available_ports

def process_directory(port, directory_path, baud_rate=115200):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inference_results_{timestamp}.txt"
    
    if not is_port_available(port):
        print(f"ERROR: Port '{port}' is not available.")
        return

    try:
        image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"No valid images found in {directory_path}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        total_images = len(image_files)
        correct_predictions = 0
        
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            successful = 0
            for i, image_path in enumerate(image_files, 1):
                print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
                success, classification, is_correct = send_image(ser, image_path, log_file)
                
                if success:
                    successful += 1
                    if is_correct:
                        correct_predictions += 1
                
                time.sleep(1)
            
            # Calculate and log accuracy
            accuracy = (correct_predictions / total_images) * 100
            summary = (
                f"\n{'=' * 50}\n"
                f"FINAL RESULTS:\n"
                f"Total images processed: {total_images}\n"
                f"Correct predictions: {correct_predictions}\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"{'=' * 50}"
            )
            log_message(summary, log_file, print_to_console=True)
            
    except Exception as e:
        print(f"Error: {e}")

def list_available_ports():
    """List all available serial ports with details."""
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    print("\nAvailable Serial Ports:")
    print("-" * 60)
    for port in ports:
        print(f"Port: {port.device}")
        print(f"Description: {port.description}")
        print(f"Hardware ID: {port.hwid}")
        print("-" * 60)

if __name__ == "__main__":
    # List available ports
    list_available_ports()
    
    # Get port from user or use default
    PORT = input(f"Enter the port name (press Enter for default /dev/cu.SLAB_USBtoUART): ").strip()
    if not PORT:
        PORT = "/dev/cu.SLAB_USBtoUART"
    
    # Get directory path from user
    DIRECTORY_PATH = input("Enter the path to the image directory: ").strip()
    
    if not os.path.isdir(DIRECTORY_PATH):
        print(f"ERROR: Directory '{DIRECTORY_PATH}' does not exist.")
    else:
        process_directory(PORT, DIRECTORY_PATH)