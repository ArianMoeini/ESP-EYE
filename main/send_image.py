import serial
import time
import os
import cv2
import numpy as np

def preprocess_image(img):
    """Preprocess image to match model input requirements"""
    # Resize to 224x224 while maintaining aspect ratio
    target_size = (224, 224)
    
    # Calculate aspect ratio
    h, w = img.shape[:2]
    aspect = w / h
    
    # Determine padding
    if aspect > 1:  # width > height
        new_w = target_size[0]
        new_h = int(new_w / aspect)
        pad_h = (target_size[1] - new_h) // 2
        pad_w = 0
    else:  # height > width
        new_h = target_size[1]
        new_w = int(new_h * aspect)
        pad_w = (target_size[0] - new_w) // 2
        pad_h = 0
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create black canvas
    processed = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Place resized image in center
    if aspect > 1:
        processed[pad_h:pad_h+new_h, :] = resized
    else:
        processed[:, pad_w:pad_w+new_w] = resized
    
    return processed

def send_image(port, image_path, baud_rate=115200):
    try:
        print(f"Opening serial port {port} at {baud_rate} baud...")
        ser = serial.Serial(
            port,
            baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        # Clear any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.5)
        
        # Process image
        print("Processing image...")
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Could not read image {image_path}")
        
        processed_img = preprocess_image(img)
        img_bytes = processed_img.tobytes()
        
        # Verify image size
        expected_size = 224 * 224 * 3
        if len(img_bytes) != expected_size:
            raise Exception(f"Invalid image size: got {len(img_bytes)}, expected {expected_size}")
        
        # Send START message
        print("Sending START marker")
        ser.write(f"START:{len(img_bytes)}\n".encode())
        time.sleep(0.1)
        
        # Send image data with smaller chunks and longer delays
        chunk_size = 512  # Reduced chunk size
        total_sent = 0
        
        for i in range(0, len(img_bytes), chunk_size):
            chunk = img_bytes[i:i + chunk_size]
            bytes_written = ser.write(chunk)
            total_sent += bytes_written
            print(f"Sent {total_sent}/{len(img_bytes)} bytes")
            time.sleep(0.1)  # Increased delay between chunks
            
            # Verify all bytes were sent
            if bytes_written != len(chunk):
                raise Exception(f"Failed to send complete chunk: {bytes_written}/{len(chunk)}")
        
        # Verify total bytes sent
        if total_sent != len(img_bytes):
            raise Exception(f"Failed to send complete image: {total_sent}/{len(img_bytes)}")
        
        # Send END marker with delay
        time.sleep(0.2)
        print("Sending END marker")
        ser.write(b"END\n")
        
        # Wait for result
        print("Waiting for processing result...")
        timeout = 10  # 10 seconds timeout
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if ser.in_waiting:
                response = ser.readline().decode().strip()
                print(f"ESP32 response: {response}")
                if response == "DONE":
                    print("Image processed successfully")
                    break
                elif response.startswith("ERROR"):
                    print(f"ESP32 reported error: {response}")
                    break
            time.sleep(0.1)
        
        ser.close()
        print("Done")
        
    except Exception as e:
        print(f"Error: {e}")
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    PORT = "/dev/cu.SLAB_USBtoUART"
    IMAGE_PATH = input("Enter the path to the image file: ").strip()
    
    if not os.path.isfile(IMAGE_PATH):
        print(f"ERROR: File '{IMAGE_PATH}' does not exist.")
    else:
        send_image(PORT, IMAGE_PATH)
