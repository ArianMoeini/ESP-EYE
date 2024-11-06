import serial
import time
import os
import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    
    # Print detailed image statistics
    print("\nImage statistics:")
    for channel, name in enumerate(['R', 'G', 'B']):
        channel_data = img[:,:,channel]
        print(f"{name} channel - min: {channel_data.min()}, "
              f"max: {channel_data.max()}, "
              f"mean: {channel_data.mean():.2f}, "
              f"std: {channel_data.std():.2f}")
    
    # Sample 4x4 grid
    print("\nSampling 4x4 grid:")
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            pixel = img[y,x]
            print(f"({x},{y}): R={pixel[0]} G={pixel[1]} B={pixel[2]}")
    
    return img

def send_image(port, baud_rate=115200, image_path=None):
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Read and process image
        if image_path is None:
            raise Exception("No image path provided")
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to load image: {image_path}")
            
        img = cv2.resize(img, (64, 64))
        
        # Print image statistics
        print(f"\nImage stats before sending:")
        print(f"Shape: {img.shape}")
        print(f"Min value: {img.min()}")
        print(f"Max value: {img.max()}")
        print(f"Mean value: {img.mean():.2f}")
        print(f"First 10 pixels: {img.flatten()[:10]}")
        
        # Convert to bytes and send
        img_bytes = img.tobytes()
        print(f"First 10 bytes being sent: {list(img_bytes[:10])}")
        
        # Send in smaller chunks
        chunk_size = 1024
        total_size = len(img_bytes)
        sent = 0
        
        while sent < total_size:
            current_chunk = min(chunk_size, total_size - sent)
            chunk = img_bytes[sent:sent + current_chunk]
            ser.write(chunk)
            ser.flush()  # Ensure data is sent
            
            # Add a small delay between chunks
            time.sleep(0.001)
            
            sent += current_chunk
            print(f"Sent {sent}/{total_size} bytes")
            
        print("Transfer complete")
        ser.close()
        
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
        send_image(PORT, image_path=IMAGE_PATH)
