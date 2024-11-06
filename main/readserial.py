import serial
import time

def read_from_serial(port, baud_rate, output_file):
    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser, open(output_file, 'w') as f:
            print(f"Listening on {port} at {baud_rate} baud rate...")
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    print(line)  # Print to console for real-time monitoring
                    f.write(line + '\n')  # Write to file
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    PORT = "/dev/cu.SLAB_USBtoUART"  # Update with your serial port
    BAUD_RATE = 115200
    OUTPUT_FILE = "inference_results.txt"
    
    read_from_serial(PORT, BAUD_RATE, OUTPUT_FILE)