def convert_tflite_to_header(tflite_model_path, header_path):
    # Read the TFLite model file
    with open(tflite_model_path, 'rb') as f:
        model_data = f.read()
    
    # Create the header file
    with open(header_path, 'w') as f:
        # Write header guards and array declaration
        f.write("// Auto-generated file from model.tflite\n")
        f.write("#ifndef MODEL_TFLITE_H\n")
        f.write("#define MODEL_TFLITE_H\n\n")
        f.write("// Model data array\n")
        f.write("alignas(8) const unsigned char model_tflite[] = {\n    ")
        
        # Write the model data as hex values
        for i, byte in enumerate(model_data):
            f.write(f"0x{byte:02x}")
            if i < len(model_data) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0:  # Line break every 12 bytes
                f.write("\n    ")
        
        # Write array size and close header guard
        f.write("\n};\n\n")
        f.write(f"const unsigned int model_tflite_len = {len(model_data)};\n\n")
        f.write("#endif  // MODEL_TFLITE_H\n")

    print(f"Converted {tflite_model_path} to {header_path}")
    print(f"Model size: {len(model_data):,} bytes")

# Convert the model
convert_tflite_to_header('optimized_model.tflite', 'model.h')