idf_component_register(
    SRCS "main.c" "inference_handler.cpp" "model/model.cpp"  # Include model.cpp
    INCLUDE_DIRS "." "model"
    REQUIRES "esp-tflite-micro"
)

# Optimize for size and silence specific warnings
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error -Os)
