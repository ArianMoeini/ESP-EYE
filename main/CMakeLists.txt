idf_component_register(
    SRCS "main.c" "inference_handler.cpp" "model/model.c"
    INCLUDE_DIRS "." "model"
    REQUIRES "esp-tflite-micro" "driver" "esp_driver_uart" "esp_timer" "esp_psram"
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error -Os)


