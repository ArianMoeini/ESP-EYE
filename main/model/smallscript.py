import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/Users/arianmoeini/Desktop/hardware/esp-idf/examples/arian/esp-eye-image-classification/main/model/model.c")
operator_codes = interpreter._get_ops_list()

print("Operators in the model:")
for op in operator_codes:
    print(op)