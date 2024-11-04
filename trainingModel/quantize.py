import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os

# Set parameters
img_height = 64
img_width = 64
batch_size = 8
epochs = 10  # Adjust as needed

# Define directories
train_dir = '/Users/arianmoeini/Desktop/hardware/esp-idf/examples/arian/esp-eye-image-classification/trainingModel/dataset/train'
test_dir = '/Users/arianmoeini/Desktop/hardware/esp-idf/examples/arian/esp-eye-image-classification/trainingModel/dataset/test'

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=123
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Data augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Prepare datasets
def prepare_dataset(ds, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = prepare_dataset(train_ds, shuffle=True)
test_ds = prepare_dataset(test_ds)

# Define the model
def create_model():
    # Create the rescaling layer separately
    rescaling = tf.keras.layers.Rescaling(1./255)
    
    # Create the main model without rescaling
    main_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create the final model
    inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
    x = rescaling(inputs)
    outputs = main_model(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# Pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=np.ceil(len(train_ds) / batch_size).astype(np.int32) * epochs
    )
}

# Apply pruning (replace this section)
model = create_model()
# Only prune the main layers, not the preprocessing
prunable_layers = ['conv2d', 'dense']
model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=lambda layer: tfmot.sparsity.keras.prune_low_magnitude(
        layer, **pruning_params
    ) if any(layer_type in layer.name for layer_type in prunable_layers)
    else layer
)

# Use model_for_pruning instead of model
model = model_for_pruning

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
]

# Train the model
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=callbacks
)

# Strip pruning wrappers
model = tfmot.sparsity.keras.strip_pruning(model)

# Save the pruned model
model.save('pruned_model.h5')

# Representative dataset for quantization
def representative_dataset():
    for images, _ in train_ds.take(100):
        yield [images]

# Convert to TensorFlow Lite model with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save the quantized model
tflite_model_file = 'optimized_model.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

# Convert the TFLite model to a C array
def convert_to_c_array(model_path, output_path):
    with open(model_path, 'rb') as f:
        model_data = f.read()
    hex_data = ', '.join(f'0x{b:02x}' for b in model_data)
    c_array = f'const unsigned char model[] = {{{hex_data}}};\nunsigned int model_len = {len(model_data)};'
    with open(output_path, 'w') as f:
        f.write(c_array)

convert_to_c_array(tflite_model_file, 'model.h')

# Display model size
model_size = os.path.getsize(tflite_model_file)
print(f"Optimized TFLite model size: {model_size / 1024:.2f} KB")
