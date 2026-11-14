
import tensorflow as tf

print("TensorFlow:", tf.__version__)

# CNN Test
cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("CNN Model Created Successfully!")

# MobileNetV2 Test
mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights=None
)

print("MobileNetV2 Model Created Successfully!")
