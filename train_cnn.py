
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Reduce TensorFlow CPU thread usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Settings
DATA_DIR = "dataset/images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

# Load and split dataset
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20
)
train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED
)
print("Class mapping:", train_data.class_indices)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# CNN architecture
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/cnn_image_model.keras")

print("\nCNN MODEL TRAINING COMPLETED!")

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/cnn_accuracy.png")
plt.close()

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/cnn_loss.png")
plt.close()

# Evaluation
val_data.reset()
predictions = model.predict(val_data)
y_pred = (predictions.ravel() >= 0.5).astype(int)
y_true = val_data.classes

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"],
    zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"]
)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/cnn_confusion_matrix.png")
plt.close()

print("\nSaved CNN model and evaluation graphs.")
