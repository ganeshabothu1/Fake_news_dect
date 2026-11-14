
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# CPU optimization for 8 GB RAM laptop
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Settings
DATA_DIR = "dataset/images"
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Image preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.20
)

# Training data
train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED
)

# Validation data
val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Add classification layers
inputs = layers.Input(shape=(160, 160, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

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
model.save("models/mobilenetv2_image_model.keras")

print("\nMobileNetV2 MODEL TRAINING COMPLETED!")

# Accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/mobilenetv2_accuracy.png")
plt.close()

# Loss graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/mobilenetv2_loss.png")
plt.close()

# Evaluation
val_data.reset()
probabilities = model.predict(val_data).ravel()
predictions = (probabilities >= 0.5).astype(int)
actual = val_data.classes

print("\nClassification Report:")
print(classification_report(
    actual,
    predictions,
    target_names=["FAKE", "REAL"],
    zero_division=0
))

auc = roc_auc_score(actual, probabilities)
print("ROC AUC:", auc)

# Confusion Matrix
cm = confusion_matrix(actual, predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["FAKE", "REAL"])
plt.yticks([0, 1], ["FAKE", "REAL"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("results/mobilenetv2_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(actual, probabilities)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"MobileNetV2 AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("MobileNetV2 ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("results/mobilenetv2_roc_curve.png")
plt.close()

print("\nSaved MobileNetV2 model and evaluation graphs.")
