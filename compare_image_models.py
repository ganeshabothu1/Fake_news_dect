
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_DIR = "dataset/images"
BATCH_SIZE = 8
SEED = 42

# Load saved models
cnn_model = tf.keras.models.load_model(
    "models/cnn_image_model.keras"
)

mobilenet_model = tf.keras.models.load_model(
    "models/mobilenetv2_image_model.keras"
)

# CNN validation data
cnn_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20
)

cnn_data = cnn_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# MobileNetV2 validation data
mobile_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

mobile_data = mobile_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(160, 160),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# Verify same validation labels and order
actual = cnn_data.classes

if not np.array_equal(actual, mobile_data.classes):
    raise ValueError("Validation image order does not match!")

print("Class mapping:", cnn_data.class_indices)
print("Validation images:", len(actual))

# CNN predictions
cnn_data.reset()
cnn_prob = cnn_model.predict(
    cnn_data, verbose=1
).ravel()

cnn_pred = (cnn_prob >= 0.5).astype(int)

# MobileNetV2 predictions
mobile_data.reset()

mobile_prob = mobilenet_model.predict(
    mobile_data, verbose=1
).ravel()


mobile_pred = (mobile_prob >= 0.5).astype(int)

# Calculate metrics
def get_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC AUC": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    }

cnn_results = get_metrics(actual, cnn_pred, cnn_prob)
mobile_results = get_metrics(actual, mobile_pred, mobile_prob)

# Comparison table
comparison = pd.DataFrame(
    [cnn_results, mobile_results],
    index=["CNN", "MobileNetV2"]
)

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4))

os.makedirs("results", exist_ok=True)

comparison.to_csv(
    "results/image_model_comparison.csv"
)

# Comparison graph
comparison.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1.05)
)

plt.title("CNN vs MobileNetV2 Performance")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "results/image_model_comparison.png"
)

plt.close()

print("\nComparison completed successfully!")
print("Saved: results/image_model_comparison.csv")
print("Saved: results/image_model_comparison.png")
