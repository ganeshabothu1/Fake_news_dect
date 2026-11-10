
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)


# ==========================================
# 2. READ CSV FILES
# ==========================================

def read_csv_file(path):

    for encoding in ("utf-8", "cp1252", "latin-1"):

        try:
            return pd.read_csv(path, encoding=encoding)

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to read CSV: {path}")


# ==========================================
# 3. MAP LABELS
# 0 = FAKE
# 1 = TRUE
# ==========================================

def map_label(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"true", "real", "genuine", "1", "1.0"}:
        return 1

    if value in {"fake", "false", "0", "0.0"}:
        return 0

    return None


# ==========================================
# 4. BUILD NEWS TEXT
# ==========================================

def build_text(df):

    df = df.copy()

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    possible_columns = ["title", "text", "statement"]

    available_columns = [
        col for col in possible_columns
        if col in df.columns
    ]

    if not available_columns:

        raise ValueError(
            "No title, text, or statement column found. "
            f"Available columns: {df.columns.tolist()}"
        )

    text = (
        df[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    return text


# ==========================================
# 5. LOAD FAKE AND TRUE DATASETS
# ==========================================

def load_existing_dataset(filename, label):

    path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = read_csv_file(path)

    df["content"] = build_text(df)
    df["label"] = label

    return df[["content", "label"]]


print("\n========== DATASET LOADING ==========")

fake_df = load_existing_dataset("Fake.csv", 0)
true_df = load_existing_dataset("True.csv", 1)

print("Fake.csv records:", len(fake_df))
print("True.csv records:", len(true_df))


# ==========================================
# 6. LOAD IFND DATASET
# ==========================================

ifnd_path = os.path.join(DATASET_DIR, "IFND.csv")

if not os.path.exists(ifnd_path):

    raise FileNotFoundError(
        "IFND.csv not found inside dataset folder."
    )

print("\nLoading IFND dataset...")

ifnd_df = read_csv_file(ifnd_path)

ifnd_df.columns = [
    str(col).strip().lower()
    for col in ifnd_df.columns
]

if "label" not in ifnd_df.columns:

    raise ValueError(
        "IFND label column not found. "
        f"Available columns: {ifnd_df.columns.tolist()}"
    )

print("\nOriginal IFND labels:")
print(ifnd_df["label"].value_counts(dropna=False))

ifnd_df["content"] = build_text(ifnd_df)

ifnd_df["label"] = ifnd_df["label"].apply(map_label)

print("\nMapped IFND labels:")
print(ifnd_df["label"].value_counts(dropna=False))

ifnd_df = ifnd_df.dropna(subset=["label"])

ifnd_df = ifnd_df[
    ifnd_df["content"].str.len() > 0
]

ifnd_df["label"] = ifnd_df["label"].astype(int)

ifnd_df = ifnd_df[["content", "label"]]

print("Valid IFND records:", len(ifnd_df))


# ==========================================
# 7. COMBINE DATASETS
# ==========================================

data = pd.concat(
    [fake_df, true_df, ifnd_df],
    ignore_index=True
)

data["content"] = (
    data["content"]
    .fillna("")
    .astype(str)
    .str.strip()
)

data = data[data["content"].str.len() > 0]

data = data.drop_duplicates(
    subset=["content"]
).reset_index(drop=True)

if data["label"].nunique() != 2:

    raise ValueError(
        "Dataset must contain both Fake and True labels."
    )

print("\n========== COMBINED DATASET ==========")

print("Total records:", len(data))

print("\nLabel counts:")
print(data["label"].value_counts().sort_index())


# ==========================================
# 8. TRAIN AND TEST SPLIT
# ==========================================

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 9. TF-IDF TEXT PROCESSING
# ==========================================

print("\nTraining TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 10. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 11. MODEL TESTING
# ==========================================

print("\nTesting model...")

predictions = model.predict(X_test_tfidf)

probabilities = model.predict_proba(X_test_tfidf)

# Find probability column for class 1 (TRUE)
true_class_index = list(model.classes_).index(1)

true_probabilities = probabilities[:, true_class_index]


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

auc_score = roc_auc_score(
    y_test,
    true_probabilities
)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print(f"AUC Score : {auc_score:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["FAKE NEWS", "TRUE NEWS"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["FAKE", "TRUE"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")

plt.savefig(
    os.path.join(
        EVALUATION_DIR,
        "confusion_matrix.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 14. SAVE ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    true_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    os.path.join(
        EVALUATION_DIR,
        "roc_curve.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 15. SAVE EVALUATION RESULTS
# ==========================================

evaluation_results = {
    "model": "Logistic Regression",
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "auc_score": round(float(auc_score), 4),
    "confusion_matrix": cm.tolist(),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open(
    os.path.join(EVALUATION_DIR, "evaluation_results.json"),
    "w"
) as file:

    json.dump(
        evaluation_results,
        file,
        indent=4
    )


# ==========================================
# 16. SAVE TRAINED MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "fake_news_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

joblib.dump(model, model_path)

joblib.dump(vectorizer, vectorizer_path)


# ==========================================
# 17. FINAL OUTPUT
# ==========================================

print("\n========== TRAINING COMPLETED ==========")

print("Model saved:", model_path)
print("Vectorizer saved:", vectorizer_path)

print("\nEvaluation files saved in:", EVALUATION_DIR)

print("\nCreated files:")
print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. evaluation_results.json")

print("\nYou can now run your Flask application.")
