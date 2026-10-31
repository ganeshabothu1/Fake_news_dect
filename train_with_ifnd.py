
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# Read CSV files using supported encodings
def read_csv_file(path):
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to read CSV encoding: {path}")


# Convert labels to 0 = Fake, 1 = True
def map_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"true", "real", "genuine", "1", "1.0"}:
        return 1

    if value in {"fake", "false", "0", "0.0"}:
        return 0

    return None


# Extract news text from available columns
def build_text(df):
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    possible_columns = ["title", "text", "statement"]
    available_columns = [
        col for col in possible_columns if col in df.columns
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


# Load existing Fake.csv and True.csv
def load_existing_dataset(filename, label):
    path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = read_csv_file(path)
    df["content"] = build_text(df)
    df["label"] = label

    return df[["content", "label"]]


print("Loading existing datasets...")

fake_df = load_existing_dataset("Fake.csv", 0)
true_df = load_existing_dataset("True.csv", 1)

print("Fake.csv records:", len(fake_df))
print("True.csv records:", len(true_df))


# Load IFND dataset
ifnd_path = os.path.join(DATASET_DIR, "IFND.csv")

if not os.path.exists(ifnd_path):
    raise FileNotFoundError(
        "IFND.csv not found. Place it inside the dataset folder."
    )

print("Loading IFND dataset...")

ifnd_df = read_csv_file(ifnd_path)
ifnd_df.columns = [
    str(col).strip().lower() for col in ifnd_df.columns
]

if "label" not in ifnd_df.columns:
    raise ValueError(
        "IFND Label column not found. "
        f"Available columns: {ifnd_df.columns.tolist()}"
    )

ifnd_df["content"] = build_text(ifnd_df)
ifnd_df["label"] = ifnd_df["label"].apply(map_label)

# Remove unsupported labels and empty text
ifnd_df = ifnd_df.dropna(subset=["label"])
ifnd_df = ifnd_df[ifnd_df["content"].str.len() > 0]
ifnd_df["label"] = ifnd_df["label"].astype(int)
ifnd_df = ifnd_df[["content", "label"]]

print("Valid IFND records:", len(ifnd_df))


# Combine all datasets
data = pd.concat(
    [fake_df, true_df, ifnd_df],
    ignore_index=True
)

data["content"] = (
    data["content"].fillna("").astype(str).str.strip()
)

data = data[data["content"].str.len() > 0]
data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)

# Check labels
if data["label"].nunique() != 2:
    raise ValueError(
        "Combined dataset must contain both Fake and True labels."
    )

print("\nCombined dataset size:", len(data))
print("\nLabel counts (0 = Fake, 1 = True):")
print(data["label"].value_counts().sort_index())


# Prepare training and testing data
X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Logistic Regression
print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Evaluate the model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

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


# Save model and vectorizer
model_path = os.path.join(
    MODEL_DIR, "fake_news_model.pkl"
)

vectorizer_path = os.path.join(
    MODEL_DIR, "tfidf_vectorizer.pkl"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\n========== TRAINING COMPLETED ==========")
print("Model saved:", model_path)
print("Vectorizer saved:", vectorizer_path)
print("You can now test your Flask application.")
