import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# FAKE NEWS DETECTION SYSTEM
# ============================================================

print("=" * 60)
print("        FAKE NEWS DETECTION SYSTEM")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "Fake.csv"
)

TRUE_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "True.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "fake_news_model.pkl"
)

VECTORIZER_FILE = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# Create models folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")


if not os.path.exists(FAKE_FILE):
    print("ERROR: Fake.csv not found!")
    print(FAKE_FILE)
    exit()


if not os.path.exists(TRUE_FILE):
    print("ERROR: True.csv not found!")
    print(TRUE_FILE)
    exit()


# Read CSV files
fake_news = pd.read_csv(FAKE_FILE)

true_news = pd.read_csv(TRUE_FILE)


print("\nFake News Dataset:")
print(fake_news.head())


print("\nTrue News Dataset:")
print(true_news.head())


# ============================================================
# ADD LABELS
# ============================================================

# 0 = Fake News
# 1 = True News

fake_news["label"] = 0

true_news["label"] = 1


# ============================================================
# COMBINE DATASETS
# ============================================================

data = pd.concat(
    [fake_news, true_news],
    axis=0,
    ignore_index=True
)


print("\nCombined Dataset:")
print(data.head())


print("\nTotal records:", len(data))


# ============================================================
# CHECK TEXT COLUMN
# ============================================================

if "text" not in data.columns:

    print("\nERROR: 'text' column not found!")

    print("\nAvailable columns:")
    print(data.columns.tolist())

    exit()


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Keep only letters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PREPARE TEXT
# ============================================================

print("\nCleaning news text...")


data["text"] = data["text"].fillna("")


data["clean_text"] = data["text"].apply(
    clean_text
)


# Remove empty text rows
data = data[
    data["clean_text"].str.strip() != ""
]


# ============================================================
# FEATURES AND LABELS
# ============================================================

X = data["clean_text"]

y = data["label"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\nSplitting dataset...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("Training records:", len(X_train))

print("Testing records:", len(X_test))


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

print("\nCreating TF-IDF Vectorizer...")


vectorizer = TfidfVectorizer(

    stop_words="english",

    max_df=0.7,

    max_features=100000,

    ngram_range=(1, 2)
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


print("TF-IDF conversion completed.")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")


model = LogisticRegression(

    max_iter=1000,

    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed.")


# ============================================================
# MODEL PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("MODEL RESULTS")

print("=" * 60)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "True News"
        ]
    )
)


# ============================================================
# Get actual evaluation metrics
# ============================================================
report = classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "True News"],
    output_dict=True
)

# Calculate performance scores
metrics = {
    "Accuracy": accuracy,
    "Precision": report["weighted avg"]["precision"],
    "Recall": report["weighted avg"]["recall"],
    "F1-Score": report["weighted avg"]["f1-score"]
}

# Create bar graph
plt.figure(figsize=(9, 5))

names = list(metrics.keys())
scores = [value * 100 for value in metrics.values()]

bars = plt.bar(names, scores)

plt.ylim(0, 105)
plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metrics")
plt.title("Figure 6.1: Model Performance Results")

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph for your report
os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/figure_6_1_model_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Figure 6.1 saved successfully!")

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")


joblib.dump(
    model,
    MODEL_FILE
)


joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)


print("\nModel saved successfully!")


print(
    "Model file:"
)

print(MODEL_FILE)


print(
    "\nVectorizer file:"
)

print(VECTORIZER_FILE)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 60)