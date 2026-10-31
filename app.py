
import os
import re
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


# ============================================================
# CHECK AND LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Model not found. Run: python train_with_ifnd.py"
    )

if not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(
        "TF-IDF vectorizer not found. Run: python train_with_ifnd.py"
    )

print("Loading trained model...")

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

print("Model loaded successfully!")


# ============================================================
# READ CSV FILE
# ============================================================

def read_csv_file(path):

    for encoding in ("utf-8", "cp1252", "latin-1"):

        try:
            return pd.read_csv(path, encoding=encoding)

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read: {path}")


# ============================================================
# DATASET STATISTICS
# ============================================================

def get_dataset_stats():

    stats = {
        "fake_count": 0,
        "true_count": 0,
        "ifnd_count": 0,
        "combined_count": 0
    }

    fake_path = os.path.join(DATASET_DIR, "Fake.csv")
    true_path = os.path.join(DATASET_DIR, "True.csv")
    ifnd_path = os.path.join(DATASET_DIR, "IFND.csv")

    if os.path.exists(fake_path):
        stats["fake_count"] = len(read_csv_file(fake_path))

    if os.path.exists(true_path):
        stats["true_count"] = len(read_csv_file(true_path))

    if os.path.exists(ifnd_path):
        stats["ifnd_count"] = len(read_csv_file(ifnd_path))

    # Count unique non-empty text records

    all_texts = []

    for filename in ("Fake.csv", "True.csv", "IFND.csv"):

        path = os.path.join(DATASET_DIR, filename)

        if not os.path.exists(path):
            continue

        df = read_csv_file(path)

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        text_columns = [
            column
            for column in ("title", "text", "statement")
            if column in df.columns
        ]

        if text_columns:

            texts = (
                df[text_columns]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.strip()
            )

            all_texts.extend(
                texts[texts != ""].tolist()
            )

    stats["combined_count"] = len(set(all_texts))

    return stats


# ============================================================
# DATASET DETAILS AND COLUMN HEADINGS
# ============================================================

def get_dataset_details():

    dataset_files = [
        "Fake.csv",
        "True.csv",
        "IFND.csv"
    ]

    datasets = []

    for filename in dataset_files:

        file_path = os.path.join(DATASET_DIR, filename)

        if os.path.exists(file_path):

            try:

                df = read_csv_file(file_path)

                datasets.append({
                    "name": filename,
                    "rows": len(df),
                    "columns": df.columns.tolist()
                })

            except Exception as error:

                print(
                    f"Error reading {filename}: {error}"
                )

    return datasets


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    stats = get_dataset_stats()

    datasets = get_dataset_details()

    return render_template(
        "index.html",
        stats=stats,
        datasets=datasets
    )


# ============================================================
# NEWS PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "success": False,
                "message": "No data received."
            }), 400

        news_text = data.get("news", "").strip()

        if not news_text:

            return jsonify({
                "success": False,
                "message": "Please enter a news article."
            }), 400

        cleaned_text = clean_text(news_text)

        news_vector = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            news_vector
        )[0]

        probabilities = model.predict_proba(
            news_vector
        )[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        fake_probability = float(
            class_probabilities.get(0, 0) * 100
        )

        true_probability = float(
            class_probabilities.get(1, 0) * 100
        )

        confidence = max(probabilities) * 100

        result = (
            "FAKE NEWS"
            if prediction == 0
            else "TRUE NEWS"
        )

        return jsonify({

            "success": True,

            "result": result,

            "confidence": round(
                float(confidence), 2
            ),

            "fake_probability": round(
                fake_probability, 2
            ),

            "true_probability": round(
                true_probability, 2
            )

        })

    except Exception as error:

        app.logger.exception(
            "Prediction error"
        )

        return jsonify({

            "success": False,

            "message": (
                "An error occurred while predicting the news."
            )

        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)

    print("       FAKE NEWS DETECTION WEB APP")

    print("=" * 50)

    print("Open: http://127.0.0.1:5000/\n")

    app.run(
        debug=True,
        port=5000
    )
