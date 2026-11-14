
import os
import re
import joblib
import pandas as pd
import requests
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import xml.etree.ElementTree as ET

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


# ============================================================
# LOAD CNN AND MOBILENETV2 IMAGE MODELS
# ============================================================

CNN_MODEL_FILE = os.path.join(
    MODEL_DIR, "cnn_image_model.keras"
)

MOBILENET_MODEL_FILE = os.path.join(
    MODEL_DIR, "mobilenetv2_image_model.keras"
)

if not os.path.exists(CNN_MODEL_FILE):
    raise FileNotFoundError("CNN image model not found!")

if not os.path.exists(MOBILENET_MODEL_FILE):
    raise FileNotFoundError("MobileNetV2 image model not found!")

print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_FILE)

print("Loading MobileNetV2 model...")
mobilenet_model = tf.keras.models.load_model(
    MOBILENET_MODEL_FILE
)

print("All three models loaded successfully!")

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

    evaluation = {
        "accuracy": "98.65%",
        "auc": "0.9985",
        "confusion_matrix": [
            [4482, 72],
            [47, 4236]
        ]
    }

    return render_template(
        "index.html",
        stats=stats,
        datasets=datasets,
        evaluation=evaluation
    )

    
# ============================================================
# NEWS PREDICTION
# ============================================================
# ============================================================
# INTERNET NEWS VERIFICATION
# ============================================================

def verify_news_online(news_text):

    try:
        # Prepare search query
        search_query = " ".join(news_text.split()[:15])

        url = "https://news.google.com/rss/search"

        params = {
            "q": search_query,
            "hl": "en-IN",
            "gl": "IN",
            "ceid": "IN:en"
        }

        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        articles = []

        for item in root.findall(".//item")[:5]:

            source = item.find("source")

            articles.append({

                "title": item.findtext("title", ""),

                "link": item.findtext("link", ""),

                "published": item.findtext("pubDate", ""),

                "source": (
                    source.text
                    if source is not None
                    else "Unknown"
                )

            })

        if articles:

            return {
                "status": "RELATED_ARTICLES_FOUND",
                "message": "Related news articles found online.",
                "articles": articles
            }

        return {
            "status": "NO_MATCHING_ARTICLES_FOUND",
            "message": "No related news articles found.",
            "articles": []
        }

    except Exception:

        app.logger.exception(
            "Internet verification error"
        )

        return {
            "status": "UNAVAILABLE",
            "message": "Internet news search is currently unavailable.",
            "articles": []
        }
   # ============================================================
# NEWS PREDICTION + INTERNET NEWS SEARCH
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

        # --------------------------------
        # 1. MACHINE LEARNING PREDICTION
        # --------------------------------

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

        # DEBUG INFORMATION
        print("MODEL CLASSES:", model.classes_)
        print("PREDICTION:", prediction)
        print("PROBABILITIES:", probabilities)

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

        # --------------------------------
        # 2. INTERNET NEWS SEARCH
        # --------------------------------

        internet_result = verify_news_online(
            news_text
        )

        # --------------------------------
        # 3. SEND RESPONSE
        # --------------------------------

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
            ),

            "internet_verification": internet_result

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
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_file, image_size, model_type):

    img = Image.open(image_file).convert("RGB")

    img = img.resize(image_size)

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "cnn":
        img_array = img_array / 255.0

    elif model_type == "mobilenet":
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

    return img_array


# ============================================================
# COMMON IMAGE PREDICTION
# ============================================================

def predict_image_model(image_file, selected_model, model_type):

    if model_type == "cnn":
        image_array = prepare_image(
            image_file, (224, 224), "cnn"
        )
    else:
        image_array = prepare_image(
            image_file, (160, 160), "mobilenet"
        )

    probability = float(
        selected_model.predict(image_array, verbose=0)[0][0]
    )

    # Class mapping: FAKE = 0, REAL = 1

    if probability >= 0.5:
        result = "TRUE NEWS"
        confidence = probability * 100
    else:
        result = "FAKE NEWS"
        confidence = (1 - probability) * 100

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "fake_probability": round((1 - probability) * 100, 2),
        "true_probability": round(probability * 100, 2)
    }


# ============================================================
# CNN IMAGE PREDICTION
# ============================================================

@app.route("/predict_image", methods=["POST"])
def predict_cnn_image():

    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "Please upload an image."
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "message": "No image selected."
            }), 400

        result = predict_image_model(
            image_file,
            cnn_model,
            "cnn"
        )

        return jsonify({
            "success": True,
            "model": "CNN",
            **result
        })

    except Exception as error:
        app.logger.exception("CNN image prediction error")

        return jsonify({
            "success": False,
            "message": "CNN image prediction failed. Please upload a valid image."
        }), 500


# ============================================================
# MOBILENETV2 IMAGE PREDICTION
# ============================================================

@app.route("/predict_mobilenet", methods=["POST"])
def predict_mobilenet_image():

    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "Please upload an image."
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "message": "No image selected."
            }), 400

        result = predict_image_model(
            image_file,
            mobilenet_model,
            "mobilenet"
        )

        return jsonify({
            "success": True,
            "model": "MobileNetV2",
            **result
        })

    except Exception as error:
        app.logger.exception("MobileNetV2 image prediction error")

        return jsonify({
            "success": False,
            "message": "MobileNetV2 image prediction failed. Please upload a valid image."
        }), 500

# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\n" + "=" * 50)
    print("       FAKE NEWS DETECTION WEB APP")
    print("=" * 50)

    print(f"Server running on port: {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )