import os
import re
import joblib

from flask import Flask, render_template, request, jsonify


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "fake_news_model.pkl"
)

VECTORIZER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

# if not os.path.exists(MODEL_FILE):

#     print("ERROR: Machine Learning model not found!")

#     print(
#         "Please run this command first:"
#     )

#     print(
#         "python fake_news.py"
#     )

#     exit()

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "fake_news_model.pkl not found in models folder."
    )

if not os.path.exists(VECTORIZER_FILE):
    raise FileNotFoundError(
        "tfidf_vectorizer.pkl not found in models folder."
    )


if not os.path.exists(VECTORIZER_FILE):

    print("ERROR: TF-IDF vectorizer not found!")

    print(
        "Please run this command first:"
    )

    print(
        "python fake_news.py"
    )

    exit()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")


model = joblib.load(
    MODEL_FILE
)


vectorizer = joblib.load(
    VECTORIZER_FILE
)


print("Model loaded successfully!")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

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
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# NEWS PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Get JSON request
        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "message":
                    "No data received."

            }), 400


        # Get news text
        news_text = data.get(
            "news",
            ""
        ).strip()


        # Check empty input
        if not news_text:

            return jsonify({

                "success": False,

                "message":
                    "Please enter a news article."

            }), 400


        # Clean text
        cleaned_text = clean_text(
            news_text
        )


        # Convert text to TF-IDF
        news_vector = vectorizer.transform(
            [cleaned_text]
        )


        # Make prediction
        prediction = model.predict(
            news_vector
        )[0]


        # Get prediction probability
        probabilities = model.predict_proba(
            news_vector
        )[0]


        confidence = max(
            probabilities
        ) * 100


        # Convert numeric label to result
        if prediction == 0:

            result = "FAKE NEWS"

        else:

            result = "TRUE NEWS"


        # Return result
        return jsonify({

            "success": True,

            "result": result,

            "confidence":
                round(
                    confidence,
                    2
                )

        })


    except Exception as error:

        print(
            "Prediction error:",
            error
        )


        return jsonify({

            "success": False,

            "message":
                "An error occurred while predicting the news."

        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "        FAKE NEWS DETECTION WEB APP"
    )

    print("=" * 60)

    print(
        "\nOpen this URL in your browser:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()


if __name__ == "__main__":
    app.run(debug=True, port=5000)