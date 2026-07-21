import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('vader_lexicon')

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# loading the models saved from the notebook
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
best_model = joblib.load('models/best_model.pkl')

LABELS = {1: 'CG', 0: 'OR'}   # CG = fake, OR = real


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    review = (data.get('review') or '').strip()

    if not review:
        return jsonify({"error": "Review cannot be empty."}), 400

    cleaned = preprocess(review)
    X = vectorizer.transform([cleaned])

    pred = best_model.predict(X)[0]

    # confidence score depending on model type
    if hasattr(best_model, 'predict_proba'):
        confidence = float(max(best_model.predict_proba(X)[0]))
    else:
        score = best_model.decision_function(X)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))

    return jsonify({
        "prediction": LABELS[int(pred)],
        "confidence": round(confidence, 3)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
