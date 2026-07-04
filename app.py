from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Tokenization
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load data
df = pd.read_csv('frd.csv')
texts = df["text_"].tolist()
labels = df["label"].tolist()

# Train/Test Split
train_texts, _, train_labels, _ = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Build model (from your original code)
alpha = 1
word_counts = {}
total_words = {}

for text, label in zip(train_texts, train_labels):
    if label not in word_counts:
        word_counts[label] = {}
        total_words[label] = 0
    tokens = tokenize(text)
    for token in tokens:
        word_counts[label][token] = word_counts[label].get(token, 0) + 1
        total_words[label] += 1

vocab = set()
for counts in word_counts.values():
    vocab.update(counts.keys())
V = len(vocab)

unique_labels = list(word_counts.keys())
label1, label2 = unique_labels[0], unique_labels[1]

log_odds = {}
for word in vocab:
    count1 = word_counts[label1].get(word, 0)
    count2 = word_counts[label2].get(word, 0)
    prob1 = (count1 + alpha) / (total_words[label1] + alpha * V)
    prob2 = (count2 + alpha) / (total_words[label2] + alpha * V)
    log_odds[word] = np.log(prob1) - np.log(prob2)

num_label1 = sum(1 for l in train_labels if l == label1)
num_label2 = sum(1 for l in train_labels if l == label2)
prior_log_odds = np.log(num_label1 / len(train_labels)) - np.log(num_label2 / len(train_labels))

def classify_text(text, threshold=0):
    tokens = tokenize(text)
    score = prior_log_odds
    for token in tokens:
        if token in log_odds:
            score += log_odds[token]
    predicted_label = label1 if score > threshold else label2
    confidence = abs(score)
    return predicted_label, confidence

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    review = data['review'].strip()
    
    if not review:
        return jsonify({"error": "Review cannot be empty. Please enter a review."}), 400

    prediction, confidence = classify_text(review)
    response = {
        "prediction": prediction,
        "confidence": round(confidence, 3)
    }
    return jsonify(response)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

