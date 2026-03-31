from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from emotion_inference import load_model, predict_emotions
from data_loader import load_data, get_label_names
import numpy as np

app = Flask(__name__)

MODEL_NAME = "roberta-base"

print("🔄 Loading model...")

dataset = load_data()
label_names = get_label_names(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = load_model(len(label_names))

print("✅ Model loaded successfully")


# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return "Emotion API is running 🚀"


# =========================
# Single Sentence Prediction
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON"}), 400

    text = data["text"]
    sentences = [text]

    probs = predict_emotions(model, tokenizer, sentences, label_names)

    top_indices = np.argsort(probs[0])[-3:][::-1]

    results = [
        {
            "emotion": label_names[i],
            "confidence": float(probs[0][i])
        }
        for i in top_indices
    ]

    return jsonify({
        "input": text,
        "top_emotions": results
    })


# =========================
# Narrative Prediction (USP)
# =========================
@app.route("/predict_narrative", methods=["POST"])
def predict_narrative():
    data = request.get_json()

    # Input validation
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' as a list"}), 400

    texts = data["text"]

    if not isinstance(texts, list):
        return jsonify({"error": "'text' must be a list of sentences"}), 400

    # Model inference
    probs = predict_emotions(model, tokenizer, texts, label_names)

    # Per sentence analysis
    sentence_results = []

    for i, sentence in enumerate(texts):
        top_indices = np.argsort(probs[i])[-3:][::-1]

        top_emotions = [
            {
                "emotion": label_names[idx],
                "confidence": float(probs[i][idx])
            }
            for idx in top_indices
        ]

        sentence_results.append({
            "sentence": sentence,
            "top_emotions": top_emotions
        })

    # Narrative features
    from narrative_features import (
        get_dominant_emotions,
        compute_polarity_sequence,
        compute_volatility
    )

    dominant = get_dominant_emotions(probs, label_names)
    polarity_seq = compute_polarity_sequence(dominant)
    volatility = compute_volatility(probs)

    # -------------------------
    # Narrative Summary + Trend
    # -------------------------

    # Trend detection
    if polarity_seq[0] == 1 and polarity_seq[-1] == -1:
        trend = "declining"
    elif polarity_seq[0] == -1 and polarity_seq[-1] == 1:
        trend = "improving"
    else:
        trend = "stable"

    # Summary
    summary = f"Emotion shifts from {dominant[0]} to {dominant[-1]} showing a {trend} emotional trend"

    # Final response
    return jsonify({
        "num_sentences": len(texts),
        "sentence_analysis": sentence_results,
        "narrative_analysis": {
            "dominant_emotions": dominant,
            "polarity_sequence": polarity_seq,
            "volatility": float(volatility),
            "trend": trend,
            "summary": summary
        }
    })


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)