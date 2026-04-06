from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer
from emotion_inference import load_model, predict_emotions
from data_loader import load_data, get_label_names
import numpy as np
import torch

from emotion_mapper import interpret_emotions
from response_generator import generate_response
from conversation_memory import ConversationMemory
from trend_analyzer import analyze_conversation_trend

# 🔥 LSTM imports
from lstm_model import EmotionLSTM
from lstm_features import get_sequence_embeddings


app = Flask(__name__)

MODEL_NAME = "roberta-base"

print("🔄 Loading model...")

dataset = load_data()
label_names = get_label_names(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = load_model(len(label_names))

print("✅ Model loaded successfully")

# =========================
# 🔥 LSTM INIT
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = EmotionLSTM().to(DEVICE)
lstm_model.load_state_dict(torch.load("models/lstm_model.pt"))
lstm_model.eval()

# =========================
# 🔥 Global conversation memory
# =========================
memory = ConversationMemory(max_history=10)


# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return "Emotion API is running 🚀"

@app.route("/ui")
def ui():
    return render_template("index.html")
# =========================
# Single Sentence Prediction
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'text' in JSON"
        }), 400

    text = data["text"]

    probs = predict_emotions(model, tokenizer, [text], label_names)

    top_indices = np.argsort(probs[0])[-3:][::-1]

    top_emotions = [
        {
            "emotion": label_names[i],
            "confidence": float(probs[0][i])
        }
        for i in top_indices
    ]

    return jsonify({
        "status": "success",
        "text": text,
        "top_emotions": top_emotions
    })


# =========================
# 🔥 CHAT ENDPOINT (FINAL)
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "status": "error",
            "message": "Please provide 'text'"
        }), 400

    text = data["text"]

    # 🔥 Step 1 — Store message
    memory.add_message(text)

    # 🔥 Step 2 — Predict emotions
    probs = predict_emotions(model, tokenizer, [text], label_names)

    top_indices = np.argsort(probs[0])[-3:][::-1]

    top_emotions = [
        {
            "emotion": label_names[i],
            "confidence": float(probs[0][i])
        }
        for i in top_indices
    ]

    # 🔥 Step 3 — Interpret emotions
    interpreted = interpret_emotions(top_emotions)

    # 🔥 Step 4 — Trend analysis
    trend_result = analyze_conversation_trend(
        model,
        tokenizer,
        memory.get_history(),
        label_names
    )

    trend = trend_result["trend"]

    # =========================
    # 🔥 Step 5 — LSTM SIGNAL
    # =========================
    lstm_signal = None

    history = memory.get_history()

    if len(history) >= 2:
        seq_embeddings = get_sequence_embeddings(history)

        with torch.no_grad():
            seq_embeddings = seq_embeddings.to(DEVICE)
            lstm_output = lstm_model(seq_embeddings)

            # simple scalar signal
            lstm_signal = torch.mean(lstm_output).item()

    # 🔥 Step 6 — Generate response
    response = generate_response(
        interpreted,
        trend=trend,
        lstm_signal=lstm_signal
    )

    return jsonify({
        "status": "success",
        "input_text": text,
        "raw_emotions": top_emotions,
        "interpreted_emotions": interpreted,
        "trend": trend,
        "lstm_signal": lstm_signal,
        "response": response,
        "history": memory.get_history()
    })


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=True)