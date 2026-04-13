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
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD LSTM
# =========================
lstm_model = EmotionLSTM().to(DEVICE)
lstm_model.load_state_dict(torch.load("models/lstm_model.pt", weights_only=True))
lstm_model.eval()

# =========================
# MEMORY
# =========================
memory = ConversationMemory(max_history=10)


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Emotion API is running 🚀"

@app.route("/ui")
def ui():
    return render_template("index.html")


# =========================
# SINGLE PREDICT
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
# CHAT (FINAL)
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

    # =========================
    # STEP 1 — MEMORY
    # =========================
    memory.add_message(text)
    history = memory.get_history()[-5:]

    # =========================
    # STEP 2 — HYBRID EMOTION
    # =========================
    probs = predict_emotions(model, tokenizer, [text], label_names)

    # =========================
    # STEP 3 — LSTM SIGNAL + FUSION
    # =========================
    lstm_signal = None

    if len(history) >= 2:
        # 🔥 FIXED: use shared RoBERTa
        seq_embeddings = get_sequence_embeddings(
            history,
            tokenizer,
            model.roberta
        ).to(DEVICE)

        with torch.no_grad():
            lstm_output = lstm_model(seq_embeddings)

            # LSTM emotion vector
            lstm_vector = lstm_output.squeeze().cpu().numpy()

            # Sequence-based variation
            seq_np = seq_embeddings.squeeze().cpu().numpy()

            first_step = seq_np[0]
            last_step = seq_np[-1]

            delta = np.abs(last_step - first_step)

            lstm_signal = {
                "intensity": float(np.max(lstm_vector)),
                "variation": float(np.mean(delta))
            }

            # Fusion
            alpha = 0.7
            beta = 0.3
            probs[0] = alpha * probs[0] + beta * lstm_vector

    # =========================
    # STEP 4 — TOP EMOTIONS
    # =========================
    top_indices = np.argsort(probs[0])[-3:][::-1]

    top_emotions = [
        {
            "emotion": label_names[i],
            "confidence": float(probs[0][i])
        }
        for i in top_indices
    ]

    # =========================
    # STEP 5 — INTERPRET
    # =========================
    interpreted = interpret_emotions(top_emotions)

    # =========================
    # STEP 6 — TREND
    # =========================
    trend_result = analyze_conversation_trend(
        model,
        tokenizer,
        memory.get_history(),
        label_names
    )

    trend = trend_result["trend"]

    # =========================
    # STEP 7 — RESPONSE
    # =========================
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
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)