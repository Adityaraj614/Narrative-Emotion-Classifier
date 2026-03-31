import torch
import numpy as np
from transformers import AutoTokenizer

from hybrid_model import HybridEmotionModel
from data_loader import get_label_names, load_data

from narrative_features import (
    get_dominant_emotions,
    compute_polarity_sequence,
    compute_volatility
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "roberta-base"


# =========================
# Load Model
# =========================
def load_model(num_labels):
    model = HybridEmotionModel(
        num_labels=num_labels,
        num_features=3  # same as training
    )

    model.load_state_dict(torch.load("models/emotion_model.pt"))
    print("✅ Trained model loaded")

    model.to(DEVICE)
    model.eval()

    return model


# =========================
# Compute Narrative Features
# =========================
def compute_features(probs, label_names):
    features = []

    for p in probs:
        dominant = get_dominant_emotions([p], label_names)[0]
        polarity = compute_polarity_sequence([dominant])[0]
        volatility = compute_volatility(p)
        intensity = max(p)

        # same scaling used in training
        volatility *= 10

        features.append([polarity, volatility, intensity])

    return torch.tensor(features, dtype=torch.float32).to(DEVICE)


# =========================
# Prediction Function
# =========================
def predict_emotions(model, tokenizer, sentences, label_names):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        # -------------------------
        # PASS 1: Initial prediction (no features)
        # -------------------------
        outputs = model.roberta(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        temp_features = torch.zeros((len(sentences), 3)).to(DEVICE)

        combined = torch.cat((cls_embedding, temp_features), dim=1)
        logits = model.classifier(combined)

        probs = torch.sigmoid(logits).cpu().numpy()

        # -------------------------
        # PASS 2: Compute features
        # -------------------------
        features = compute_features(probs, label_names)

        # -------------------------
        # PASS 3: Final prediction
        # -------------------------
        logits = model(
            inputs["input_ids"],
            inputs["attention_mask"],
            features
        )

        probs = torch.sigmoid(logits).cpu().numpy()

    return probs


# =========================
# Test Block
# =========================
if __name__ == "__main__":
    print("Running emotion inference...")

    dataset = load_data()
    label_names = get_label_names(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(len(label_names))

    sentences = [
        "I am very happy today",
        "But I am also a bit nervous",
        "I hope everything goes well"
    ]

    probs = predict_emotions(model, tokenizer, sentences, label_names)

    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")

        top_indices = probs[i].argsort()[-3:][::-1]

        for idx in top_indices:
            print(f"{label_names[idx]}: {probs[i][idx]:.3f}")