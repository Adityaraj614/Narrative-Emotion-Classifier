from emotion_inference import predict_emotions
from narrative_features import (
    get_dominant_emotions,
    compute_polarity_sequence,
    compute_volatility
)


def analyze_conversation_trend(model, tokenizer, history, label_names):
    """
    history = list of previous messages
    """

    # =========================
    # EDGE CASE
    # =========================
    if len(history) < 2:
        return {
            "trend": "neutral",
            "volatility": 0.0,
            "dominant_emotions": []
        }

    # =========================
    # STEP 1 — EMOTION PREDICTION
    # =========================
    probs = predict_emotions(model, tokenizer, history, label_names)

    # =========================
    # STEP 2 — DOMINANT EMOTIONS
    # =========================
    dominant = get_dominant_emotions(probs, label_names)

    # =========================
    # STEP 3 — POLARITY SEQUENCE
    # =========================
    polarity_seq = compute_polarity_sequence(dominant)

    # =========================
    # STEP 4 — VOLATILITY
    # =========================
    volatility = compute_volatility(probs)

    # =========================
    # 🔥 STEP 5 — FINAL ROBUST TREND LOGIC
    # =========================
    if len(polarity_seq) >= 2:
        negative_count = polarity_seq.count(-1)
        positive_count = polarity_seq.count(1)

        if negative_count > positive_count:
            trend = "declining"
        elif positive_count > negative_count:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "neutral"

    # =========================
    # FINAL RETURN
    # =========================
    return {
        "trend": trend,
        "volatility": float(volatility),
        "dominant_emotions": dominant
    }