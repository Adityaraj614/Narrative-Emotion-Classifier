# src/trend_analyzer.py

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

    if len(history) < 2:
        return {
            "trend": "neutral",
            "volatility": 0,
            "dominant_emotions": []
        }

    # Step 1 — Predict emotions for entire history
    probs = predict_emotions(model, tokenizer, history, label_names)

    # Step 2 — Extract dominant emotions
    dominant = get_dominant_emotions(probs, label_names)

    # Step 3 — Compute polarity
    polarity_seq = compute_polarity_sequence(dominant)

    # Step 4 — Compute volatility
    volatility = compute_volatility(probs)

    # Step 5 — Detect trend
    
    # 🔥 Advanced trend logic
    if len(polarity_seq) >= 2:
        if polarity_seq[-1] < polarity_seq[0]:
            trend = "declining"
        elif polarity_seq[-1] > polarity_seq[0]:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "neutral"
    
    return {
        "trend": trend,
        "volatility": float(volatility),
        "dominant_emotions": dominant
    }