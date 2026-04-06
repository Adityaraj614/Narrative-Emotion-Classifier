# src/response_generator.py

import random

RESPONSE_TEMPLATES = {
    "frustration": [
        "I'm really sorry you're facing this issue.",
        "That sounds frustrating. Let me help you with it.",
        "I understand how frustrating this can be.",
        "This seems really annoying, let’s fix it together.",
    ],

    "distress": [
        "I'm sorry you're feeling this way.",
        "That sounds tough. I'm here for you.",
        "I understand this is difficult for you.",
        "This seems hard, but we’ll work through it.",
    ],

    "anxiety": [
        "I understand your concern.",
        "It's okay to feel this way, let's work through it.",
        "I can see why you're worried.",
        "Let’s take this step by step, we’ll figure it out.",
    ],

    "confusion": [
        "Let me explain this clearly.",
        "I can help clarify that for you.",
        "No worries, I’ll break this down for you.",
        "Let’s go through this step by step.",
    ],

    "positive": [
        "That's great to hear!",
        "Awesome, glad things are going well!",
        "I'm really happy things worked out for you!",
        "That's a great improvement!",
        "Nice! Keep it going!",
    ],

    "neutral": [
        "I'm here to help.",
        "Tell me more about that.",
        "Could you explain a bit more?",
        "I'm listening."
    ]
}


def generate_response(interpreted_emotions, trend=None, lstm_signal=None):
    if not interpreted_emotions:
        return "I'm here to help."

    primary_emotion, primary_score = interpreted_emotions[0]

        # 🔥 LSTM-based signal (basic usage)
    if lstm_signal is not None:
        if lstm_signal > 0.5:
            return "It seems like this issue has been building up over time. Let’s try to resolve it step by step."

    # =========================
    # 🔥 TREND-AWARE RESPONSES
    # =========================

    if trend == "declining":
        return random.choice([
            "I can see this is getting more frustrating over time. Let’s try to fix this together.",
            "It seems things are getting worse, I’m here to help you through this.",
        ])

    if trend == "stable" and primary_emotion in ["frustration", "distress"]:
        return random.choice([
            "I understand this has been consistently frustrating. Let me help you sort it out.",
            "It seems this issue has been bothering you for a while. Let’s fix it.",
        ])

    if trend == "improving" and primary_emotion == "positive":
        return random.choice([
            "I’m glad things are getting better. That’s great to hear!",
            "Nice! It looks like things are improving.",
        ])

    # =========================
    # 🔥 STRONG SINGLE EMOTION
    # =========================

    if primary_score > 0.6:
        responses = RESPONSE_TEMPLATES.get(primary_emotion, RESPONSE_TEMPLATES["neutral"])
        return random.choice(responses)

    # =========================
    # 🔥 MULTI-EMOTION LOGIC
    # =========================

    if len(interpreted_emotions) > 1:
        second_emotion, _ = interpreted_emotions[1]

        if primary_emotion == "anxiety" and second_emotion == "positive":
            return "I understand your concern, but things can improve."

        if primary_emotion == "frustration" and second_emotion == "confusion":
            return "I understand this is frustrating. Let me explain it clearly."

        if primary_emotion == "distress" and second_emotion == "positive":
            return "I know this is tough, but things can get better."

        r1 = random.choice(RESPONSE_TEMPLATES.get(primary_emotion, [""]))
        r2 = random.choice(RESPONSE_TEMPLATES.get(second_emotion, [""]))

        return f"{r1} {r2}"

    # =========================
    # 🔥 FALLBACK
    # =========================

    responses = RESPONSE_TEMPLATES.get(primary_emotion, RESPONSE_TEMPLATES["neutral"])
    return random.choice(responses)