import numpy as np

# =========================
# POLARITY GROUPS
# =========================
POSITIVE = {
    "joy", "love", "optimism", "admiration",
    "approval", "gratitude", "excitement"
}

NEGATIVE = {
    "sadness", "anger", "fear", "disgust",
    "remorse", "disappointment", "nervousness", "grief",
    "annoyance", "confusion"   # 🔥 FIXED (added missing)
}


# =========================
# DOMINANT EMOTION
# =========================
def get_dominant_emotions(probs, label_names):
    dominant = []

    for p in probs:
        idx = np.argmax(p)
        dominant.append(label_names[idx])

    return dominant


# =========================
# POLARITY MAPPING
# =========================
def get_polarity(emotion):
    if emotion in POSITIVE:
        return 1
    elif emotion in NEGATIVE:
        return -1
    else:
        return 0  # neutral


# =========================
# POLARITY SEQUENCE
# =========================
def compute_polarity_sequence(dominant_emotions):
    return [get_polarity(e) for e in dominant_emotions]


# =========================
# VOLATILITY (IMPROVED)
# =========================
def compute_volatility(probs):
    """
    Measures variation across predictions
    """
    probs = np.array(probs)

    if len(probs.shape) == 1:
        return float(np.var(probs))

    # 🔥 Better sequence-aware volatility
    return float(np.mean(np.var(probs, axis=0)))


# =========================
# TEST BLOCK
# =========================
if __name__ == "__main__":
    label_names = ["joy", "fear", "optimism"]

    probs = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7]
    ])

    dominant = get_dominant_emotions(probs, label_names)
    polarity_seq = compute_polarity_sequence(dominant)
    volatility = compute_volatility(probs)

    print("Dominant emotions:", dominant)
    print("Polarity sequence:", polarity_seq)
    print("Volatility:", volatility)