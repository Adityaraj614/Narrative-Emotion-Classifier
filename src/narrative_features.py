import numpy as np

# Define polarity mapping
POSITIVE = {"joy", "love", "optimism", "admiration", "approval", "gratitude", "excitement"}
NEGATIVE = {"sadness", "anger", "fear", "disgust", "remorse", "disappointment", "nervousness", "grief"}

def get_dominant_emotions(probs, label_names):
    dominant = []

    for p in probs:
        idx = np.argmax(p)
        dominant.append(label_names[idx])

    return dominant


def get_polarity(emotion):
    if emotion in POSITIVE:
        return 1
    elif emotion in NEGATIVE:
        return -1
    else:
        return 0  # neutral


def compute_polarity_sequence(dominant_emotions):
    return [get_polarity(e) for e in dominant_emotions]


def compute_volatility(probs):
    return np.var(probs)


if __name__ == "__main__":
    # Example (use your previous output)
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