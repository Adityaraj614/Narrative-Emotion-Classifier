# src/emotion_mapper.py

# Mapping raw emotions → categories
EMOTION_MAP = {
    "anger": "frustration",
    "annoyance": "frustration",

    "sadness": "distress",
    "disappointment": "distress",
    "grief": "distress",
    "remorse": "distress",

    "fear": "anxiety",
    "nervousness": "anxiety",

    "confusion": "confusion",

    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "admiration": "positive",
    "approval": "positive",
    "gratitude": "positive",
    "excitement": "positive",

    "neutral": "neutral"
}


def map_emotions_to_categories(top_emotions):
    category_scores = {}

    for item in top_emotions:
        emotion = item["emotion"]
        score = item["confidence"]

        category = EMOTION_MAP.get(emotion, "neutral")

        if category not in category_scores:
            category_scores[category] = 0

        category_scores[category] += score

    return category_scores


def normalize_scores(category_scores):
    total = sum(category_scores.values())

    if total == 0:
        return category_scores

    return {
        k: v / total
        for k, v in category_scores.items()
    }


def interpret_emotions(top_emotions):
    category_scores = map_emotions_to_categories(top_emotions)
    normalized = normalize_scores(category_scores)

    sorted_categories = sorted(
        normalized.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_categories