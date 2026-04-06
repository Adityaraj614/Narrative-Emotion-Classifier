# src/test_lstm_features.py

from lstm_features import get_sequence_embeddings

sentences = [
    "not working",
    "still not working",
    "very frustrated"
]

seq = get_sequence_embeddings(sentences)

print("Shape:", seq.shape)