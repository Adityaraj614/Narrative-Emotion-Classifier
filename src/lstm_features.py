# src/lstm_features.py

import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
roberta.eval()


def get_sequence_embeddings(sentences):
    """
    sentences = list of strings (sequence)
    returns tensor of shape (1, seq_len, 768)
    """

    embeddings = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(DEVICE)

            outputs = roberta(**inputs)

            # CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

            embeddings.append(cls_embedding)

    # Stack into sequence
    sequence = torch.stack(embeddings, dim=1)  # (1, seq_len, 768)

    return sequence