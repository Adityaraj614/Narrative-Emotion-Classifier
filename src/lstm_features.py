import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sequence_embeddings(sentences, tokenizer, roberta):
    """
    sentences = list of strings
    returns (1, seq_len, 768)
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

            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)

    sequence = torch.stack(embeddings, dim=1)

    return sequence