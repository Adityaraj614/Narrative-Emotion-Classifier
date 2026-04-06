# src/lstm_data.py

from data_loader import load_data


def create_sequences(dataset, seq_len=3):
    texts = dataset["train"]["text"]

    sequences = []

    for i in range(len(texts) - seq_len + 1):
        seq = texts[i:i + seq_len]
        sequences.append(seq)

    return sequences


if __name__ == "__main__":
    dataset = load_data()
    sequences = create_sequences(dataset)

    print("Sample sequence:")
    for s in sequences[0]:
        print(s)