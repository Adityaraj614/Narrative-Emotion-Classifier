from data_loader import load_data

def create_sequences(dataset, seq_len=3):
    texts = dataset["train"]["text"]

    sequences = []

    for i in range(len(texts) - seq_len + 1):
        seq = texts[i:i+seq_len]
        sequences.append(seq)

    return sequences


if __name__ == "__main__":
    print("Creating narrative sequences...")

    dataset = load_data()

    sequences = create_sequences(dataset, seq_len=3)

    print(f"Total sequences: {len(sequences)}")

    print("\nSample sequence:")
    for i, sentence in enumerate(sequences[0]):
        print(f"S{i+1}: {sentence}")