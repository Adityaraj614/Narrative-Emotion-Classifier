import torch
from transformers import AutoTokenizer
from model import EmotionClassifier
from data_loader import get_label_names, load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "roberta-base"


def load_model(num_labels):
    model = EmotionClassifier(num_labels=num_labels)

    model.load_state_dict(torch.load("models/emotion_model.pt"))
    print("✅ Trained model loaded")

    model.to(DEVICE)
    model.eval()

    return model


def predict_emotions(model, tokenizer, sentences, label_names):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(outputs).cpu().numpy()

    return probs


if __name__ == "__main__":
    print("Running emotion inference...")

    dataset = load_data()
    label_names = get_label_names(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(len(label_names))

    # Sample sequence
    sentences = [
        "I am very happy today",
        "But I am also a bit nervous",
        "I hope everything goes well"
    ]

    probs = predict_emotions(model, tokenizer, sentences, label_names)

    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")

        top_indices = probs[i].argsort()[-3:][::-1]

        for idx in top_indices:
            print(f"{label_names[idx]}: {probs[i][idx]:.3f}")