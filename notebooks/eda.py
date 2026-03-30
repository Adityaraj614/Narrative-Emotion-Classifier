from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

def load_data():
    dataset = load_dataset("go_emotions")
    return dataset

def get_label_names(dataset):
    return dataset["train"].features["labels"].feature.names

def multi_hot(labels, num_classes):
    vec = np.zeros(num_classes)
    for label in labels:
        vec[label] = 1
    return vec

def preprocess(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_names = get_label_names(dataset)
    num_classes = len(label_names)

    def tokenize(example):
        encoding = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoding["labels"] = multi_hot(example["labels"], num_classes)
        return encoding

    dataset = dataset.map(tokenize, batched=False)

    # Set format for PyTorch
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset, label_names


if __name__ == "__main__":
    dataset = load_data()
    dataset, label_names = preprocess(dataset)

    print(dataset)

    sample = dataset["train"][0]
    print("\nSample Processed:")
    print(sample)