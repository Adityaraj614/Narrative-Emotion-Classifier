import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "roberta-base"


class EmotionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(EmotionClassifier, self).__init__()

        print("Loading RoBERTa model...")

        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

        print("Model initialized successfully ✅")

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]

        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits


# Test block
if __name__ == "__main__":
    print("Running model test...")

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones((2, 10))

    model = EmotionClassifier(num_labels=28)

    outputs = model(input_ids, attention_mask)

    print("Output shape:", outputs.shape)
    print("Model working correctly ✅")