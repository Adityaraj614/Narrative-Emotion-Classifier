import torch
import torch.nn as nn
from transformers import RobertaModel

class HybridEmotionModel(nn.Module):
    def __init__(self, num_labels, num_features):
        super(HybridEmotionModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        self.dropout = nn.Dropout(0.3)

        # Combine RoBERTa + narrative features
        self.classifier = nn.Sequential(
            nn.Linear(768 + num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

        combined = torch.cat((cls_embedding, features), dim=1)

        logits = self.classifier(combined)

        return logits