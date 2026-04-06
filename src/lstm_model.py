# src/lstm_model.py

import torch
import torch.nn as nn


class EmotionLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_layers=1):
        super(EmotionLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_size, 28)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 768)
        """

        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        x = self.dropout(last_output)
        x = self.fc(x)

        return x