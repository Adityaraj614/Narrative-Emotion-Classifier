import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from lstm_model import EmotionLSTM
from lstm_features import get_sequence_embeddings
from data_loader import load_data, get_label_names
from emotion_inference import load_model, predict_emotions
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# SETTINGS
# =========================
EPOCHS = 5
LR = 1e-3
SEQ_LEN = 3


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(texts, seq_len=3):
    sequences = []

    for i in range(len(texts) - seq_len):
        seq = texts[i:i + seq_len]
        target = texts[i + seq_len]

        sequences.append((seq, target))

    return sequences


# =========================
# TRAIN FUNCTION
# =========================
def train():
    print("🚀 Training LSTM...")

    # =========================
    # Load dataset
    # =========================
    dataset = load_data()
    texts = dataset["train"]["text"][:5000]

    label_names = get_label_names(dataset)

    # 🔥 Load tokenizer + hybrid model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    emotion_model = load_model(len(label_names))

    # =========================
    # Create sequences
    # =========================
    sequences = create_sequences(texts, SEQ_LEN)
    print(f"Total sequences: {len(sequences)}")

    # =========================
    # LSTM Model
    # =========================
    lstm_model = EmotionLSTM().to(DEVICE)

    optimizer = Adam(lstm_model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    lstm_model.train()

    # =========================
    # Training Loop
    # =========================
    for epoch in range(EPOCHS):
        total_loss = 0

        print(f"\n🔥 Epoch {epoch+1}")

        for seq, target_text in tqdm(sequences):

            # 🔥 FIXED: Use SAME RoBERTa (no duplication)
            seq_embeddings = get_sequence_embeddings(
                seq,
                tokenizer,
                emotion_model.roberta
            ).to(DEVICE)

            # 🔥 Target emotion vector
            probs = predict_emotions(
                emotion_model,
                tokenizer,
                [target_text],
                label_names
            )

            target_vector = torch.tensor(
                probs[0],
                dtype=torch.float32
            ).to(DEVICE)

            # 🔥 Forward pass
            output = lstm_model(seq_embeddings).squeeze()

            # 🔥 Loss
            loss = loss_fn(output, target_vector)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(sequences)
        print(f"✅ Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    # =========================
    # Save model
    # =========================
    torch.save(lstm_model.state_dict(), "models/lstm_model.pt")
    print("💾 LSTM model saved!")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()