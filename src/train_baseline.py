import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score

from data_loader import load_data, preprocess
from model import EmotionClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAME SETTINGS AS HYBRID (IMPORTANT ⚖️)
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5


def train():
    print("🚀 Starting BASELINE training pipeline...")

    # =========================
    # Load Data (SAME AS HYBRID)
    # =========================
    dataset = load_data()
    print("✅ Dataset loaded")

    dataset, label_names = preprocess(dataset)
    print("✅ Preprocessing complete")

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)

    print("✅ DataLoader ready")

    # =========================
    # Model (BASELINE)
    # =========================
    model = EmotionClassifier(num_labels=len(label_names)).to(DEVICE)
    print("✅ Baseline model loaded on", DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # SAME as hybrid

    # =========================
    # Training Loop
    # =========================
    for epoch in range(EPOCHS):
        print(f"\n🔥 Starting Epoch {epoch+1}")

        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].float().to(DEVICE)

            optimizer.zero_grad()

            # ❌ NO narrative features here
            outputs = model(input_ids, attention_mask)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Loss: {avg_loss:.4f}")

        evaluate(model, val_loader)

    # =========================
    # Save Model
    # =========================
    torch.save(model.state_dict(), "models/baseline_model.pt")
    print("💾 Baseline model saved!")


def evaluate(model, dataloader):
    print("🔍 Evaluating BASELINE...")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask)

            probs = torch.sigmoid(outputs).cpu().numpy()

            # SAME threshold as hybrid (IMPORTANT ⚖️)
            preds = (probs > 0.15).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"🎯 BASELINE Validation Macro F1: {f1:.4f}")


if __name__ == "__main__":
    train()