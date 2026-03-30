# 🧠 Narrative-Aware Emotion Classification

An advanced emotion classification system that combines transformer-based contextual understanding with narrative-level features to analyze emotions in text.

---

## 🚀 Project Overview

This project explores whether incorporating **narrative-aware features** (like emotional polarity shifts and volatility) can improve emotion classification beyond standard transformer models.

We compare:

- ✅ **Baseline Model** → RoBERTa (text-only)
- 🔥 **Hybrid Model** → RoBERTa + Narrative Features

---


---

## 📊 Dataset

- Dataset used: **GoEmotions**
- Multi-label emotion classification
- 28 emotion categories

---

## ⚙️ Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- NumPy

---

# 🧩 Project Phases

---

## 🟢 Phase 0: Project Setup

- Initialized GitHub repository
- Set up project structure
- Created virtual environment
- Installed dependencies

---

## 🟢 Phase 1: Baseline Model Development

- Implemented **RoBERTa-based classifier**
- Used CLS token representation
- Added dropout + linear classification layer

👉 File: `model.py`

---

## 🟢 Phase 2: Data Pipeline

- Loaded GoEmotions dataset using HuggingFace
- Tokenized text using RoBERTa tokenizer
- Converted labels into **multi-hot vectors**
- Prepared dataset for PyTorch training

👉 File: `data_loader.py`

---

## 🟢 Phase 3: Narrative Feature Engineering

Designed features to capture emotional flow:

- 🔹 **Dominant Emotion**
- 🔹 **Polarity Mapping** (positive / negative / neutral)
- 🔹 **Polarity Sequence**
- 🔹 **Volatility (variance of emotion probabilities)**

👉 File: `narrative_features.py`

---

## 🟢 Phase 4: Hybrid Model Design

- Combined:
  - RoBERTa CLS embedding (768-dim)
  - Narrative features (3-dim)
- Concatenated features and passed through classifier

👉 File: `hybrid_model.py`

---

## 🟢 Phase 5: Hybrid Training Pipeline

- Built full training loop
- Computed narrative features dynamically
- Used:
  - BCEWithLogitsLoss (multi-label)
  - AdamW optimizer
- Evaluated using **Macro F1 Score**

👉 File: `train.py`

---

## 🟢 Phase 6: Model Comparison

### Goal:
Compare baseline vs hybrid under identical conditions

### ⚖️ Fair Comparison Setup:
- Same dataset
- Same preprocessing
- Same batch size (16)
- Same epochs (4)
- Same learning rate (2e-5)
- Same evaluation metric (Macro F1)

---

## 📈 Results

| Model | Description | Macro F1 |
|------|------------|---------|
| Baseline | RoBERTa | **0.5154** |
| Hybrid | RoBERTa + Narrative Features | **0.51** |

---

## 🔍 Key Observations

- Baseline slightly outperforms hybrid model
- Performance difference is minimal (~0.005)
- RoBERTa captures strong contextual signals

---

## 🧠 Insights

- Simple narrative features are not sufficient to improve performance
- Feature design is critical in hybrid architectures
- Transformer models already encode significant emotional context

---

## ⚠️ Limitations

- Narrative features are derived from label distributions
- Not directly usable in real-time inference
- Limited feature complexity

---

## 🚀 Future Work

- Develop richer narrative features (emotion transitions, temporal modeling)
- Make features inference-compatible
- Integrate into an **emotion-aware chatbot system**
- Explore sequence models (LSTM / Transformer over conversations)

---

## 💡 Key Takeaway

This project demonstrates that:

> Adding features is not enough — meaningful feature design and integration are crucial to improving model performance.

---

## 🤝 Contributions

Open to improvements, suggestions, and collaborations!

---

## 📬 Contact

If you found this project interesting, feel free to connect!