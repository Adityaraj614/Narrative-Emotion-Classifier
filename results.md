# 📊 Phase 6: Model Comparison

## Models Evaluated

1. **Baseline Model**
   - RoBERTa (text-only)
   - No additional features

2. **Hybrid Model**
   - RoBERTa + Narrative Features
   - Features: polarity, volatility, intensity

---

## ⚙️ Training Setup (Same for both models)

- Dataset: GoEmotions
- Batch Size: 16
- Epochs: 4
- Learning Rate: 2e-5
- Loss Function: BCEWithLogitsLoss
- Evaluation Metric: Macro F1 Score

---

## 📈 Results

| Model | Description | Macro F1 |
|------|------------|---------|
| Baseline | RoBERTa | **0.5154** |
| Hybrid | RoBERTa + Narrative Features | **0.51** |

---

## 🔍 Observations

- The baseline model slightly outperforms the hybrid model.
- Performance difference is minimal (~0.005).

---

## 🧠 Key Insights

- RoBERTa already captures strong contextual emotion signals.
- Simple handcrafted narrative features were not sufficient to improve performance.
- Feature design plays a critical role in hybrid models.

---

## ⚠️ Limitations

- Narrative features are derived from label distributions.
- This approach may not generalize to real-time inference.
- Features are relatively simple and may not capture complex narrative flow.

---

## 🚀 Future Work

- Design better narrative features (emotion transitions, sequence modeling)
- Use real-time inference-compatible features
- Integrate model into an emotion-aware chatbot system