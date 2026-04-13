# 🧠 Narrative-Aware Emotion Classification & Chatbot System
An AI-powered system for narrative-aware emotion classification, with a lightweight chatbot interface for real-time demonstration.
---

# 📌 1. Introduction — What is this project?

This project focuses on building an **emotion-aware conversational system** that goes beyond traditional text-based emotion classification.

Most existing models analyze emotions **sentence-by-sentence**, treating each input independently.

However, real human conversations are **continuous narratives**, where emotions evolve over time.

👉 This project aims to bridge that gap by introducing:

- Narrative-aware emotion understanding  
- Temporal emotion tracking  
- Real-time conversational interaction  

---

# 🎯 2. Intent & Motivation — Why this project?

In real-world applications like:

- Customer support  
- Mental health systems  
- AI assistants  

Understanding **how emotions change over time** is more important than detecting emotion in a single sentence.

Traditional systems:
- Detect emotion per message ❌  
- Ignore emotional progression ❌  

Our goal:
> To build a system that understands **emotional flow across a conversation** and responds accordingly.

## ⚠️ Note on Chatbot Interface

The chatbot UI in this project is designed as a **demonstration layer** to showcase the system’s capabilities.

The primary focus of the project is:

- Emotion classification
- Narrative-aware modeling
- Temporal emotion tracking

The chatbot is not intended to be a fully developed conversational agent, but rather a way to visualize:

- Emotion predictions  
- Trend detection  
- System behavior in real time

---

### 📊 Emotion Classification (Static Models)

Early approaches focused on classifying emotions at the sentence level using:

- Traditional ML models (SVM, Naive Bayes)
- Deep learning models (CNN, LSTM)

With the introduction of transformers:

- **BERT** and **RoBERTa** significantly improved performance
- Models trained on datasets like **GoEmotions (Google)** enabled fine-grained emotion classification (27 emotion labels)

👉 Limitation:
These models treat each sentence independently and ignore conversational context.

---

### 💬 Context-Aware Emotion Detection

Recent work explores emotion detection in conversations:

- Models consider **previous utterances** for better understanding  
- Use architectures like:
  - LSTM over dialogue sequences  
  - Transformer-based dialogue models  

👉 Limitation:
Most approaches still focus on classification, not **emotion evolution**

---

### ⚠️ Gap Identified

Despite advancements, existing systems:

- Focus on **static emotion classification**  
- Do not explicitly model **narrative flow of emotions**  
- Rarely combine:
  - Emotion detection  
  - Temporal tracking  
  - Interactive response generation  

---

### 💡 Our Contribution

This project addresses these gaps by:

- Modeling **emotion as a dynamic narrative process**  
- Integrating:
  - Transformer-based classification  
  - Temporal modeling (LSTM)  
  - Trend analysis (improving / declining / stable)  
- Providing a **real-time interactive interface** to visualize emotional evolution  

## 🧠 System Architecture
```
User Input  
↓  
RoBERTa → Emotion Scores  
↓  
Narrative Features (polarity, volatility, intensity)  
↓  
LSTM (sequence modeling across conversation)  
↓  
Fusion Layer (Hybrid + Temporal signals)  
↓  
Trend Analyzer (improving / declining / stable)  
↓  
Response Generator  
↓  
Chatbot Output
```
---

# 🌍 4. Where can this be used?

This system can be applied in:

- 💬 Customer Support Chatbots  
- 🧠 Mental Health Assistants  
- 🤖 AI Companions  
- 🎮 Game AI (emotion-aware NPCs)  
- 📊 Sentiment Tracking Systems  

---

## 🧩 Project Structure

```
Narrative-Emotion-Classifier/
│
├── src/
│   ├── api.py                  # Flask backend (main entry point)
│   ├── model.py                # Baseline RoBERTa model
│   ├── hybrid_model.py         # Hybrid model (RoBERTa + features)
│   ├── data_loader.py          # Dataset loading & preprocessing
│   ├── narrative_features.py   # Narrative feature extraction
│   ├── train.py                # Training pipeline for models
│   │
│   ├── emotion_mapper.py       # Maps model output → readable emotion
│   ├── conversation_memory.py  # Stores conversation history
│   ├── trend_analyzer.py       # Detects emotion trend (📈 / ⚠️)
│   ├── response_generator.py   # Generates contextual responses
│   │
│   ├── lstm_model.py           # LSTM for temporal modeling
│   ├── lstm_features.py        # Feature prep for LSTM
│   ├── lstm_data.py            # LSTM dataset processing
│   ├── train_lstm.py           # LSTM training script
│   │
│   ├── static/
│   │   ├── style.css           # UI styling (glassmorphism)
│   │   ├── script.js           # Frontend logic (chat + graph)
│   │
│   ├── templates/
│   │   └── index.html          # Chat UI layout
│   │
│   └── test_response.py        # Testing response pipeline
│
├── models/                     # Saved model weights (.pt files)
├── README.md
└── requirements.txt
```

---


# 🎮  How to Use the System

## 🛠️ How to Run
```
git clone https://github.com/Adityaraj614/Narrative-Emotion-Classifier.git
cd Narrative-Emotion-Classifier
pip install -r requirements.txt
python src/api.py
```
Open in browser:
http://127.0.0.1:5000/ui
---

🧪 Example Interaction
this is not working
still broken
this is getting worse

👉 System detects:

Emotion → frustration
Trend → ⚠️ declining

Then:

okay it is better now
yes it works
great it is fixed

👉 System detects:

Trend → 📈 improving

## 📸 Demo

### 🔻 Declining Emotion (Frustration Build-up)

![Declining Emotion](assets/declining.png)

---

### 📈 Improving Emotion (Recovery Phase)

![Improving Emotion](assets/improving.png)

---

### 🔄 Mixed Emotion (Confusion → Understanding)

![Mixed Emotion](assets/mixed.png)

## 🧠 9. Key Learnings
Emotion is not static — it evolves
Transformers are strong, but lack explicit temporal reasoning
System design matters as much as model design

## ⚠️ 10. Limitations
Trend logic is rule-based
LSTM can be improved further
No backend database (uses local storage)

## 🚀 11. Future Improvements
Multi-chat sessions (like ChatGPT)
Database integration
Better temporal modeling
Deployment as web app
Improved emotion calibration
Advanced UI enhancements
LLM integration 

🤝 Contributions

Open to feedback, suggestions, and collaboration.
