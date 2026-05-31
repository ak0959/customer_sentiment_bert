# Dual-Engine Movie Review Sentiment Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://customersentimentbert-n6emezcfiq7fbm72hqqgbs.streamlit.app)

An interactive NLP evaluation platform that compares a traditional machine learning baseline against a fine-tuned deep learning Transformer. This architecture serves as a performance test-bench to evaluate the real-world trade-offs between inference latency and contextual accuracy.

🔗 **Live Production App:** [Launch Streamlit Dashboard](https://customersentimentbert-n6emezcfiq7fbm72hqqgbs.streamlit.app)

---

## Project Architecture & Step-by-Step Implementation

```text
customer_sentiment_bert/
├── data/
│   └── processed/                # Tokenized and cleaned train/val/test splits
├── models/
│   └── baseline_logreg.pkl       # Serialized traditional baseline model
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_bert_finetuning.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py                # PyTorch Dataset pipeline for tokenization
│   └── model.py                  # Helper functions for inference pipelines
├── requirements.txt              # Production dependency footprint
└── streamlit_app.py              # Live multi-engine UI portal

Phase 1: Environment & Directory Initialization
The project began by establishing a structured workspace modeled after clean software engineering workflows. This setup isolates raw data structures, source scripts, Jupyter development environments, and serialized model binaries into distinct, trackable zones.
