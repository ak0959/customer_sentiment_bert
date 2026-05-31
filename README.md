# Dual-Engine Movie Review Sentiment Classifier

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

Phase 2: Exploratory Data Analysis & Text Preprocessing
Using the IMDB movie review dataset, text sequences underwent a clean preprocessing pipeline:

HTML tags, non-alphabetic characters, and system noise were stripped.

Sequences were normalized through lowercasing and whitespacing.

The clean text corpuses were stratified and split into deterministic train_clean.csv, val_clean.csv, and test_clean.csv distributions to prevent target leakage.

Phase 3: Traditional Baseline Modeling
To establish a performance floor, a fast, traditional statistical machine learning engine was developed:

Vectorization: Cleaned text blocks were converted into numerical feature matrices using Term Frequency-Inverse Document Frequency (TF-IDF) unigram and bigram feature extractions.

Classification: A Logistic Regression classifier was trained on the sparse matrices.

Outcome: This baseline achieved high execution speeds, processing inference requests in under 2 milliseconds, but struggled to capture structural context like sarcasm, structural shifts, or complex negations.

Phase 4: Transformer Fine-Tuning (Deep Learning)
To address the contextual blind spots of the baseline, a state-of-the-art Transformer engine was integrated:

Architecture: Selected DistilBERT (distilbert-base-uncased) for its lightweight signature, retaining 95% of BERT’s linguistic capabilities while reducing parameter density by 40%.

Tokenization: Text sequences were processed using Hugging Face's fast tokenizers into dynamic attention masks and input IDs.

Training: The base model was fine-tuned using PyTorch and the Hugging Face Trainer API across multiple epochs, optimizing cross-entropy loss against validation metrics.

Phase 5: Cloud Decoupling & Storage Optimization
To maintain a clean version-control history and bypass GitHub’s strict 100MB repository limit, a strategic design choice was implemented:

Git tracking rules (.gitignore) were set up to completely block heavy binary model layers (*.safetensors, *.pt, *.bin) from the GitHub code tree.

A specialized model repository was established on the Hugging Face Model Hub at amitkadia79/movie-sentiment-distilbert.

The fully trained model weights, tokenizer vocabularies, and network configurations were uploaded directly to the Hugging Face Cloud infrastructure.

Phase 6: Production UI & Live Deployment
The project culminated in a live, user-facing diagnostic application:

Dashboard Logic: A Streamlit application was constructed to handle live text input fields and coordinate parallel backend scoring runs.

Dynamic Streaming: The code uses Hugging Face's transformers client API inside the application layer. When the app initializes in the cloud, it dynamically fetches configuration files and model tensors directly from the Hugging Face Hub, caching them locally for subsequent executions.

Infrastructure Hosting: The repository was linked to Streamlit Community Cloud, creating a continuous deployment pipeline that automatically re-initializes whenever changes land on the main branch.

