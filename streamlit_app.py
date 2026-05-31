import streamlit as pd
import streamlit as st
import joblib
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up page layout configurations
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("🎬 Dual-Engine Movie Review Sentiment Classifier")
st.markdown("""
This interactive interface acts as a production test-bench for our Capstone modeling layer. 
Type a custom movie review below to evaluate and compare our traditional linear baseline against our fine-tuned transformer engine.
""")

# Cache model loading to prevent performance bottlenecks on page refreshes
@st.cache_resource
def load_baseline_engine():
    # Relative path from root directory
    return joblib.load("models/baseline_logreg.pkl")

@st.cache_resource
def load_transformer_engine():
    model_path = "amitkadia79/movie-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Initialize models
with st.spinner("Loading analytical engines into system memory..."):
    baseline_model = load_baseline_engine()
    transformer_tokenizer, transformer_model = load_transformer_engine()

# Main user text interface input area
user_review = st.text_area(
    "Enter a review string for sentiment classification:",
    value="The cinematography was breathtaking and the acting was top-tier, but the narrative pacing dragged heavily in the final act, making it a mixed experience.",
    height=120
)

if st.button("Run Head-to-Head Classification"):
    if user_review.strip() == "":
        st.warning("Please input a valid text string to evaluate.")
    else:
        st.markdown("### Engine Performance Metrics")
        col1, col2 = st.columns(2)
        
        # --- Engine 1: TF-IDF + Logistic Regression Baseline ---
        with col1:
            st.subheader("TF-IDF + LogReg Baseline")
            start_time = time.time()
            
            # Predict
            baseline_pred = baseline_model.predict([user_review])[0]
            baseline_probs = baseline_model.predict_proba([user_review])[0]
            
            latency_ms = (time.time() - start_time) * 1000
            
            sentiment = "Positive 🟢" if baseline_pred == 1 else "Negative 🔴"
            confidence = baseline_probs[1] if baseline_pred == 1 else baseline_probs[0]
            
            st.metric(label="Predicted Sentiment", value=sentiment)
            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
            st.metric(label="Inference Latency", value=f"{latency_ms:.2f} ms")
            
        # --- Engine 2: Fine-Tuned DistilBERT Transformer ---
        with col2:
            st.subheader("Fine-Tuned DistilBERT")
            start_time = time.time()
            
            # Process inputs and run tensor prediction
            inputs = transformer_tokenizer(user_review, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = transformer_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
                prediction = torch.argmax(outputs.logits, dim=1).item()
                
            latency_ms = (time.time() - start_time) * 1000
            
            sentiment = "Positive 🟢" if prediction == 1 else "Negative 🔴"
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            st.metric(label="Predicted Sentiment", value=sentiment)
            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
            st.metric(label="Inference Latency", value=f"{latency_ms:.2f} ms")