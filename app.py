import streamlit as st
import pandas as pd
from app.utils import load_model, preprocess_text, predict_sentiment, generate_wordcloud

# Load model and vectorizer
model, vectorizer = load_model()

# Streamlit App UI
st.set_page_config(page_title="Smart Sentiment Analyzer", layout="centered")
st.title("🧠 Smart Sentiment Analyzer")
st.markdown("Analyze the sentiment of your feedback or tweet using NLP and machine learning!")

# Text input
user_input = st.text_area("✍️ Enter your text below:", placeholder="e.g., Amazing service and friendly staff!")

# Prediction section
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = preprocess_text(user_input)
        prediction = predict_sentiment(cleaned, model, vectorizer)
        st.success(f"✅ Sentiment Prediction: **{prediction}**")

# Optional: Upload CSV file and generate word clouds
st.markdown("---")
st.subheader("📄 Upload Dataset to Visualize Word Clouds by Sentiment (Optional)")
uploaded_file = st.file_uploader("Upload a CSV file with two columns: 'text' and 'target' (0=Neg, 1=Neu, 2=Pos)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = df[['text', 'target']]
        df['cleaned_text'] = df['text'].apply(preprocess_text)

        st.success("✅ File uploaded and processed successfully.")

        # Generate word clouds by sentiment
        sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for code, label in sentiments.items():
            texts = df[df['target'] == code]['cleaned_text'].tolist()
            if texts:
                st.markdown(f"### ☁️ Word Cloud for {label} Tweets")
                generate_wordcloud(texts, title=f"{label} Tweets")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
