import re
import string
import joblib
import numpy as np

import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔹 Load model and vectorizer
def load_model():
    model = joblib.load("C:\\ML\\DataSet\\NewLearning\\smart_sentiment_dashboard\\models\\sentiment_model.pkl")
    vectorizer = joblib.load("C:\\ML\\DataSet\\NewLearning\\smart_sentiment_dashboard\\models\\tfidf_vectorizer.pkl")
    return model, vectorizer

# 🔹 Preprocess text (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# 🔹 Predict sentiment
def predict_sentiment(text, model, vectorizer):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map.get(prediction, "Unknown")

# 🔹 Generate word cloud from list of texts
def generate_wordcloud(texts, title='Word Cloud'):
    all_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
