# 💬 Smart Sentiment Dashboard

A Streamlit-based app that performs sentiment analysis on individual texts or bulk tweets using a trained Logistic Regression model.

## 🔍 Features

- Analyze individual tweets or sentences
- Upload CSV for batch sentiment analysis
- View sentiment distribution
- Generate word clouds from the text

## 🧠 Built With

- Python, scikit-learn, nltk, Streamlit
- Dataset: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart_sentiment_dashboard.git
cd smart_sentiment_dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/smart_sentiment_dashboard.py
```

## 🧪 Folder Structure

```
models/                  → Trained model and vectorizer
app/
  ├── sentiment_utils.py → Preprocessing & prediction functions
  └── smart_sentiment_dashboard.py → Streamlit app
```

## 📌 Notes

- You need to download the Sentiment140 dataset and train the model if not included.
- Pretrained models (`sentiment_model.pkl`, `tfidf_vectorizer.pkl`) should be placed inside the `models/` directory.

## 📄 License

MIT License
