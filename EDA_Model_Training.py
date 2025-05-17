import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
import joblib

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Step 2: Load Dataset 
df = pd.read_csv("C:\\ML\\DataSet\\NewLearning\\smart_sentiment_dashboard\\data\\training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]
df['target'] = df['target'].map({0: 0, 2: 1, 4: 2})  # 0: Negative, 1: Neutral, 2: Positive
print(df.head())

# Step 3: Preprocess Tweets
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

print(df['clean_text'])

#4: Visualize Sentiment Distribution
sns.countplot(df['target'])
plt.xticks([0,1,2], ['Negative', 'Neutral', 'Positive'])
plt.title("Sentiment Distribution")
plt.show()


#Step 5: TF-IDF Vectorization

X = df['clean_text']
y = df['target']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

#Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

#Step 7: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Step 8: Evaluate Model
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Step 9: Save Model and Vectorizer
joblib.dump(model, 'C:\\ML\\DataSet\\NewLearning\\smart_sentiment_dashboard\\models\\sentiment_model.pkl')
joblib.dump(vectorizer, 'C:\\ML\\DataSet\\NewLearning\\smart_sentiment_dashboard\\models\\tfidf_vectorizer.pkl')

'''You Can Test This in app.py Like:

from app.utils import load_model, preprocess_text, predict_sentiment

model, vectorizer = load_model()
user_input = "Amazing service and friendly staff!"
cleaned = preprocess_text(user_input)
sentiment = predict_sentiment(cleaned, model, vectorizer)
print(sentiment)

'''
