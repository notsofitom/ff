# %%


# %%
import pandas as pd

# Load your emails.csv
df = pd.read_csv("emails.csv")

# Inspect columns
df.head()
df.columns


# %%
import nltk
nltk.download('stopwords')
import re

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop = set(stopwords.words('english'))

def clean_email(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop)
    return text

df['cleaned_text'] = df['message'].apply(clean_email)


df.head()


# %%
from transformers import pipeline

# Load pre-trained DistilBERT sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Example prediction
example_email = df['cleaned_text'][0]
result = sentiment_model(example_email)[0]
print(result)


# %%
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

st.title("📧 Email Sentiment Analyzer (with NEUTRAL Support)")

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()[0]
    label_id = np.argmax(scores)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[label_id], float(np.max(np.exp(scores) / np.sum(np.exp(scores))))

email_text = st.text_area("Enter email text:")

if st.button("Analyze Sentiment"):
    if email_text.strip() == "":
        st.warning("Please enter some email text!")
    else:
        sentiment, confidence = predict_sentiment(email_text)
        st.success(f"Sentiment: {sentiment} | Confidence: {confidence:.2f}")


# %%



