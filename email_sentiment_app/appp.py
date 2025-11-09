import streamlit as st
from transformers import pipeline

st.title("📧 Email Sentiment Analyzer (DistilBERT)")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

email_text = st.text_area("Enter email text here:")

if st.button("Analyze Sentiment"):
    if email_text.strip() == "":
        st.warning("Please enter some email text!")
    else:
        result = model(email_text)[0]
        st.success(f"Sentiment: {result['label']} | Confidence: {result['score']:.2f}")
