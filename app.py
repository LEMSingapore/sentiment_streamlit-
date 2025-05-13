import streamlit as st
from tensorflow.keras.models import load_model
import pickle

# 1. Load model and tokenizer (adjust paths as needed)
@st.cache_resource
def load_sentiment_model():
    model = load_model("model/sentiment_model.h5")
    with open("tokenizer/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_sentiment_model()

# 2. Preprocessing helper
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text, tokenizer, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=maxlen)

# 3. Streamlit UI
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review to see whether it's positive or negative.")

user_input = st.text_area("Your review", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        x = preprocess_text(user_input, tokenizer)
        score = model.predict(x)[0][0]
        label = "👍 Positive" if score > 0.5 else "👎 Negative"
        st.subheader(label)
        st.write(f"Confidence: {score:.2%}")
