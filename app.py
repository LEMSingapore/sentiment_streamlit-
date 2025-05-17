import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pickle

# Download once; Streamlit will cache it
nltk.download('punkt', quiet=True)

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # Adjust these filenames if yours differ
    model = tf.keras.models.load_model('sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def summarize_text(text: str, sentences_count: int = 3) -> str:
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary)

def predict_sentiment(model, tokenizer, text: str, maxlen: int = 200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    return label, float(pred)

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")

st.markdown(
    """
    Enter a movie review below, choose how many sentences to keep in the summary,  
    then hit **Analyze** to see the summary and the predicted sentiment.
    """
)

review_text = st.text_area("Movie review text", height=200)
num_sentences = st.slider("Summary length (sentences)", min_value=1, max_value=5, value=3)

if st.button("Analyze"):
    if not review_text.strip():
        st.warning("Please enter some review text first.")
    else:
        with st.spinner("Summarizing and analyzing..."):
            summary = summarize_text(review_text, sentences_count=num_sentences)
            model, tokenizer = load_model_and_tokenizer()
            label, score = predict_sentiment(model, tokenizer, summary)

        st.subheader("🔍 Summary")
        st.write(summary)

        st.subheader("😊 Sentiment Prediction")
        st.write(f"**{label}** (confidence: {score:.2f})")
