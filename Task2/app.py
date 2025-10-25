# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===========================
# Load Model and Tokenizers
# ===========================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("model/translator_model.h5")
    with open("model/tokenizers.pkl", "rb") as f:
        tokenizer_eng, tokenizer_fr, tokenizer_hi, max_len = pickle.load(f)
    return model, tokenizer_eng, tokenizer_fr, tokenizer_hi, max_len

model, tokenizer_eng, tokenizer_fr, tokenizer_hi, max_len = load_resources()

# ===========================
# Helper Functions
# ===========================
def vectorize_input(text, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text.lower()])
    return pad_sequences(seq, maxlen=max_len, padding='post')

def decode_prediction(pred, tokenizer):
    index = np.argmax(pred, axis=-1)
    word = [k for k, v in tokenizer.word_index.items() if v == index[0]]
    return word[0] if word else "N/A"

def translate_text(text):
    X = vectorize_input(text, tokenizer_eng, max_len)
    pred_fr, pred_hi = model.predict(X)
    french = decode_prediction(pred_fr, tokenizer_fr)
    hindi = decode_prediction(pred_hi, tokenizer_hi)
    return french, hindi

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Dual Language Translator", page_icon="🌍")
st.title("🌍 Dual Language Translator")
st.markdown("Translate **English → French + Hindi** (only if ≥ 10 letters)")

english_text = st.text_input("Enter an English word or sentence:")

if st.button("Translate"):
    cleaned = english_text.replace(" ", "")
    if len(cleaned) < 10:
        st.warning("⚠️ Upload again — The text must contain 10 or more letters.")
    else:
        french, hindi = translate_text(english_text)
        st.success(f"**French Translation:** {french}")
        st.success(f"**Hindi Translation:** {hindi}")
