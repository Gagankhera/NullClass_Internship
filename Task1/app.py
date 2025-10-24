import streamlit as st
from PIL import Image
import tempfile, os, cv2
import numpy as np
from ocr_model_tf import build_crnn
from utils import preprocess_image_pil, is_english
from translate import load_translator, translate_texts

st.set_page_config(page_title="OCR Translator", page_icon="🧠", layout="wide")
st.title("📄 OCR Translator (English-only)")

# Sidebar
st.sidebar.header("Settings")
target_lang = st.sidebar.selectbox("Translate English →", ["es", "fr", "de", "hi", "zh"])
st.sidebar.info("Only English words will be processed.")

uploaded_file = st.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "mp4", "mov", "avi"])
run_button = st.button("Run OCR + Translation")

# Load translation model
@st.cache_resource
def get_translator(lang):
    return load_translator(lang)

tokenizer, translator_model = get_translator(target_lang)

# Load OCR model (for demonstration, no training)
@st.cache_resource
def load_ocr_model():
    model = build_crnn()
    # model.load_weights("ocr_weights.h5")  # if trained
    return model

ocr_model = load_ocr_model()

def simulate_ocr_output(image):
    """Simulated OCR result for demo"""
    # You can replace this with actual OCR model inference
    return "hello world"  # placeholder

if run_button and uploaded_file:
    suffix = uploaded_file.name.split(".")[-1].lower()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
    tmp_file.write(uploaded_file.getbuffer())
    tmp_path = tmp_file.name

    if suffix in ["png", "jpg", "jpeg"]:
        image = Image.open(tmp_path).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        text = simulate_ocr_output(image)
        st.subheader("📝 Extracted Text")
        st.write(text)

        if is_english(text):
            translated = translate_text([text], tokenizer, translator_model)[0]
            st.subheader(f"🌐 Translated ({target_lang})")
            st.write(translated)
        else:
            st.warning("The detected text is not English. No translation performed.")

    else:
        # Video input
        cap = cv2.VideoCapture(tmp_path)
        st.write("Processing video frames...")

        frame_count = 0
        all_texts = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 != 0:  # sample 1 frame per second if 30fps
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)
            text = simulate_ocr_output(frame_img)
            if is_english(text):
                all_texts.append(text)

        cap.release()

        st.subheader("📝 Extracted English Texts")
        st.write(all_texts)

        if all_texts:
            translated = translate_text(all_texts, tokenizer, translator_model)
            st.subheader(f"🌐 Translated ({target_lang})")
            for orig, trans in zip(all_texts, translated):
                st.write(f"**{orig}** → {trans}")
