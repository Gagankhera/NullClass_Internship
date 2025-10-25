# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# ===========================
# 1️⃣ Load Dataset
# ===========================
with open("data/english.txt", "r", encoding="utf-8") as f:
    english_texts = [line.strip().lower() for line in f.readlines()]

with open("data/french.txt", "r", encoding="utf-8") as f:
    french_texts = [line.strip().lower() for line in f.readlines()]

with open("data/hindi.txt", "r", encoding="utf-8") as f:
    hindi_texts = [line.strip().lower() for line in f.readlines()]

assert len(english_texts) == len(french_texts) == len(hindi_texts), "❌ Dataset lines mismatch!"

# ===========================
# 2️⃣ Tokenization
# ===========================
def tokenize(texts):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding='post')
    return tokenizer, padded

tokenizer_eng, X = tokenize(english_texts)
tokenizer_fr, y_fr = tokenize(french_texts)
tokenizer_hi, y_hi = tokenize(hindi_texts)

max_len = X.shape[1]

# ===========================
# 3️⃣ Model Architecture
# ===========================
input_dim = len(tokenizer_eng.word_index) + 1
output_dim_fr = len(tokenizer_fr.word_index) + 1
output_dim_hi = len(tokenizer_hi.word_index) + 1

inputs = Input(shape=(max_len,))
embed = Embedding(input_dim, 128)(inputs)

lstm = LSTM(128)(embed)   # ✅ FIXED: removed time_major=False

out_fr = Dense(output_dim_fr, activation="softmax", name="french_output")(lstm)
out_hi = Dense(output_dim_hi, activation="softmax", name="hindi_output")(lstm)

model = Model(inputs, [out_fr, out_hi])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

print(model.summary())

# ===========================
# 4️⃣ Prepare Output Labels
# ===========================
y_fr = np.expand_dims(y_fr[:, 0], -1)
y_hi = np.expand_dims(y_hi[:, 0], -1)

# ===========================
# 5️⃣ Train the Model
# ===========================
model.fit(X, [y_fr, y_hi], epochs=200, verbose=1)

# ===========================
# 6️⃣ Save Model and Tokenizers
# ===========================
os.makedirs("model", exist_ok=True)
model.save("model/translator_model.h5")

with open("model/tokenizers.pkl", "wb") as f:
    pickle.dump((tokenizer_eng, tokenizer_fr, tokenizer_hi, max_len), f)

print("✅ Model and tokenizers saved successfully!")