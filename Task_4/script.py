import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tkinter import Tk, Label, Entry, Button
# Load data from CSV file
data = pd.read_csv("./french_to_tamil.csv")

# Strip any leading/trailing spaces in French sentences
data['French'] = data['French'].str.strip()
data['French_length'] = data['French'].str.len()

# Optionally print lengths to verify
print(data[['French', 'French_length']].drop_duplicates())

# Skip filtering by length (or adjust based on your data)
filtered_data = data

# Preprocess data
french_sentences = filtered_data['French'].tolist()
tamil_sentences = filtered_data['Tamil'].tolist()

# Check if lists are non-empty
if not french_sentences:
    raise ValueError("No French sentences found. Please check your data.")
if not tamil_sentences:
    raise ValueError("No Tamil sentences found. Please check your data.")

tokenizer_french = Tokenizer()
tokenizer_tamil = Tokenizer()

tokenizer_french.fit_on_texts(french_sentences)
tokenizer_tamil.fit_on_texts(tamil_sentences)

french_sequences = tokenizer_french.texts_to_sequences(french_sentences)
tamil_sequences = tokenizer_tamil.texts_to_sequences(tamil_sentences)

# Ensure sequences are non-empty
if not french_sequences:
    raise ValueError("French sequences are empty after tokenization. Check the tokenizer.")

max_french_length = max(len(seq) for seq in french_sequences)
max_tamil_length = max(len(seq) for seq in tamil_sequences)

french_padded = pad_sequences(french_sequences, maxlen=max_french_length, padding="post")
tamil_padded = pad_sequences(tamil_sequences, maxlen=max_french_length, padding="post")  # Ensure same length for targets

# Build the model
model = Sequential()
model.add(Embedding(len(tokenizer_french.word_index) + 1, 128, input_length=max_french_length))
model.add(LSTM(128, return_sequences=True))  # Return sequences to match target shape
model.add(Dense(len(tokenizer_tamil.word_index) + 1, activation='softmax'))  # Output shape (sequence_length, vocab_size)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(french_padded, np.expand_dims(tamil_padded, -1), epochs=10, batch_size=32, validation_split=0.2)

# Create the GUI
root = Tk()
root.title("French-to-Tamil Translator")

french_label = Label(root, text="Enter a French word:")
french_label.pack()

french_entry = Entry(root)
french_entry.pack()

translate_button = Button(root, text="Translate", command=lambda: translate())
translate_button.pack()

tamil_label = Label(root, text="Translated Tamil word:")
tamil_label.pack()

tamil_output = Label(root)
tamil_output.pack()

def translate():
    french_word = french_entry.get()
    french_seq = tokenizer_french.texts_to_sequences([french_word])
    french_padded = pad_sequences(french_seq, maxlen=max_french_length, padding="post")
    predicted_seq = model.predict(french_padded)
    predicted_indices = np.argmax(predicted_seq[0], axis=-1)
    predicted_words = [tokenizer_tamil.index_word.get(idx, '') for idx in predicted_indices]
    tamil_word = ''.join(predicted_words).strip()
    tamil_output.config(text=tamil_word)

root.mainloop()
