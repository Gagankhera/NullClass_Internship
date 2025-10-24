# utils.py
import numpy as np
import cv2
from PIL import Image
import csv
import os
from langdetect import detect

# character set: lowercase a-z
CHARS = list("abcdefghijklmnopqrstuvwxyz")  # 26 chars
NUM_CLASSES = len(CHARS) + 1  # plus blank

IMG_H = 32
IMG_W = 128

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def load_labels(csv_path, images_dir):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r["filename"]
            text = r["text"].lower()
            rows.append((os.path.join(images_dir, fname), text))
    return rows

def text_to_labels(text):
    # map char -> index in CHARS; unknown chars are ignored
    out = []
    for c in text.lower():
        if c in CHARS:
            out.append(CHARS.index(c))
    return np.array(out, dtype=np.int32)

def preprocess_image_pil(img):
    # expects PIL.Image
    im = img.convert("L")
    im = im.resize((IMG_W, IMG_H))
    arr = np.array(im).astype("float32") / 255.0
    arr = np.expand_dims(arr, -1)
    return arr

def data_generator(rows, batch_size=32, shuffle=True):
    n = len(rows)
    idxs = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_idx = idxs[start:end]
            images = []
            labels = []
            label_lengths = []
            for i in batch_idx:
                path, text = rows[i]
                img = Image.open(path)
                images.append(preprocess_image_pil(img))
                lbl = text_to_labels(text)
                labels.append(lbl)
                label_lengths.append(len(lbl))
            images = np.stack(images, axis=0)
            # convert variable-length labels to a flat array for ctc_batch_cost usage
            # K.ctc_batch_cost expects labels as dense tensor (batch, max_label_len)
            max_label_len = max(label_lengths)
            labels_padded = np.ones((len(labels), max_label_len), dtype=np.int32) * -1
            for i, l in enumerate(labels):
                labels_padded[i, :len(l)] = l
            # compute input_length: time steps of network output
            # approximate: after conv/pool, time dimension ~ IMG_W / 4 (depends on architecture)
            time_steps = images.shape[2] // 4  # approximation; adjust if needed
            input_length = np.ones((len(images), 1), dtype=np.int32) * time_steps
            label_length = np.array(label_lengths).reshape(-1,1).astype(np.int32)
            yield (images, labels_padded, input_length, label_length)
