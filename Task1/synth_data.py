# synth_data.py
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import csv
from tqdm import tqdm

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
FONTS_DIR = os.path.join(OUT_DIR, "fonts")
TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "val")
LABELS_DIR = os.path.join(OUT_DIR, "labels")

# ✅ Ensure all directories exist
for d in [OUT_DIR, FONTS_DIR, TRAIN_DIR, VAL_DIR, LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Word list (English only) ---
WORDS = [
    "hello", "world", "openai", "test", "example", "stream", "image", "text", "apple", "banana",
    "computer", "vision", "translate", "language", "model", "data", "train", "validation", "sample", "demo",
    "camera", "video", "streamlit", "python", "code", "ocr", "learning", "network", "pixel", "font"
]

def available_fonts():
    """Find all fonts in the fonts folder."""
    fonts = []
    if os.path.isdir(FONTS_DIR):
        for f in os.listdir(FONTS_DIR):
            if f.lower().endswith((".ttf", ".otf")):
                fonts.append(os.path.join(FONTS_DIR, f))
    return fonts

FONTS = available_fonts()

def random_bg(w, h):
    """Generate a simple noisy background image."""
    base = np.ones((h, w, 3), dtype=np.uint8) * np.random.randint(200, 255)
    noise = (np.random.randn(h, w, 3) * np.random.randint(0, 30)).astype(np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def draw_word(word, out_path, imgw=128, imgh=32):
    """Render a single word image."""
    bg = random_bg(imgw, imgh)
    draw = ImageDraw.Draw(bg)

    # Choose font
    if FONTS:
        font_path = random.choice(FONTS)
        size = random.randint(14, 24)
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Text color
    fill = tuple([random.randint(0, 80) for _ in range(3)])

    # ✅ Use textbbox (Pillow ≥10) or fallback to textsize
    try:
        bbox = draw.textbbox((0, 0), word, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.textsize(word, font=font)

    # Random position ensuring text fits
    x = max(1, random.randint(1, max(1, imgw - w - 1)))
    y = max(1, random.randint(0, max(1, imgh - h - 1)))
    draw.text((x, y), word, font=font, fill=fill)

    # Random augmentations
    if random.random() < 0.3:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.random() * 1.2))
    if random.random() < 0.2:
        bg = bg.rotate(random.uniform(-5, 5), expand=False, fillcolor=(255, 255, 255))

    bg.save(out_path)

def generate(out_dir, labels_csv, n_images=2000):
    """Generate synthetic OCR dataset and save labels CSV."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    rows = []
    for i in tqdm(range(n_images), desc=f"Generating {out_dir}"):
        word = random.choice(WORDS)
        fname = f"{i}_{word}.png"
        out_path = os.path.join(out_dir, fname)
        draw_word(word, out_path)
        rows.append((fname, word))

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text"])
        writer.writerows(rows)

if __name__ == "__main__":
    print("🖼️ Generating synthetic OCR training data...")
    generate(TRAIN_DIR, os.path.join(LABELS_DIR, "train_labels.csv"), n_images=4000)
    generate(VAL_DIR, os.path.join(LABELS_DIR, "val_labels.csv"), n_images=800)
    print("✅ Done! Check the 'data/train' and 'data/val' folders.")
