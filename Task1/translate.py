# translate.py
from transformers import MarianMTModel, MarianTokenizer
import torch

def load_translator(target_lang="es"):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_texts(texts, tokenizer, model, device="cpu"):
    model.to(device)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    translated = model.generate(**inputs)
    out = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return out
