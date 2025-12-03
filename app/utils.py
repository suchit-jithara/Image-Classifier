# app/utils.py
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

import torch
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP once (model selection: base/large can be changed)
MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None

def get_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor

def image_to_embedding(image_path):
    model, processor = get_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
    emb = image_emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb.astype(np.float32)

def texts_to_embedding(texts, batch_size=16):
    model, processor = get_clip()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        emb = text_emb.cpu().numpy()
        # normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
        all_embs.append(emb.astype(np.float32))
    return np.vstack(all_embs)

def qdrant_client(url=None):
    url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)
