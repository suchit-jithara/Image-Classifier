# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.utils import image_to_embedding, qdrant_client
import numpy as np
from pydantic import BaseModel
import io
from PIL import Image
import os

app = FastAPI(title="SwiftBid CLIP Classifier")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "labels"
TOP_K = 5

@app.get("/")
async def root():
    return {"status": "ok"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), top_k: int = TOP_K):
#     # read image bytes
#     data = await file.read()
#     try:
#         image = Image.open(io.BytesIO(data)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image")

#     # temporary save to compute embedding using utils method
#     tmp_path = "/tmp/tmp_input.jpg"
#     image.save(tmp_path)
#     emb = image_to_embedding(tmp_path)
#     # query qdrant
#     client = qdrant_client(QDRANT_URL)
#     hits = client.search(collection_name=COLLECTION_NAME, query_vector=emb.tolist(), top=top_k)
#     results = []
#     for h in hits:
#         payload = h.payload or {}
#         label = payload.get("label")
#         score = float(h.score) if hasattr(h, "score") else None
#         # split Category > Subcategory
#         cat = None
#         subcat = None
#         if label and ">" in label:
#             cat, subcat = [s.strip() for s in label.split(">", 1)]
#         else:
#             subcat = label
#         results.append({"label": label, "category": cat, "subcategory": subcat, "score": score})
#     return {"predictions": results}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = TOP_K):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    tmp_path = "/tmp/tmp_input.jpg"
    image.save(tmp_path)
    emb = image_to_embedding(tmp_path)

    client = qdrant_client(QDRANT_URL)

    # NEW API — qdrant >= 1.4
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=top_k
    )

    hits = result.points

    results = []
    for h in hits:
        payload = h.payload or {}
        label = payload.get("label")
        score = float(h.score)

        cat = None
        subcat = None
        if label and ">" in label:
            cat, subcat = [s.strip() for s in label.split(">", 1)]
        else:
            subcat = label

        results.append({
            "label": label,
            "category": cat,
            "subcategory": subcat,
            "score": score
        })

    return {"predictions": results}
