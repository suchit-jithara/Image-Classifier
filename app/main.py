from fastapi import FastAPI, File, UploadFile, HTTPException
from app.utils import image_to_embedding, qdrant_client
import io
from PIL import Image
import os

app = FastAPI(title="SwiftBid CLIP Classifier")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "labels"
TOP_K = 5


# ---------------------------
# CLEANING FUNCTION (IMPORTANT)
# ---------------------------
def clean_subcategory(raw: str) -> str:
    if not raw:
        return raw

    text = raw.lower()

    # remove unwanted phrases
    replacements = [
        "photo of",
        "product",
        "- fashion",
        "fashion",
        "Grocery & Food",
    ]
    for r in replacements:
        text = text.replace(r, "")

    text = text.replace(">", "")
    text = text.replace("-", " ")

    # remove extra spaces
    text = " ".join(text.split())

    return text.title()


@app.get("/")
async def root():
    return {"status": "ok"}


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

        if label and ">" in label:
            category, subraw = [s.strip() for s in label.split(">", 1)]
            subcategory = clean_subcategory(subraw)
        else:
            category = None
            subcategory = clean_subcategory(label)

        results.append({
            "category": category,
            "subcategory": subcategory,
            "score": score
        })

    # select best result
    best = max(results, key=lambda x: x["score"])

    return {
        "category": best["category"],
        "subcategory": best["subcategory"],
        "confidence": best["score"]
    }
