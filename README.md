# 🚀 **SwiftBid – CLIP + Qdrant Image Classification System**

A production-grade image classification API built using:

* **CLIP (OpenAI Vision Transformer)**
* **FastAPI**
* **Qdrant Vector Database**
* **Docker + Docker Compose**

This system predicts **Category** and **Subcategory** from an input image using **text–image similarity** (CLIP).
No model training required. Highly scalable and customizable.

---

# 📌 **Features**

* Zero-training image classification
* Custom taxonomy support (any domain: Fashion, Furniture, Electronics…)
* All categories stored as text → CLIP text embeddings
* Fast nearest-neighbor lookup using Qdrant
* Clean FastAPI endpoint (`/predict`)
* Easy deployment with Docker Compose
* Supports millions of labels with Qdrant scalability

---

# 🧠 **How It Works (High-Level Overview)**

<p align="center">
  <img src="https://dummyimage.com/900x350/000/fff&text=CLIP+%2B+Qdrant+Flow" />
</p>

### **1. Prepare Taxonomy → Labels**

You define your category/subcategory structure in `taxonomy.json`.

Example:

```
Fashion > Men Sports Shoes
Electronics > Laptop
Furniture > Sofa
```

### **2. Convert Labels to Text Embeddings**

Each textual label is passed through CLIP **text encoder**, generating a 512-dim normalized embedding.

### **3. Store Embeddings in Qdrant**

All text vectors are inserted into a Qdrant vector collection.

### **4. API Receives an Image**

User uploads an image to `/predict`.

### **5. CLIP Converts Image → Embedding**

Image → CLIP Image Encoder → 512-dim embedding.

### **6. Qdrant Finds Nearest Text Label**

Using cosine similarity, the nearest label = best match.

### **7. API Returns Category, Subcategory + Confidence**

Example:

```
{
  "category": "Fashion",
  "subcategory": "Men Sports Shoes",
  "confidence": 0.86
}
```

---

# 📁 **Project Structure**

```
.
├── app/
│   ├── main.py                 # FastAPI app
│   ├── utils.py                # CLIP + Qdrant utilities
│   └── ...
│
├── scripts/
│   ├── prepare_labels.py       # Convert taxonomy.json → labels.csv
│   ├── ingest_embeddings.py    # Generate embeddings & upload to Qdrant
│   └── predict_local.py        # Local testing script
│
├── taxonomy.json               # All categories & subcategories
├── labels.csv                  # Generated label list
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# ⚙️ **Setup Instructions**

## ✅ **1. Install Docker & Docker Compose**

If not installed:

* [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
* [https://docs.docker.com/compose/](https://docs.docker.com/compose/)

---

# 🏗 **2. Prepare Labels (Create `labels.csv`)**

```
python scripts/prepare_labels.py --taxonomy taxonomy.json --out labels.csv --add-variants
```

This generates all labels such as:

```
Fashion > Men Sports Shoes
Fashion > photo of Men Sports Shoes
Fashion > Men Sports Shoes product
```

> Adding variants helps CLIP better understand diverse phrasing.

---

# 🧬 **3. Generate and Upload Embeddings to Qdrant**

⚠️ Make sure Qdrant is running before ingestion:

```
docker-compose up qdrant
```

Then run:

```
python scripts/ingest_embeddings.py --labels labels.csv --collection labels
```

This will:

✔ Generate CLIP text embeddings
✔ Create Qdrant collection
✔ Upload all vectors

---

# 🐳 **4. Run the Full Application with Docker Compose**

```
docker-compose up --build
```

This starts:

1. **Qdrant** → on port `6333`
2. **FastAPI App** → on port `8001` (mapped to internal 8000)

---

# 🔍 **5. Test the API**

### **Curl**

```
curl -X POST http://localhost:8001/predict \
  -F "file=@test.jpg"
```

### **Python**

```
python scripts/predict_local.py test.jpg
```

### Expected Response

```
{
  "category": "Fashion",
  "subcategory": "Men Sports Shoes",
  "confidence": 0.88
}
```

---

# 📌 **API Endpoints**

## **GET /**

Health check
Response:

```
{"status": "ok"}
```

## **POST /predict**

Upload an image → returns category & subcategory.

Form-Data:

```
file: <image>
```

---

# 🏆 **Approach Explanation (CLIP + Vector Search)**

### ✔ Why CLIP?

* Understands both **images and text**
* Puts both into the **same embedding space**
* Great zero-shot performance
* No training required

### ✔ Why Qdrant?

* Optimized for high-dimensional vector search
* Fast cosine similarity search
* Easy to scale horizontally
* Production-ready

### ✔ Why Text Labels Instead of Training?

* Add unlimited categories instantly
* No GPU training cost
* No retraining required
* Much faster iteration

---

# ⚖️ **Pros & Cons of This Approach**

## ✅ **Pros**

### ⭐ 1. Zero Training Needed

You don’t train any model — CLIP already understands visual concepts.

### ⭐ 2. Highly Scalable

Add new categories → regenerate embeddings → done.

### ⭐ 3. Domain Independent

Works for:

* Fashion
* Electronics
* Furniture
* Groceries
* Custom datasets

### ⭐ 4. Extremely Fast Inference

Just:

1. Generate image embedding
2. Query nearest label

### ⭐ 5. Easy Deployment

Just run via Docker Compose.

### ⭐ 6. Very Low Maintenance

No need to maintain ML training pipelines.

---

## ❌ **Cons**

### ⚠️ 1. Depends heavily on label quality

Better label phrasing → better accuracy.

### ⚠️ 2. CLIP sometimes confuses similar items

Example:

* "Cotton Shoes" vs "Sneakers"
* "Men Shirt" vs "Men T-Shirt"

Solution → Add more text variants.

### ⚠️ 3. Cannot detect objects in multi-object scenes

If image has 3 objects → It picks the most dominant one.

Solution → Integrate YOLO for detection + CLIP for classification.

### ⚠️ 4. Accuracy depends on lighting & image quality

Blurry images reduce embedding quality.

---

# 🐞 **Troubleshooting**

### ❗ Qdrant not reachable

Check:

```
docker logs <container>
curl http://localhost:6333
```

### ❗ Prediction inaccurate

Try:

* Add more variants
* Use descriptive subcategory names
* Clean background images
* Add “photo of …” variant

---

# 🎯 Future Improvements (Optional)

✔ Hybrid YOLO + CLIP pipeline
✔ Add multilingual taxonomy
✔ Embed product descriptions
✔ Product similarity search
✔ Train fine-tuned CLIP (optional)

---

# 🙏 **Credits**

* OpenAI CLIP
* Qdrant Vector DB
* FastAPI
* HuggingFace Transformers

---
