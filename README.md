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
* Text labels → CLIP text embeddings
* Fast nearest-neighbor lookup via Qdrant
* Clean FastAPI endpoint (`/predict`)
* Easy deployment with Docker
* Supports millions of labels

---

# 🧠 **How It Works (High-Level Overview)**

<p align="center">
  <img src="https://dummyimage.com/900x350/000/fff&text=CLIP+%2B+Qdrant+Flow" />
</p>

### **1. Prepare Taxonomy → Labels**

You define category/subcategory structure in `taxonomy.json`:

```
Fashion > Men Sports Shoes
Electronics > Laptop
Furniture > Sofa
```

### **2. Convert Labels to Text Embeddings**

Each label is encoded using CLIP’s **text encoder** → a 512-dim vector.

### **3. Store Embeddings in Qdrant**

All vectors are uploaded into Qdrant for fast search.

### **4. API Receives an Image**

User uploads an image to `/predict`.

### **5. CLIP Converts Image → Embedding**

CLIP image encoder extracts a vector representation.

### **6. Qdrant Finds Nearest Text Label**

Cosine similarity returns the closest match.

### **7. API Returns Category + Subcategory**

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
│   ├── main.py              # FastAPI app
│   ├── utils.py             # CLIP + Qdrant utilities
│
├── scripts/
│   ├── prepare_labels.py    # Convert taxonomy.json → labels.csv
│   ├── ingest_embeddings.py # Generate embeddings & upload to Qdrant
│   ├── predict_local.py     # Local testing script
│
├── taxonomy.json
├── labels.csv               # Generated after step 1
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ **Setup Instructions**

## ✅ **1. Install Docker & Docker Compose**

If not installed:

* [https://docs.docker.com/get-docker](https://docs.docker.com/get-docker)
* [https://docs.docker.com/compose](https://docs.docker.com/compose)

---

# 🏗 **2. Prepare Labels (Generate `labels.csv`)**

Run:

```
python scripts/prepare_labels.py --add-variants
```

Verify:

```
ls -l labels.csv
```

This creates entries like:

```
Fashion > Men Sports Shoes
Fashion > photo of Men Sports Shoes
Fashion > Men Sports Shoes product
```

---

# 🧬 **3. Ingest Embeddings into Qdrant**

⚠️ **IMPORTANT**
Do **NOT** run ingestion inside the **Qdrant** container — Qdrant does **not** have Python.

### ✔ Step 1 — Start Qdrant & App containers

```
docker-compose up --build -d
```

### ✔ Step 2 — Enter the app container

Find container name:

```
docker ps
```

Enter:

```
docker exec -it swiftbid-clip-classifier-app-1 bash
```

### ✔ Step 3 — (Optional but recommended)

```
export PYTHONPATH=/app
echo $PYTHONPATH
# Output: /app
```

### ✔ Step 4 — Run embedding ingestion

```
python scripts/ingest_embeddings.py --labels labels.csv --collection labels
```

This will:

* Load all labels
* Generate CLIP embeddings
* Create/update Qdrant collection
* Upload all vectors

---

# 🐳 **4. Run the Full Application**

```
docker-compose up --build
```

Services:

| Service     | Port | Description    |
| ----------- | ---- | -------------- |
| Qdrant      | 6333 | Vector DB      |
| FastAPI app | 8001 | Prediction API |

---

# 🔍 **5. Test the API**

### ✔ Using curl

```
curl -X POST http://localhost:8001/predict \
  -F "file=@test.jpg"
```

### ✔ Using Python

```
python scripts/predict_local.py test.jpg
```

### Expected Output

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

Health check:

```
{"status": "ok"}
```

## **POST /predict**

Form-Data:

```
file: <image>
```

Response:

```
{
  "category": "...",
  "subcategory": "...",
  "confidence": 0.87
}
```

---

# 🏆 **Approach Explanation (CLIP + Vector Search)**

### ✔ Why CLIP?

* Understands both images & text
* No training required
* Zero-shot prediction
* Robust across many domains

### ✔ Why Qdrant?

* Optimized vector search
* Fast cosine similarity
* Works at scale
* Simple & production-ready

### ✔ Why text-based classification?

* Add unlimited categories
* No training cost
* No need to collect dataset
* Iteration is instant

---

# ⚖️ **Pros & Cons**

## ✅ Pros

* Zero training
* Easy to scale
* Robust across domains
* Fast inference
* Easy deployment
* Very low maintenance

## ❌ Cons

* Requires well-defined labels
* CLIP may confuse similar items
* Not ideal for multi-object images
* Dependent on image quality

---

# 🐞 **Troubleshooting**

### ❗ Python not found

You entered Qdrant container.
Correct:

```
docker exec -it swiftbid-clip-classifier-app-1 bash
```

### ❗ Import errors

Run:

```
export PYTHONPATH=/app
```

### ❗ Low accuracy

Try:

* Add variants
* Improve subcategory naming
* Add descriptive labels
* Use better images

---

# 🎯 Future Improvements

* YOLO + CLIP hybrid pipeline
* Multilingual labels
* Product attribute extraction
* Brand-level classification
* CLIP-Large for higher accuracy

---

# 🙏 Credits

* OpenAI CLIP
* HuggingFace Transformers
* Qdrant Vector DB
* FastAPI
* Uvicorn
