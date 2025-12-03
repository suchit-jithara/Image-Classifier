# scripts/ingest_embeddings.py
import csv
from app.utils import texts_to_embedding, qdrant_client
import argparse
from qdrant_client.http.models import Distance, VectorParams

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="labels.csv")
parser.add_argument("--collection", default="labels")
args = parser.parse_args()

labels = []
with open(args.labels, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            labels.append(row[0])

print("Computing embeddings for", len(labels), "labels...")
embeddings = texts_to_embedding(labels, batch_size=32)
print("Done embeddings:", embeddings.shape)

client = qdrant_client()
# create collection (if not exists)
if args.collection not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )

# insert points
points = []
for idx, (label, emb) in enumerate(zip(labels, embeddings)):
    pts = {
        "id": idx,
        "vector": emb.tolist(),
        "payload": {"label": label}
    }
    points.append(pts)

print("Uploading to Qdrant (batching)...")
# batch upload
BATCH = 256
for i in range(0, len(points), BATCH):
    client.upsert(collection_name=args.collection, points=points[i:i+BATCH])
print("Upload complete.")
