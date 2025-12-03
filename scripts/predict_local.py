# scripts/predict_local.py
import requests
import sys

def predict(image_path):
    url = "http://localhost:8000/predict"
    files = {"file": open(image_path, "rb")}
    resp = requests.post(url, files=files)
    print(resp.json())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_local.py image.jpg")
    else:
        predict(sys.argv[1])
