"""
Catalog Ingestion Script for Football Products
Reads football_catalog.json (pointing to Catalog_images/ball1.jpg, etc.),
extracts CLIP vector embeddings, and stores points in local Qdrant.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CATALOG_FILE = "football_catalog.json"
QDRANT_STORAGE = "./qdrant_storage"
COLLECTION_NAME = "football_catalog"
MODEL_NAME = "openai/clip-vit-base-patch32"

def ensure_placeholder_image(path_str: str, title: str, rgb_tuple: list):
    path = Path(path_str)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 400), color=tuple(rgb_tuple))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 380, 380], outline=(255, 255, 255, 120), width=3)
    draw.text((40, 180), title, fill=(255, 255, 255))
    draw.text((40, 210), "[Catalog Photo: " + path.name + "]", fill=(200, 200, 200))
    img.save(path, quality=92)
    print(f"  Created placeholder at: {path_str}")

def extract_features(model_outputs):
    """Safely extracts the tensor across all versions of Hugging Face transformers."""
    if isinstance(model_outputs, torch.Tensor):
        return model_outputs
    if hasattr(model_outputs, "pooler_output") and model_outputs.pooler_output is not None:
        return model_outputs.pooler_output
    if hasattr(model_outputs, "image_embeds") and model_outputs.image_embeds is not None:
        return model_outputs.image_embeds
    return model_outputs[0]

def main():
    print(f"--- Starting Football Catalog Ingestion ---")
    if not os.path.exists(CATALOG_FILE):
        raise FileNotFoundError(f"Missing {CATALOG_FILE}")

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"Loaded {len(products)} products from {CATALOG_FILE}.")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading vision model '{MODEL_NAME}' on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    qdrant = QdrantClient(path=QDRANT_STORAGE)
    vector_dim = model.config.projection_dim if hasattr(model.config, "projection_dim") else 512

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'.")

    points = []
    for idx, item in enumerate(products, start=1):
        image_path = item["image_path"]
        print(f"\nProcessing [{idx}/{len(products)}]: {item['title']} ({image_path})")

        color_rgb = item.get("color_rgb", [120, 140, 160])
        ensure_placeholder_image(image_path, item["title"], color_rgb)

        img = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            raw_features = model.get_image_features(**inputs)
            features = extract_features(raw_features)
            normalized = F.normalize(features, p=2, dim=-1)
            vector = normalized.squeeze().cpu().tolist()

        payload = {
            "sku": item["sku"],
            "title": item["title"],
            "brand": item["brand"],
            "category": item["category"],
            "color": item.get("color", ""),
            "image_url": f"/{image_path}",
            "specifications": item.get("specifications", {}),
        }

        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    total_count = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"\nIngestion Complete! Total items indexed in Qdrant: {total_count}")
    qdrant.close()

if __name__ == "__main__":
    main()