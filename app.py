import os
import io
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Football Catalog Lens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount Local Images Folder (Catalog_images)
IMAGES_DIR = "Catalog_images"
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/Catalog_images", StaticFiles(directory=IMAGES_DIR), name="Catalog_images")

# 2. Vision Engine & Vector DB Setup
MODEL_NAME = "openai/clip-vit-base-patch32"
COLLECTION_NAME = "football_catalog"

# Minimum cosine similarity threshold (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.60

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading vision model '{MODEL_NAME}' on {device}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

DB_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_storage")
qdrant = QdrantClient(path=DB_PATH)

if not qdrant.collection_exists(COLLECTION_NAME):
    vector_dim = model.config.projection_dim if hasattr(model.config, "projection_dim") else 512
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

def extract_features(model_outputs):
    if isinstance(model_outputs, torch.Tensor):
        return model_outputs
    if hasattr(model_outputs, "pooler_output") and model_outputs.pooler_output is not None:
        return model_outputs.pooler_output
    if hasattr(model_outputs, "image_embeds") and model_outputs.image_embeds is not None:
        return model_outputs.image_embeds
    return model_outputs[0]

@torch.no_grad()
def get_image_embedding(image: Image.Image) -> List[float]:
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    raw_features = model.get_image_features(**inputs)
    features = extract_features(raw_features)
    normalized = F.normalize(features, p=2, dim=-1)
    return normalized.squeeze().cpu().tolist()

def crop_image_bounding_box(image: Image.Image, box: Optional[str] = None) -> Image.Image:
    if not box:
        return image
    try:
        coords = [float(c.strip()) for c in box.split(",")]
        if len(coords) != 4:
            return image
        x_min, y_min, x_max, y_max = coords
        w, h = image.size

        if max(coords) <= 1.0:
            left = max(0, int(x_min * w))
            top = max(0, int(y_min * h))
            right = min(w, int(x_max * w))
            bottom = min(h, int(y_max * h))
        else:
            left = max(0, int(x_min))
            top = max(0, int(y_min))
            right = min(w, int(x_max))
            bottom = min(h, int(y_max))

        if right > left and bottom > top:
            return image.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Warning: could not parse bounding box '{box}': {e}")
    return image

@app.post("/api/search")
async def visual_search(
    file: UploadFile = File(...),
    bounding_box: Optional[str] = Form(None),
    category_filter: Optional[str] = Form(None),
    top_k: int = Form(6),
    threshold: float = Form(CONFIDENCE_THRESHOLD),
):
    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    cropped = crop_image_bounding_box(pil_image, bounding_box)
    query_vector = get_image_embedding(cropped)

    query_filter = None
    if category_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category_filter)
                )
            ]
        )

    # Compatible with both modern qdrant-client (>=1.10 query_points) and older (search)
    if hasattr(qdrant, "query_points"):
        search_res = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=threshold,
        )
        results = search_res.points
    else:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=threshold,
        )

    matches = []
    top_score = 0.0
    for hit in results:
        top_score = max(top_score, hit.score)
        matches.append({
            "score": round(float(hit.score), 4),
            "percentage": f"{round(float(hit.score) * 100, 1)}%",
            "sku": hit.payload.get("sku"),
            "title": hit.payload.get("title"),
            "brand": hit.payload.get("brand"),
            "category": hit.payload.get("category"),
            "color": hit.payload.get("color"),
            "image_url": hit.payload.get("image_url", ""),
            "specifications": hit.payload.get("specifications", {}),
        })

    if not matches:
        return {
            "status": "not_found",
            "message": "Object not found in catalog. No visual matches met the confidence threshold.",
            "matches": [],
            "top_confidence": 0.0,
            "threshold_used": threshold,
        }

    return {
        "status": "success",
        "matches": matches,
        "count": len(matches),
        "top_confidence": round(top_score, 4),
        "threshold_used": threshold,
        "bounding_box_applied": bool(bounding_box),
    }

@app.get("/api/catalog")
async def list_catalog():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=50,
        with_payload=True,
        with_vectors=False
    )
    return {"total": len(points), "items": [p.payload for p in points]}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Lens API is running. Place index.html alongside app.py to view UI.</h1>"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)