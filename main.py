import io
import json
import os
import threading

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

HF_REPO_ID = os.environ.get("HF_REPO_ID", "YOUR_USERNAME/rfmid-retinal-classifier")
MODEL_FILE = os.environ.get("MODEL_FILE", "rfmid_model.keras")

# MUST match the training notebook.
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", "224"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", "224"))
RESCALE_BY_255 = os.environ.get("RESCALE_BY_255", "true").lower() == "true"
OUTPUT_IS_LOGITS = os.environ.get("OUTPUT_IS_LOGITS", "false").lower() == "true"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

state = {"model": None, "labels": None, "error": None}


def load_artifacts():
    try:
        model_path = hf_hub_download(HF_REPO_ID, MODEL_FILE)
        labels_path = hf_hub_download(HF_REPO_ID, "labels.json")
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(labels_path) as f:
            labels = json.load(f)["labels"]
        n_out = model.output_shape[-1]
        if len(labels) != n_out:
            raise RuntimeError(f"{len(labels)} labels but model outputs {n_out}")
        state["labels"], state["model"] = labels, model
    except Exception as e:
        state["error"] = str(e)


app = FastAPI(title="RFMiD Retinal Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    threading.Thread(target=load_artifacts, daemon=True).start()


@app.get("/")
def root():
    return {"status": "RFMiD API running"}


@app.get("/health")
def health():
    return {
        "model_loaded": state["model"] is not None,
        "num_labels": len(state["labels"]) if state["labels"] else 0,
        "error": state["error"],
    }


def preprocess(raw: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a valid image.")
    img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))  # PIL = (width, height)
    arr = np.asarray(img, dtype=np.float32)
    if RESCALE_BY_255:
        arr /= 255.0
    return arr[None, ...]


@app.post("/predict")
def predict(file: UploadFile = File(...)):  # plain def: runs in threadpool
    if state["error"]:
        raise HTTPException(500, f"Model failed to load: {state['error']}")
    if state["model"] is None:
        raise HTTPException(503, "Model is still loading. Retry in a few seconds.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Use a JPEG, PNG or WebP image.")

    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller.")

    x = preprocess(raw)
    out = state["model"](x, training=False).numpy()[0]
    if OUTPUT_IS_LOGITS:
        out = 1.0 / (1.0 + np.exp(-out))

    preds = [
        {"label": l, "score": float(s), "predicted": bool(s >= THRESHOLD)}
        for l, s in zip(state["labels"], out)
    ]
    preds.sort(key=lambda p: p["score"], reverse=True)
    return {
        "predictions": preds,
        "threshold": THRESHOLD,
        "notice": "Research prototype only; not a medical diagnosis.",
    }
