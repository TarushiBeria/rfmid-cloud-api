"""
RFMiD retinal classifier API (lightweight TFLite/LiteRT version for Render's free tier).

Why this exists: full TensorFlow needs more RAM than Render's free 512 MB instance
provides, so the process gets killed while the model loads (shows up as a restart with
no traceback, then 502 Bad Gateway). This version runs the same model through the small
LiteRT interpreter instead, so TensorFlow is not needed on the server at all.
"""
import io
import json
import os
import threading

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarushiberia/rfmid-retinal-classifier")
MODEL_FILE = os.environ.get("MODEL_FILE", "rfmid_model.tflite")
LABELS_FILE = os.environ.get("LABELS_FILE", "labels.json")

# MUST match the training notebook.
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", "224"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", "224"))
RESCALE_BY_255 = os.environ.get("RESCALE_BY_255", "true").lower() == "true"
OUTPUT_IS_LOGITS = os.environ.get("OUTPUT_IS_LOGITS", "false").lower() == "true"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

# Comma-separated list of allowed website origins (origin only: no path, no trailing slash).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "https://tarushiberia.github.io").split(",")
    if o.strip()
]
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

state = {
    "interpreter": None,
    "input_index": None,
    "input_dtype": None,
    "output_index": None,
    "labels": None,
    "error": None,
    "stage": "starting",
}
# The interpreter is not thread-safe, and FastAPI runs plain `def` endpoints in a threadpool.
infer_lock = threading.Lock()


def load_artifacts():
    try:
        state["stage"] = "downloading model"
        model_path = hf_hub_download(HF_REPO_ID, MODEL_FILE)
        labels_path = hf_hub_download(HF_REPO_ID, LABELS_FILE)

        state["stage"] = "loading model"
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        in_shape = tuple(int(d) for d in inp["shape"])
        expected = (1, IMG_HEIGHT, IMG_WIDTH, 3)
        if in_shape != expected:
            raise RuntimeError(f"Model expects input {in_shape}, but API sends {expected}")

        with open(labels_path) as f:
            labels = json.load(f)["labels"]
        n_out = int(out["shape"][-1])
        if len(labels) != n_out:
            raise RuntimeError(f"{len(labels)} labels but model outputs {n_out}")

        state.update(
            interpreter=interpreter,
            input_index=inp["index"],
            input_dtype=inp["dtype"],
            output_index=out["index"],
            labels=labels,
            stage="ready",
        )
    except Exception as e:
        state["error"] = str(e)
        state["stage"] = "failed"


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
    # Load in the background so the port opens immediately.
    threading.Thread(target=load_artifacts, daemon=True).start()


# Render's port probe sends HEAD /, so accept it (otherwise it logs 405).
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "RFMiD API running"}


@app.get("/health")
def health():
    return {
        "model_loaded": state["interpreter"] is not None,
        "stage": state["stage"],
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
    if state["interpreter"] is None:
        raise HTTPException(503, "Model is still loading. Retry in a few seconds.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Use a JPEG, PNG or WebP image.")

    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller.")

    x = preprocess(raw).astype(state["input_dtype"])

    with infer_lock:
        interp = state["interpreter"]
        interp.set_tensor(state["input_index"], x)
        interp.invoke()
        out = interp.get_tensor(state["output_index"])[0].copy()

    out = out.astype(np.float32)
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
