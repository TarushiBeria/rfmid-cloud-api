
import os

# Set these before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import logging
import threading

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Configuration
# --------------------------------------------------

HF_REPO_ID = os.environ.get(
    "HF_REPO_ID",
    "tarushiberia/rfmid-retinal-classifier",
)

MODEL_FILE = os.environ.get("MODEL_FILE", "rfmid_model.keras")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", "224"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", "224"))

RESCALE_BY_255 = (
    os.environ.get("RESCALE_BY_255", "true").lower() == "true"
)

OUTPUT_IS_LOGITS = (
    os.environ.get("OUTPUT_IS_LOGITS", "false").lower() == "true"
)

THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

MAX_BYTES = 10 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limit TensorFlow CPU parallelism.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# --------------------------------------------------
# Application and state
# --------------------------------------------------

app = FastAPI(
    title="RFMiD Retinal Classification API",
    description="Multi-label retinal fundus image classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

state = {
    "model": None,
    "labels": None,
    "error": None,
    "loading": False,
}

load_lock = threading.Lock()
prediction_lock = threading.Lock()


# --------------------------------------------------
# Load model and labels once
# --------------------------------------------------

def load_artifacts():
    with load_lock:
        if state["model"] is not None:
            return

        state["loading"] = True

        try:
            logger.info("Downloading RFMiD model from %s", HF_REPO_ID)

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILE,
                repo_type="model",
            )

            labels_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="labels.json",
                repo_type="model",
            )

            logger.info("Loading model file: %s", MODEL_FILE)

            loaded_model = tf.keras.models.load_model(
                model_path,
                compile=False,
            )

            with open(labels_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            labels = label_data["labels"]

            output_shape = loaded_model.output_shape

            if isinstance(output_shape, list):
                raise RuntimeError(
                    "This API expects a single-output Keras model."
                )

            output_count = output_shape[-1]

            if output_count is None:
                raise RuntimeError(
                    "Cannot determine the model output count."
                )

            if len(labels) != int(output_count):
                raise RuntimeError(
                    f"labels.json contains {len(labels)} labels, "
                    f"but the model outputs {output_count} values."
                )

            state["model"] = loaded_model
            state["labels"] = labels
            state["error"] = None

            logger.info(
                "Model loaded successfully with %d output labels.",
                len(labels),
            )

        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("RFMiD model loading failed")

        finally:
            state["loading"] = False


# --------------------------------------------------
# Startup
# --------------------------------------------------

@app.on_event("startup")
def startup():
    # Start loading in the background so the HTTP server can start.
    threading.Thread(
        target=load_artifacts,
        daemon=True,
        name="rfmid-model-loader",
    ).start()


# --------------------------------------------------
# Health and home endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "RFMiD API running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": (
            "ready"
            if state["model"] is not None
            else "loading" if state["loading"]
            else "error" if state["error"]
            else "starting"
        ),
        "model_loaded": state["model"] is not None,
        "num_labels": len(state["labels"]) if state["labels"] else 0,
        "error": state["error"],
    }


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image = image.convert("RGB")
            image = image.resize(
                (IMG_WIDTH, IMG_HEIGHT),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(image, dtype=np.float32)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    if RESCALE_BY_255:
        array = array / 255.0

    # Shape: (1, height, width, 3)
    return np.expand_dims(array, axis=0)


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if state["error"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {state['error']}",
        )

    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry shortly.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        raw = file.file.read(MAX_BYTES + 1)

        if len(raw) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller.",
            )

        if not raw:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        x = preprocess(raw)

        # Serialize inference to reduce concurrent memory pressure.
        with prediction_lock:
            output = state["model"](x, training=False)
            scores = np.asarray(output)[0].astype(np.float32)

        if OUTPUT_IS_LOGITS:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80, 80)))

        if len(scores) != len(state["labels"]):
            raise HTTPException(
                status_code=500,
                detail="Prediction output count does not match labels.",
            )

        if not np.all(np.isfinite(scores)):
            raise HTTPException(
                status_code=500,
                detail="Model produced invalid prediction values.",
            )

        if np.any(scores < 0) or np.any(scores > 1):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Outputs are outside [0, 1]. Check "
                    "OUTPUT_IS_LOGITS and the model's final activation."
                ),
            )

        predictions = [
            {
                "label": label,
                "score": round(float(score), 6),
                "predicted": bool(score >= THRESHOLD),
            }
            for label, score in zip(state["labels"], scores)
        ]

        predictions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return {
            "filename": file.filename,
            "threshold": THRESHOLD,
            "predictions": predictions,
            "notice": (
                "Research prototype only; not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

    finally:
        file.file.close()
