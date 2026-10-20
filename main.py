import io
import json
import os

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

# Replace with your actual Hugging Face repository ID.
HF_REPO_ID = os.environ.get(
    "HF_REPO_ID",
    "tarushiberia/rfmid-retinal-classifier"
)

# MUST match your training notebook.
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", "224"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH", "224"))

# Only use 1/255 if your training pipeline used this scaling.
RESCALE_BY_255 = os.environ.get(
    "RESCALE_BY_255", "true"
).lower() == "true"

# Multi-label models commonly use sigmoid outputs.
# Confirm this from your notebook.
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

app = FastAPI(title="RFMiD Retinal Classification API")

# For initial testing only. Restrict this to your GitHub Pages
# origin after the site is deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)