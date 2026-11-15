"""FastAPI inference service: pneumonia classification of DICOM instances stored in Orthanc.

For education only, not for clinical use.
"""
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pydicom
import requests
import torch
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.rsna_dataset import build_transform  # noqa: E402
from src.explain.gradcam import load_model  # noqa: E402

DISCLAIMER = "For education only, not for clinical use"
CLASSES = ["No pneumonia", "Pneumonia"]

with open(ROOT / "configs" / "config.yaml") as f:
    cfg = yaml.safe_load(f)
CLS = cfg["classification"]
ORTHANC_URL = cfg["orthanc"]["url"].rstrip("/")
ORTHANC_TIMEOUT = cfg["orthanc"]["timeout"]
CKPT_PATH = ROOT / cfg["paths"]["checkpoints"] / "efficientnet_b0.pth"

state = {}


@asynccontextmanager
async def lifespan(app):
    try:
        state["model"] = load_model(CKPT_PATH, device="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint at {CKPT_PATH}: {e}") from e
    yield


app = FastAPI(title="Medical Imaging Inference API", description=DISCLAIMER, lifespan=lifespan)


def orthanc_get(path):
    """GET from Orthanc and map failures to HTTP errors."""
    try:
        resp = requests.get(f"{ORTHANC_URL}{path}", timeout=ORTHANC_TIMEOUT)
    except requests.RequestException:
        raise HTTPException(502, "Orthanc is unreachable")
    if resp.status_code == 404:
        raise HTTPException(404, "Not found in Orthanc")
    if not resp.ok:
        raise HTTPException(502, f"Orthanc returned status {resp.status_code}")
    return resp


def dicom_to_image(data):
    """Decode DICOM bytes to an 8-bit RGB PIL image."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
    except Exception:
        raise HTTPException(422, "Could not decode DICOM pixel data")
    photometric = ds.get("PhotometricInterpretation")
    if photometric not in ("MONOCHROME1", "MONOCHROME2") or ds.get("SamplesPerPixel", 1) != 1:
        raise HTTPException(422, f"Unsupported DICOM image: {photometric} (only single-channel grayscale is supported)")
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        raise HTTPException(422, "Unsupported DICOM image: multi-frame images are not supported")
    try:
        pixels = ds.pixel_array.astype(np.float32)
    except Exception:
        raise HTTPException(422, "Could not decode DICOM pixel data")
    if pixels.ndim != 2:
        raise HTTPException(422, "Unsupported DICOM image: expected a single 2D grayscale frame")
    # Convert stored values to modality values when rescale tags are present
    pixels = pixels * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    if photometric == "MONOCHROME1":
        pixels = pixels.max() - pixels  # MONOCHROME1: low values are bright, so invert
    pixels = (pixels - pixels.min()) / max(float(pixels.max() - pixels.min()), 1e-6)
    return Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "device": "cpu", "disclaimer": DISCLAIMER}


@app.get("/orthanc/studies")
def list_studies():
    studies = orthanc_get("/studies?expand").json()
    return {
        "studies": [
            {
                "id": s["ID"],
                "patient_id": s.get("PatientMainDicomTags", {}).get("PatientID"),
                "study_date": s.get("MainDicomTags", {}).get("StudyDate"),
                "description": s.get("MainDicomTags", {}).get("StudyDescription"),
            }
            for s in studies
        ],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict/{instance_id}")
def predict(instance_id: str):
    image = dicom_to_image(orthanc_get(f"/instances/{instance_id}/file").content)
    x = build_transform(CLS["image_size"])(image).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(state["model"](x)).item()
    label = int(prob >= CLS["threshold"])
    return {
        "instance_id": instance_id,
        "predicted_class": CLASSES[label],
        "pneumonia_probability": round(prob, 4),
        "threshold": CLS["threshold"],
        "disclaimer": DISCLAIMER,
    }
