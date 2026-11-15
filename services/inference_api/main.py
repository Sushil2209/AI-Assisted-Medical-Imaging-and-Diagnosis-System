"""FastAPI inference service: pneumonia classification of DICOM instances stored in Orthanc.

For education only, not for clinical use.
"""
import base64
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
from src.explain.gradcam import explain, load_model  # noqa: E402

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
        raw = ds.pixel_array
    except Exception:
        raise HTTPException(422, "Could not decode DICOM pixel data")
    if raw.ndim != 2:
        raise HTTPException(422, "Unsupported DICOM image: expected a single 2D grayscale frame")
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    if ds.get("BitsAllocated") == 8 and raw.dtype == np.uint8 and slope == 1 and intercept == 0:
        # 8-bit without rescale: keep values exactly (no normalization)
        pixels = 255 - raw if photometric == "MONOCHROME1" else raw
    else:
        pixels = raw.astype(np.float64) * slope + intercept
        if photometric == "MONOCHROME1":
            pixels = pixels.max() - pixels  # MONOCHROME1: low values are bright, so invert
        span = pixels.max() - pixels.min()
        pixels = (pixels - pixels.min()) / span * 255 if span > 0 else np.zeros_like(pixels)
        pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


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
def predict(instance_id: str, gradcam: bool = False):
    image = dicom_to_image(orthanc_get(f"/instances/{instance_id}/file").content)
    if gradcam:
        try:
            prob, overlay = explain(state["model"], image, CLS["image_size"], "cpu")
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            raise HTTPException(500, "Grad-CAM generation failed")
    else:
        x = build_transform(CLS["image_size"])(image).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(state["model"](x)).item()
    label = int(prob >= CLS["threshold"])
    response = {
        "instance_id": instance_id,
        "predicted_class": CLASSES[label],
        "pneumonia_probability": round(prob, 4),
        "threshold": CLS["threshold"],
        "disclaimer": DISCLAIMER,
    }
    if gradcam:
        response["gradcam_png_base64"] = gradcam_b64
    return response
