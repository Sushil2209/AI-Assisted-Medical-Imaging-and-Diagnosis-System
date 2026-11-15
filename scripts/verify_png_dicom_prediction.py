"""Temporary check: PNG vs the same image as DICOM from Orthanc.

For education only, not for clinical use.
"""
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from services.inference_api.main import dicom_to_image  # noqa: E402
from src.data.rsna_dataset import build_transform  # noqa: E402
from src.explain.gradcam import load_model  # noqa: E402

PNG = ROOT / "data/raw/rsna/Training/Images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.png"
INSTANCE_ID = "298aab3d-dc2f2aea-ed9b9723-d326acc1-efc74da0"

cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
size = cfg["classification"]["image_size"]
model = load_model(ROOT / cfg["paths"]["checkpoints"] / "efficientnet_b0.pth", device="cpu")
transform = build_transform(size)

png_img = Image.open(PNG).convert("RGB")
resp = requests.get(f"{cfg['orthanc']['url'].rstrip('/')}/instances/{INSTANCE_ID}/file",
                    timeout=cfg["orthanc"]["timeout"])
resp.raise_for_status()
dcm_img = dicom_to_image(resp.content)

png_arr, dcm_arr = np.array(png_img), np.array(dcm_img)
print("PNG shape:", png_arr.shape, "DICOM shape:", dcm_arr.shape)
equal = png_arr.shape == dcm_arr.shape and np.array_equal(png_arr, dcm_arr)
print("RGB arrays exactly equal:", equal)
if png_arr.shape == dcm_arr.shape:
    print("Max abs pixel diff:", int(np.abs(png_arr.astype(int) - dcm_arr.astype(int)).max()))


def predict(image):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


p_png, p_dcm = predict(png_img), predict(dcm_img)
print(f"PNG probability:    {p_png:.8f}")
print(f"DICOM probability:  {p_dcm:.8f}")
print(f"Absolute difference: {abs(p_png - p_dcm):.3e}")
