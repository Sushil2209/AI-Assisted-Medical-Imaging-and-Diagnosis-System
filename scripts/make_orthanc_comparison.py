"""Save original vs Grad-CAM for the RSNA image that was uploaded to Orthanc.

The source PNG is found by matching the pixels of the DICOM in data/processed.
For education only, not for clinical use.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.explain.gradcam import explain, load_model  # noqa: E402


def find_source_png(dcm_path, cfg):
    pixels = pydicom.dcmread(dcm_path).pixel_array
    for path in sorted(Path(cfg["paths"]["rsna_images"]).glob("*.png")):
        gray = np.array(Image.open(path).convert("L"))
        if gray.shape == pixels.shape and (gray == pixels).all():
            return path
    raise SystemExit("No RSNA image matches the DICOM")


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(cfg["paths"]["checkpoints"]) / "efficientnet_b0.pth", device)

    dcm_path = sorted(Path("data/processed").glob("*.dcm"))[-1]
    png_path = find_source_png(dcm_path, cfg)
    print("DICOM:", dcm_path)
    print("image:", png_path)

    image = Image.open(png_path)
    prob, overlay = explain(model, image, cfg["classification"]["image_size"], device)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    axes[0].imshow(image.convert("L"), cmap="gray")
    axes[0].set_title("Original chest X-ray")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM, P(pneumonia) = {prob:.2f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("For education only, not for clinical use")
    fig.tight_layout()

    out = Path("docs/gradcam_orthanc_comparison.png")
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
