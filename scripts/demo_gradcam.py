"""Save a grid of Grad-CAM examples (3 pneumonia, 3 normal) from the test split.

For education only, not for clinical use.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.explain.gradcam import explain, load_model  # noqa: E402


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    size = cfg["classification"]["image_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(cfg["paths"]["checkpoints"]) / "efficientnet_b0.pth", device)

    df = pd.read_csv(Path(cfg["paths"]["splits"]) / "test.csv")
    rows = pd.concat([
        df[df["Target"] == 1].head(3),
        df[df["Target"] == 0].head(3),
    ])

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        image = Image.open(Path(cfg["paths"]["rsna_images"]) / row["filename"])
        prob, overlay = explain(model, image, size, device)
        truth = "pneumonia" if row["Target"] == 1 else "normal"
        ax.imshow(overlay)
        ax.set_title(f"true: {truth}\nP(pneumonia) = {prob:.2f}")
        ax.axis("off")
    fig.suptitle("Grad-CAM (EfficientNet-B0) - For education only, not for clinical use")
    fig.tight_layout()

    out = Path("docs/gradcam_examples.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()
