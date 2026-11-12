"""Grad-CAM heatmaps for the EfficientNet-B0 pneumonia classifier.

For education only, not for clinical use.
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models

from src.data.rsna_dataset import build_transform


def load_model(ckpt_path, device="cpu"):
    # Same architecture as make_model() in scripts/train_classifier.py (weights come from the checkpoint)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def explain(model, image, image_size=224, device="cpu"):
    """Return (pneumonia probability, heatmap overlay as uint8 RGB array) for a PIL image."""
    image = image.convert("RGB")
    x = build_transform(image_size)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    # Single output head: target index 0 is the pneumonia logit
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]

    rgb = np.asarray(image.resize((image_size, image_size)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)
    return prob, overlay
