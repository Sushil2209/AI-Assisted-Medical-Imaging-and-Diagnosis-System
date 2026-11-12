from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size, train=False, rotation=10, brightness=0.1, contrast=0.1):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        # No flips: left/right matters in chest X-rays
        steps += [
            transforms.RandomRotation(rotation),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class RSNADataset(Dataset):
    """Reads a split CSV (filename, Target) and loads PNGs from image_dir."""

    def __init__(self, csv_path, image_dir, transform, limit=None):
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.head(limit)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / row["filename"]).convert("RGB")
        label = torch.tensor(float(row["Target"]))
        return self.transform(image), label
