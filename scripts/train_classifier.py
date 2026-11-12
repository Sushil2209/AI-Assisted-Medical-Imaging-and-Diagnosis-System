"""Train an EfficientNet-B0 pneumonia classifier on RSNA chest X-rays.

For education only, not for clinical use.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.rsna_dataset import RSNADataset, build_transform  # noqa: E402


def make_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def predict(model, loader, device):
    """Return (labels, probabilities) as numpy arrays."""
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
            labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def evaluate(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", help="override the image folder from the config")
    parser.add_argument("--debug", action="store_true", help="64 train, 32 val/test, 1 epoch")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    c = cfg["classification"]
    splits = Path(cfg["paths"]["splits"])
    images = args.images or cfg["paths"]["rsna_images"]
    epochs = 1 if args.debug else c["epochs"]
    limits = {"train": 64, "val": 32, "test": 32} if args.debug else {}

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    eval_tf = build_transform(c["image_size"])
    train_tf = build_transform(
        c["image_size"], train=True, rotation=c["rotation_degrees"],
        brightness=c["brightness"], contrast=c["contrast"],
    )
    sets = {
        "train": RSNADataset(splits / "train.csv", images, train_tf, limits.get("train")),
        "val": RSNADataset(splits / "val.csv", images, eval_tf, limits.get("val")),
        "test": RSNADataset(splits / "test.csv", images, eval_tf, limits.get("test")),
    }
    loaders = {
        name: DataLoader(ds, batch_size=c["batch_size"], shuffle=(name == "train"),
                         num_workers=c["num_workers"])
        for name, ds in sets.items()
    }

    # pos_weight = negatives / positives in the train split
    targets = sets["train"].df["Target"]
    pos_weight = torch.tensor([(targets == 0).sum() / max((targets == 1).sum(), 1)],
                              dtype=torch.float32).to(device)
    print("pos_weight:", pos_weight.item())

    model = make_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = "efficientnet_b0_debug" if args.debug else "efficientnet_b0"
    ckpt_path = ckpt_dir / f"{name}.pth"

    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        labels, probs = predict(model, loaders["val"], device)
        val_auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        print(f"epoch {epoch}/{epochs} train_loss={loss:.4f} val_auc={val_auc:.4f}")
        # Keep the best epoch by validation AUC (always save the first one)
        if val_auc > best_auc or epoch == 1:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    # Test with the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    labels, probs = predict(model, loaders["test"], device)
    results = evaluate(labels, probs, c["threshold"])
    results.update(best_epoch=best_epoch, val_auc=float(best_auc), n_test=int(len(labels)),
                   debug=args.debug)
    out_path = ckpt_path.with_suffix(".json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("saved:", ckpt_path, out_path)


if __name__ == "__main__":
    main()
