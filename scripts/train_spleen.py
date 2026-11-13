"""Train a MONAI 3D U-Net on MSD Task09_Spleen.

For education only, not for clinical use.
"""
import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete, CropForegroundd, EnsureChannelFirstd, LoadImaged, Orientationd,
    RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd, Compose,
)
from monai.utils import set_determinism

ROOT = Path(__file__).resolve().parent.parent


def find_files(folder):
    """Sorted *.nii* files, skipping macOS '._' files."""
    return sorted(p for p in Path(folder).glob("*.nii*") if not p.name.startswith("._"))


def make_split(data_dir, cfg, debug):
    """Split cases into train/val/test with a fixed seed and save to json."""
    images = find_files(Path(data_dir) / "imagesTr")
    labels = find_files(Path(data_dir) / "labelsTr")
    cases = [{"image": str(i), "label": str(l)} for i, l in zip(images, labels)]
    random.Random(cfg["seed"]).shuffle(cases)
    seg = cfg["segmentation"]
    n_tr, n_va, n_te = seg["n_train"], seg["n_val"], seg["n_test"]
    split = {
        "train": cases[:n_tr],
        "val": cases[n_tr:n_tr + n_va],
        "test": cases[n_tr + n_va:n_tr + n_va + n_te],
    }
    out = ROOT / cfg["paths"]["splits"] / "msd_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2))
    if debug:  # tiny subset for a quick smoke test
        split = {k: v[:2] for k, v in split.items()}
    return split


def make_transforms(seg, train):
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=seg["pixdim"], mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    if train:
        steps.append(RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=seg["patch_size"],
            pos=1, neg=1, num_samples=seg["num_samples"], image_key="image", image_threshold=0))
    return Compose(steps)


def evaluate(model, loader, seg, device):
    """Mean Dice over a loader using sliding window inference."""
    to_onehot = AsDiscrete(argmax=True, to_onehot=2)
    label_onehot = AsDiscrete(to_onehot=2)
    metric = DiceMetric(include_background=False, reduction="mean")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = sliding_window_inference(x, seg["patch_size"], 4, model)
            metric([to_onehot(p) for p in decollate_batch(pred)],
                   [label_onehot(l) for l in decollate_batch(y)])
    score = metric.aggregate().item()
    metric.reset()
    return score


def main():
    parser = argparse.ArgumentParser()
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    parser.add_argument("--data_dir", default=str(ROOT / cfg["paths"]["msd_root"]))
    parser.add_argument("--debug", action="store_true", help="tiny run for testing")
    args = parser.parse_args()

    seg = cfg["segmentation"]
    set_determinism(seed=cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    split = make_split(args.data_dir, cfg, args.debug)
    epochs = 1 if args.debug else seg["epochs"]
    val_interval = 1 if args.debug else seg["val_interval"]

    train_loader = DataLoader(Dataset(split["train"], make_transforms(seg, True)),
                              batch_size=seg["batch_size"], shuffle=True,
                              num_workers=seg["num_workers"])
    eval_tf = make_transforms(seg, False)
    val_loader = DataLoader(Dataset(split["val"], eval_tf), batch_size=1)
    test_loader = DataLoader(Dataset(split["test"], eval_tf), batch_size=1)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
                 channels=seg["unet_channels"], strides=seg["unet_strides"],
                 num_res_units=seg["num_res_units"]).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=seg["lr"],
                                  weight_decay=seg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "spleen_unet.pth"
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"epoch {epoch}/{epochs} loss {total / len(train_loader):.4f}")

        if epoch % val_interval == 0:
            dice = evaluate(model, val_loader, seg, device)
            print(f"  val Dice {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), ckpt_path)

    # Final test with the best model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_dice = evaluate(model, test_loader, seg, device)
    print(f"test mean Dice {test_dice:.4f}")
    metrics = {"test_mean_dice": test_dice, "best_val_dice": best_dice,
               "n_test": len(split["test"])}
    (ckpt_dir / "spleen_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
