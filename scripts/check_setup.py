from pathlib import Path

import monai
import torch
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

print("torch:", torch.__version__)
print("monai:", monai.__version__)
print("CUDA available:", torch.cuda.is_available())

root = Path(cfg["paths"]["msd_root"])
for folder in ["imagesTr", "labelsTr"]:
    # *.nii* matches .nii and .nii.gz; skip macOS "._" files
    files = sorted(p for p in (root / folder).glob("*.nii*") if not p.name.startswith("._"))
    print(f"{folder}: {len(files)} files")
    for p in files[:3]:
        print(f"  {p.name} ({p.stat().st_size / 1024 ** 2:.1f} MB)")
