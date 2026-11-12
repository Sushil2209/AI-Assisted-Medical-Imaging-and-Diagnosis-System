from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
seed = cfg["seed"]

# One row per bounding box -> one row per patient, Target = max
df = pd.read_csv(cfg["paths"]["rsna_meta"])
patients = df.groupby("patientId", as_index=False)["Target"].max()
patients["filename"] = patients["patientId"] + ".png"
print("Patients:", len(patients))

image_dir = Path(cfg["paths"]["rsna_images"])
missing = sum(not (image_dir / name).exists() for name in patients["filename"])
print("Missing image files:", missing)

# 70/15/15 split, stratified by Target
train, rest = train_test_split(
    patients, test_size=0.30, stratify=patients["Target"], random_state=seed
)
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["Target"], random_state=seed
)

# No patient may appear in two splits
ids = [set(s["patientId"]) for s in (train, val, test)]
assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "patient leak"

out = Path(cfg["paths"]["splits"])
out.mkdir(parents=True, exist_ok=True)
for name, split in [("train", train), ("val", val), ("test", test)]:
    split.to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(split)} patients, positive rate {split['Target'].mean():.3f}")
