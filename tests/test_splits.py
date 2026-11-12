from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_ids():
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    split_dir = ROOT / cfg["paths"]["splits"] if "paths" in cfg else ROOT / cfg["splits"]
    return {n: set(pd.read_csv(split_dir / f"{n}.csv")["patientId"]) for n in ("train", "val", "test")}


def test_no_patient_in_two_splits():
    ids = load_ids()
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        assert not ids[a] & ids[b], f"patientId overlap between {a} and {b}"
