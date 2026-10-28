"""
Tabular loading + stratified split for the Random Forest document branch.

Mirrors the CNN branch's data/dataset.py (stratified 60/20/20 split by
default), but works on the feature-row CSV manifest instead of an image
manifest.
"""

import csv
import random
from pathlib import Path

from .synthetic_metadata import ANOMALY_TYPES, FEATURE_NAMES


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for key in FEATURE_NAMES:
            row[key] = float(row[key])
        for key in ANOMALY_TYPES + ["label"]:
            row[key] = int(float(row[key]))
    return rows


def stratified_split_manifest(
    manifest_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> None:
    """Stratified split by overall `label`, matching the CNN branch's 60/20/20 default."""
    rows = load_manifest(manifest_path)
    rng = random.Random(seed)

    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train:n_train + n_val])
        test_rows.extend(group[n_train + n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    fieldnames = ["doc_id"] + FEATURE_NAMES + ANOMALY_TYPES + ["label"]

    def _write(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    _write(train_path, train_rows)
    _write(val_path, val_rows)
    _write(test_path, test_rows)

    print(f"Split {len(rows)} rows -> train={len(train_rows)}, "
          f"val={len(val_rows)}, test={len(test_rows)}")
