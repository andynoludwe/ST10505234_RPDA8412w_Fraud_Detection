"""
Loading + stratified split for the fusion meta-classifier's case-level
manifest (works for both the synthetic smoke-test manifest and the real
combined-predictions manifest, since both share the same column schema).
"""

import csv
import random
from pathlib import Path

from data.synthetic_cases import FEATURE_COLUMNS


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in FEATURE_COLUMNS:
            row[col] = float(row[col])
        row["label"] = int(float(row["label"]))
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

    fieldnames = ["case_id"] + FEATURE_COLUMNS + ["label"]

    def _write(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    _write(train_path, train_rows)
    _write(val_path, val_rows)
    _write(test_path, test_rows)

    print(f"Split {len(rows)} cases -> train={len(train_rows)}, "
          f"val={len(val_rows)}, test={len(test_rows)}")
