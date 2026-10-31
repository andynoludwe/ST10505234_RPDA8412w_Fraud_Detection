"""
PyTorch Dataset for DocumentTamperCNN.

Reads the manifest CSV produced by synthetic_tamper.generate_labeled_dataset
(or an equivalent CSV with the same columns) and returns image tensors plus
two label tensors:
    - forgery_label: 0 (genuine) / 1 (forged)
    - tamper_type_label: multi-hot vector over TAMPER_TYPES
      (all zeros for genuine samples)

Standard augmentation (rotation, scaling, colour jitter, noise) is applied
to the training split only, per Section 7.4 of the proposal.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .synthetic_tamper import TAMPER_TYPES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=3),  # documents are near-upright; keep this small
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DocumentTamperDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 224, augment: bool = False):
        self.rows: list[dict] = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} is empty.")

        self.transform = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)

        forgery_label = torch.tensor([1.0 if row["label"] == "forged" else 0.0])

        tamper_type_label = torch.zeros(len(TAMPER_TYPES))
        if row["tamper_type"] in TAMPER_TYPES:
            tamper_type_label[TAMPER_TYPES.index(row["tamper_type"])] = 1.0

        return {
            "image": image,
            "forgery_label": forgery_label,
            "tamper_type_label": tamper_type_label,
            "filepath": row["filepath"],
        }

    def class_counts(self) -> dict[str, int]:
        """Useful for computing pos_weight for imbalanced BCE loss (Section 7.5)."""
        genuine = sum(1 for r in self.rows if r["label"] == "genuine")
        forged = sum(1 for r in self.rows if r["label"] == "forged")
        return {"genuine": genuine, "forged": forged}


def stratified_split_manifest(
    manifest_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Splits the manifest into train/val/test (60/20/20 by default, matching
    Section 7.3), stratified by label so class balance is preserved across
    splits.
    """
    rng = np.random.default_rng(seed)

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train_rows, val_rows, test_rows = [], [], []
    for label, group in by_label.items():
        idx = rng.permutation(len(group))
        n_train = int(len(group) * train_frac)
        n_val = int(len(group) * val_frac)
        train_rows += [group[i] for i in idx[:n_train]]
        val_rows += [group[i] for i in idx[n_train:n_train + n_val]]
        test_rows += [group[i] for i in idx[n_train + n_val:]]

    for path, split_rows in [(train_path, train_rows), (val_path, val_rows), (test_path, test_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"Split {len(rows)} rows -> train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
