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
    def __init__(self, manifest_path: str, image_size: int = 224, augment: bool = False, data_root: str = "."):
        # filepaths in the manifest are relative to data_root (absolute paths still work)
        self.root = Path(data_root)
        self.rows: list[dict] = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} is empty.")

        self.transform = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, filepath: str) -> Path:
        p = Path(filepath.replace("\\", "/"))
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(self._resolve(row["filepath"])).convert("RGB")
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
    Splits an existing manifest into train/val/test (60/20/20 by default, Section 7.3), keeping all
    images derived from the same source document together (grouped by `source_file`) so one document
    never appears on both sides of a split.

    Limitation: a splicing donor can still sit in another split. For reported results use
    synthetic_tamper.generate_split_datasets(), which splits first and tampers afterwards.
    """
    rng = np.random.default_rng(seed)

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    sources = sorted({r["source_file"] for r in rows})
    order = rng.permutation(len(sources))
    n_train = int(len(sources) * train_frac)
    n_val = int(len(sources) * val_frac)
    assign = {}
    for rank, i in enumerate(order):
        assign[sources[i]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[assign[r["source_file"]]].append(r)

    for path, split_rows in [(train_path, out["train"]), (val_path, out["val"]), (test_path, out["test"])]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"Split {len(rows)} rows ({len(sources)} source documents) -> "
          f"train={len(out['train'])}, val={len(out['val'])}, test={len(out['test'])}")
