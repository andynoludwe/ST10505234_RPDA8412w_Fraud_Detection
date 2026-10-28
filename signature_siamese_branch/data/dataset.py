"""PyTorch Dataset for signature pairs (manifest columns: img_a, img_b, label, ...)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_H, IMG_W = 64, 160


def _load(path: Path, augment: bool, rng: np.random.Generator) -> torch.Tensor:
    img = Image.open(path).convert("L")
    if augment:  # small independent shift/scale of each image; keeps the identity, adds nuisance variation
        dx, dy = int(rng.integers(-6, 7)), int(rng.integers(-4, 5))
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return ((x - 0.5) / 0.5).unsqueeze(0)  # (1, H, W), ink is negative


class SignaturePairDataset(Dataset):
    def __init__(self, manifest_path: str, data_root: str = ".", augment: bool = False, seed: int = 0):
        with open(manifest_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} is empty.")
        self.root = Path(data_root)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _p(self, s: str) -> Path:
        p = Path(s.replace("\\", "/"))
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, i: int):
        r = self.rows[i]
        return {
            "img_a": _load(self._p(r["img_a"]), self.augment, self.rng),
            "img_b": _load(self._p(r["img_b"]), self.augment, self.rng),
            "label": torch.tensor([float(r["label"])]),  # 1 = forged
            "pair_id": r["pair_id"],
            "pair_type": r.get("pair_type", ""),
        }

    def class_counts(self) -> dict[str, int]:
        forged = sum(int(float(r["label"])) for r in self.rows)
        return {"genuine": len(self.rows) - forged, "forged": forged}
