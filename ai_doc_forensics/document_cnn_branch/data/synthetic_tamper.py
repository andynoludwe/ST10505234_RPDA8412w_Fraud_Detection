"""
Synthetic visual tampering generator.

Applies controlled splicing, copy-move, and inpainting manipulations to
genuine document images, producing labeled training data for
DocumentTamperCNN. This mirrors the manipulation techniques described in
Section 7.3 of the proposal and is consistent with the synthetic-data
methodology of Pushkarenko & Zaslavskyi (2024) and Ruiz et al. (2020).

This module works on document IMAGES (post-rasterisation, per Section 7.4
"Data Pre-processing"). It is deliberately separate from the tabular-level
tampering (text substitution, metadata edits) your existing synthetic
bank-statement generator already produces — this generator adds the
*visual* manipulations that the CNN branch needs to learn from.

Each function returns (tampered_image, mask, label) where:
    - tampered_image: np.ndarray, uint8, same size as input
    - mask: np.ndarray, uint8 {0,255}, region that was altered (useful
      for future localisation work / visual QA, optional for the classifier)
    - label: str, one of TAMPER_TYPES

Usage:
    from data.synthetic_tamper import generate_labeled_dataset

    generate_labeled_dataset(
        genuine_dir="data/genuine_documents",
        output_dir="data/synthetic_tampered",
        manifest_path="data/manifest.csv",
        n_per_technique=200,
    )
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

TAMPER_TYPES = ["splicing", "copy_move", "inpainting"]


def _random_rect(h: int, w: int, min_frac=0.08, max_frac=0.28) -> tuple[int, int, int, int]:
    """Random rectangle (x, y, rw, rh) sized as a fraction of the image."""
    rw = int(w * random.uniform(min_frac, max_frac))
    rh = int(h * random.uniform(min_frac, max_frac))
    x = random.randint(0, max(1, w - rw - 1))
    y = random.randint(0, max(1, h - rh - 1))
    return x, y, rw, rh


def apply_copy_move(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Copies a region of the document onto another region — simulates
    duplicating a figure/amount elsewhere in the same document.
    """
    h, w = image.shape[:2]
    out = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    x, y, rw, rh = _random_rect(h, w)
    patch = image[y:y + rh, x:x + rw].copy()

    # paste target: offset from source, kept in-bounds
    tx = min(max(0, x + random.randint(-w // 3, w // 3)), w - rw - 1)
    ty = min(max(0, y + random.randint(-h // 3, h // 3)), h - rh - 1)

    out[ty:ty + rh, tx:tx + rw] = patch
    mask[ty:ty + rh, tx:tx + rw] = 255
    return out, mask


def apply_splicing(image: np.ndarray, donor_image: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Pastes a region from a *different* document (donor_image) into this
    one — simulates combining parts of two different financial documents.
    Falls back to a synthetic solid/noisy patch if no donor is supplied.
    """
    h, w = image.shape[:2]
    out = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    x, y, rw, rh = _random_rect(h, w)

    if donor_image is not None:
        donor = cv2.resize(donor_image, (w, h))
        patch = donor[y:y + rh, x:x + rw]
    else:
        # Fallback: synthetic patch with plausible document-paper statistics
        mean_val = float(image[y:y + rh, x:x + rw].mean())
        patch = np.random.normal(mean_val, 8, (rh, rw, image.shape[2])).clip(0, 255).astype(np.uint8)

    out[y:y + rh, x:x + rw] = patch
    mask[y:y + rh, x:x + rw] = 255
    return out, mask


def apply_inpainting(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes a region (e.g. an amount, a line of text) and fills it back in
    using OpenCV inpainting — simulates deleting/hiding information on a
    document while leaving no obvious blank patch.
    """
    h, w = image.shape[:2]
    x, y, rw, rh = _random_rect(h, w, min_frac=0.05, max_frac=0.15)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y + rh, x:x + rw] = 255

    out = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return out, mask


_TAMPER_FUNCS = {
    "copy_move": lambda img, donor: apply_copy_move(img),
    "splicing": lambda img, donor: apply_splicing(img, donor),
    "inpainting": lambda img, donor: apply_inpainting(img),
}


def generate_labeled_dataset(
    genuine_dir: str,
    output_dir: str,
    manifest_path: str,
    n_per_technique: int = 200,
    include_genuine_negatives: bool = True,
    seed: int = 42,
) -> None:
    """
    Reads genuine document images from genuine_dir, applies each tampering
    technique n_per_technique times (randomly sampling source/donor images),
    and writes:
      - tampered images to output_dir/<technique>/
      - a manifest CSV with columns: filepath, label, tamper_type, source_file

    label is "genuine" or "forged"; tamper_type is one of TAMPER_TYPES or
    "none" for genuine samples. This manifest format is what
    data/dataset.py's DocumentTamperDataset expects.
    """
    random.seed(seed)
    np.random.seed(seed)

    genuine_paths = sorted(
        p for p in Path(genuine_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not genuine_paths:
        raise FileNotFoundError(
            f"No genuine document images found in {genuine_dir}. "
            "Rasterise PDFs to images first (Section 7.4 of the proposal)."
        )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for t in TAMPER_TYPES:
        (out_root / t).mkdir(exist_ok=True)

    rows = []

    if include_genuine_negatives:
        for p in genuine_paths:
            rows.append({"filepath": str(p), "label": "genuine", "tamper_type": "none", "source_file": p.name})

    for tamper_type in TAMPER_TYPES:
        func = _TAMPER_FUNCS[tamper_type]
        for i in range(n_per_technique):
            src_path = random.choice(genuine_paths)
            image = cv2.imread(str(src_path))
            if image is None:
                continue

            donor = None
            if tamper_type == "splicing" and len(genuine_paths) > 1:
                donor_path = random.choice([p for p in genuine_paths if p != src_path])
                donor = cv2.imread(str(donor_path))

            tampered, _mask = func(image, donor)

            out_path = out_root / tamper_type / f"{tamper_type}_{i:04d}_{src_path.stem}.png"
            cv2.imwrite(str(out_path), tampered)

            rows.append({
                "filepath": str(out_path),
                "label": "forged",
                "tamper_type": tamper_type,
                "source_file": src_path.name,
            })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "tamper_type", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    n_forged = sum(1 for r in rows if r["label"] == "forged")
    n_genuine = sum(1 for r in rows if r["label"] == "genuine")
    print(f"Wrote {len(rows)} rows to {manifest_path} ({n_genuine} genuine, {n_forged} forged).")


if __name__ == "__main__":
    # Smoke test using synthetic dummy "documents" (plain images), so this
    # runs standalone without real document data on hand.
    tmp_dir = Path("/tmp/genuine_docs_smoketest")
    tmp_dir.mkdir(exist_ok=True)
    for i in range(4):
        img = np.random.randint(200, 255, (600, 450, 3), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (400, 80), (30, 30, 30), -1)  # fake header bar
        cv2.imwrite(str(tmp_dir / f"doc_{i}.png"), img)

    generate_labeled_dataset(
        genuine_dir=str(tmp_dir),
        output_dir="/tmp/synthetic_tampered_smoketest",
        manifest_path="/tmp/manifest_smoketest.csv",
        n_per_technique=3,
    )
