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
import tempfile
from pathlib import Path

import cv2
import numpy as np

TAMPER_TYPES = ["splicing", "copy_move", "inpainting"]


def _content_center(image: np.ndarray, block: int = 24):
    """Centre of a random text-bearing block (some ink, not a solid fill such as a header band)."""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    ink = (gray < 180).astype(np.float32)
    gh, gw = ink.shape[0] // block, ink.shape[1] // block
    if gh == 0 or gw == 0:
        return None
    frac = ink[:gh * block, :gw * block].reshape(gh, block, gw, block).mean((1, 3))
    cand = np.argwhere((frac > 0.03) & (frac < 0.7))
    if len(cand) == 0:
        return None
    by, bx = cand[random.randrange(len(cand))]
    return int(bx * block + block // 2), int(by * block + block // 2)


def _random_rect(h: int, w: int, min_frac=0.08, max_frac=0.28, image: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """
    Random rectangle (x, y, rw, rh) sized as a fraction of the image. If `image` is given the
    rectangle is centred on document content, because on a mostly blank page a random rectangle
    usually changes nothing and the 'forged' image would be identical to the genuine one.
    """
    rw = int(w * random.uniform(min_frac, max_frac))
    rh = int(h * random.uniform(min_frac, max_frac))
    centre = _content_center(image) if image is not None else None
    if centre is None:
        x = random.randint(0, max(1, w - rw - 1))
        y = random.randint(0, max(1, h - rh - 1))
    else:
        x = int(np.clip(centre[0] - rw // 2, 0, max(1, w - rw - 1)))
        y = int(np.clip(centre[1] - rh // 2, 0, max(1, h - rh - 1)))
    return x, y, rw, rh


def apply_copy_move(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Copies a region of the document onto another region — simulates
    duplicating a figure/amount elsewhere in the same document.
    """
    h, w = image.shape[:2]
    out = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    x, y, rw, rh = _random_rect(h, w, image=image)
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

    x, y, rw, rh = _random_rect(h, w, image=image)

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
    x, y, rw, rh = _random_rect(h, w, min_frac=0.05, max_frac=0.15, image=image)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y + rh, x:x + rw] = 255

    out = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return out, mask


def _until_visible(func, image: np.ndarray, donor, tries: int = 12, min_pixels: int = 150):
    """Re-draw the manipulation until it really changes the page (a no-op edit must not be labelled forged)."""
    out, mask = func(image, donor)
    for _ in range(tries):
        if (np.abs(out.astype(np.int16) - image.astype(np.int16)).max(axis=2) > 12).sum() >= min_pixels:
            break
        out, mask = func(image, donor)
    return out, mask


_TAMPER_FUNCS = {
    "copy_move": lambda img, donor: _until_visible(lambda i, d: apply_copy_move(i), img, donor),
    "splicing": lambda img, donor: _until_visible(apply_splicing, img, donor),
    "inpainting": lambda img, donor: _until_visible(lambda i, d: apply_inpainting(i), img, donor),
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

    NOTE: this builds ONE pool and the split happens afterwards, so a source document (and a
    splicing donor) can end up in both train and test. For anything you report, use
    generate_split_datasets() instead, which splits the genuine documents first.
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


def _list_images(genuine_dir: str) -> list[Path]:
    paths = sorted(p for p in Path(genuine_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(
            f"No genuine document images found in {genuine_dir}. "
            "Rasterise PDFs to images first (Section 7.4 of the proposal)."
        )
    return paths


def _rel(path: Path, root: Path) -> str:
    """Manifest paths are stored relative to `root` with forward slashes, so they work on any machine."""
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def generate_split_datasets(
    genuine_dir: str,
    output_dir: str,
    manifest_dir: str,
    data_root: str = ".",
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
    forged_per_genuine: float = 1.0,
    seed: int = 42,
) -> dict[str, str]:
    """
    Leakage-free dataset builder: split the GENUINE DOCUMENTS first, then create the tampered
    variants of each split using only that split's documents (as sources and as splicing donors).

    Why: if tampered variants are made first and split afterwards, the same underlying document
    (or a donor patch) appears in train and test, and the test score then measures memorisation of
    the document, not detection of tampering.

    forged_per_genuine : tampered images per technique, as a multiple of the number of genuine
        documents in the split. 1.0 gives genuine:forged = 1:3 (one variant of each technique).

    Writes train/val/test manifests (columns: filepath, label, tamper_type, source_file, split) to
    `manifest_dir`, with filepaths relative to `data_root`. Returns {split: manifest_path}.
    """
    rng = random.Random(seed)
    np_rng_state = seed
    random.seed(seed)
    np.random.seed(np_rng_state)

    root = Path(data_root)
    genuine_paths = _list_images(genuine_dir)
    order = list(genuine_paths)
    rng.shuffle(order)
    n = len(order)
    n_tr, n_va = int(n * fracs[0]), int(n * fracs[1])
    groups = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    for name, g in groups.items():
        if len(g) < 2:
            raise ValueError(f"Split '{name}' has {len(g)} documents; provide more genuine documents.")

    out_root = Path(output_dir)
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, docs in groups.items():
        for t in TAMPER_TYPES:
            (out_root / split / t).mkdir(parents=True, exist_ok=True)
        rows = [{"filepath": _rel(p, root), "label": "genuine", "tamper_type": "none",
                 "source_file": p.name, "split": split} for p in docs]
        n_var = max(1, round(forged_per_genuine * len(docs)))
        for tamper_type in TAMPER_TYPES:
            func = _TAMPER_FUNCS[tamper_type]
            sources = [docs[i % len(docs)] for i in range(n_var)]
            rng.shuffle(sources)
            for i, src_path in enumerate(sources):
                image = cv2.imread(str(src_path))
                donor = None
                if tamper_type == "splicing":
                    donor = cv2.imread(str(rng.choice([d for d in docs if d != src_path])))
                tampered, _mask = func(image, donor)
                out_path = out_root / split / tamper_type / f"{tamper_type}_{i:04d}_{src_path.stem}.png"
                cv2.imwrite(str(out_path), tampered)
                rows.append({"filepath": _rel(out_path, root), "label": "forged", "tamper_type": tamper_type,
                             "source_file": src_path.name, "split": split})
        rng.shuffle(rows)
        mpath = Path(manifest_dir) / f"{split}.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filepath", "label", "tamper_type", "source_file", "split"])
            w.writeheader()
            w.writerows(rows)
        manifests[split] = str(mpath)
        n_g = sum(r["label"] == "genuine" for r in rows)
        print(f"{split}: {len(docs)} source documents -> {len(rows)} images ({n_g} genuine, {len(rows) - n_g} forged)")
    return manifests


if __name__ == "__main__":
    # Smoke test on plain placeholder images (plumbing only; real runs use rendered documents).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "genuine").mkdir()
        for i in range(10):
            img = np.random.randint(200, 255, (600, 450, 3), dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (400, 80), (30, 30, 30), -1)
            cv2.imwrite(str(tmp / "genuine" / f"doc_{i}.png"), img)
        generate_split_datasets(str(tmp / "genuine"), str(tmp / "out"), str(tmp / "manifests"), data_root=str(tmp))
