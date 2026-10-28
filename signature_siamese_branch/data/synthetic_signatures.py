"""
Synthetic signature pairs for the Siamese branch (all data are fictional).

An *identity* is a handful of smooth strokes (control points in a 220x90 box) plus a pen width.
A signature image is one noisy rendering of an identity.

Pair types (label 1 = forged, 0 = genuine, the same convention as the other branches):
    genuine          two independent renderings of the SAME identity           label 0
    skilled_forgery  reference rendering vs. a forger's attempt at the same
                     identity (same strokes, larger point jitter)               label 1
    random_forgery   reference rendering vs. a different identity               label 1

Leakage control: identities are split FIRST (train / val / test), and every pair is built only
from identities inside one split. The test identities are therefore never seen in training, so the
test score measures verifying a writer the model has never met, not recognising known writers.

All jitter sizes are assumptions about how forgeries differ from genuine variation; they are not
measurements of real signatures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

W, H = 220, 90


def make_identity(rng: np.random.Generator) -> dict:
    strokes = []
    for _ in range(int(rng.integers(2, 5))):
        n = int(rng.integers(4, 8))
        x = np.sort(rng.uniform(15, W - 15, n))
        y = rng.uniform(15, H - 15, n)
        strokes.append(np.stack([x, y], 1))
    return {"strokes": strokes, "width": float(rng.uniform(1.6, 3.2))}


def _smooth(points: np.ndarray, k: int = 40) -> np.ndarray:
    t = np.linspace(0, 1, len(points))
    tt = np.linspace(0, 1, k * (len(points) - 1))
    return np.stack([np.interp(tt, t, points[:, 0]), np.interp(tt, t, points[:, 1])], 1)


def render_signature(identity: dict, rng: np.random.Generator, sigma: float,
                     scale_jit: float = 0.05, rot_deg: float = 3.0,
                     pen_uniform: bool = False, pen_noise: bool = True) -> np.ndarray:
    """One grayscale signature image (uint8, white background). `sigma` = point jitter in pixels."""
    img = np.full((H, W), 255, np.uint8)
    ang = np.deg2rad(rng.normal(0, rot_deg))
    s = 1 + rng.normal(0, scale_jit)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    c = np.array([W / 2, H / 2])
    for pts in identity["strokes"]:
        p = pts + rng.normal(0, sigma, pts.shape)
        p = (p - c) @ R.T + c
        curve = _smooth(p)
        for i in range(len(curve) - 1):
            w = identity["width"]
            if not pen_uniform:  # pen pressure varies along a human stroke
                w = w * (0.75 + 0.5 * np.sin(i / 7.0 + rng.uniform(0, 6)) ** 2)
            cv2.line(img, tuple(np.round(curve[i]).astype(int)), tuple(np.round(curve[i + 1]).astype(int)),
                     0, max(1, int(round(w))), cv2.LINE_AA)
    if pen_noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


def _rel(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def generate_pairs_dataset(
    pairs_dir: str,
    manifest_dir: str,
    data_root: str = ".",
    n_identities: int = 60,
    genuine_per_id: int = 8,
    forged_per_id: int = 8,
    p_skilled: float = 0.6,
    skilled_sigma: tuple[float, float] = (1.8, 5.0),
    genuine_sigma: float = 1.5,
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, str]:
    """
    Writes signature images under `pairs_dir/<split>/` and one manifest per split in `manifest_dir`
    (columns: pair_id, img_a, img_b, identity_a, identity_b, label, pair_type, split). Image paths
    are relative to `data_root`. Returns {split: manifest_path}.
    """
    rng = np.random.default_rng(seed)
    root = Path(data_root)
    ids = [f"id_{i:03d}" for i in range(n_identities)]
    identities = {i: make_identity(rng) for i in ids}

    order = list(rng.permutation(n_identities))
    n_tr, n_va = int(n_identities * fracs[0]), int(n_identities * fracs[1])
    split_ids = {
        "train": [ids[i] for i in order[:n_tr]],
        "val": [ids[i] for i in order[n_tr:n_tr + n_va]],
        "test": [ids[i] for i in order[n_tr + n_va:]],
    }

    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    manifests, counter = {}, 0
    for split, members in split_ids.items():
        if len(members) < 2:
            raise ValueError(f"Split '{split}' has {len(members)} identities; raise n_identities.")
        out = Path(pairs_dir) / split
        out.mkdir(parents=True, exist_ok=True)
        rows = []

        def save(ident_a, ident_b, kind):
            nonlocal counter
            pid = f"pair_{counter:05d}"
            counter += 1
            ref = render_signature(identities[ident_a], rng, sigma=genuine_sigma)
            if kind == "genuine":
                qry = render_signature(identities[ident_a], rng, sigma=genuine_sigma)
            elif kind == "skilled_forgery":
                qry = render_signature(identities[ident_a], rng, sigma=float(rng.uniform(*skilled_sigma)),
                                       pen_uniform=bool(rng.random() < 0.5), pen_noise=bool(rng.random() < 0.5))
            else:
                qry = render_signature(identities[ident_b], rng, sigma=genuine_sigma)
            pa, pb = out / f"{pid}_a.png", out / f"{pid}_b.png"
            cv2.imwrite(str(pa), ref)
            cv2.imwrite(str(pb), qry)
            rows.append({"pair_id": pid, "img_a": _rel(pa, root), "img_b": _rel(pb, root),
                         "identity_a": ident_a, "identity_b": ident_b if kind == "random_forgery" else ident_a,
                         "label": 0 if kind == "genuine" else 1, "pair_type": kind, "split": split})

        for ident in members:
            for _ in range(genuine_per_id):
                save(ident, ident, "genuine")
            for _ in range(forged_per_id):
                if rng.random() < p_skilled:
                    save(ident, ident, "skilled_forgery")
                else:
                    other = str(rng.choice([m for m in members if m != ident]))
                    save(ident, other, "random_forgery")

        order_rows = rng.permutation(len(rows))
        rows = [rows[i] for i in order_rows]
        mpath = Path(manifest_dir) / f"{split}.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        manifests[split] = str(mpath)
        n_forged = sum(r["label"] for r in rows)
        print(f"{split}: {len(members)} identities -> {len(rows)} pairs ({len(rows) - n_forged} genuine, {n_forged} forged)")
    return manifests


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        generate_pairs_dataset(f"{tmp}/pairs", f"{tmp}/manifests", data_root=tmp, n_identities=10)
