"""
Render fictional GENUINE bank statements / payslips as page images, for the visual CNN branch.

Replaces the old placeholder images (random noise with a black bar), which carry no document
structure for a CNN to learn tampering against. Reuses the fictional document and page renderer
from extras/ai_doc_forensics (adf), so no real bank, person or account is involved.

python tools/make_genuine_documents.py --n 400 --out_dir outputs/genuine_documents --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extras" / "ai_doc_forensics"))

from adf import doc as D            # noqa: E402
from adf import render as R         # noqa: E402
from adf import signature as S      # noqa: E402
from adf.config import GenConfig    # noqa: E402


def make_genuine_documents(n: int, out_dir: str, seed: int = 42, p_rescan: float = 0.3) -> list[Path]:
    """Write n genuine page images (PNG, lossless) and return their paths.

    About `p_rescan` of the pages get a print-and-scan style finish (slight rotation, softening, grain)
    so that 'noisy' does not simply mean 'forged'.
    """
    cfg = GenConfig(seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        doc_type = "statement" if rng.random() < cfg.p_statement else "payslip"
        d = D.make_document(rng, cfg, doc_type)
        ident = S.make_identity(rng)
        sig = S.render_signature(ident, rng, sigma=1.5)
        style = dict(family=str(rng.choice(["sans", "serif"])),
                     accent=tuple(int(v) for v in rng.choice([[24, 62, 98], [20, 90, 70], [90, 30, 60], [60, 60, 70]])),
                     ink=(int(rng.integers(12, 40)),) * 3)
        arr, _, _ = R.render_page(d, style, rng, 0.0, sig)
        if rng.random() < p_rescan:
            arr = R.effect_rescan(arr, rng)
        p = out / f"doc_{i:04d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        paths.append(p)
    print(f"Wrote {len(paths)} genuine documents to {out}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out_dir", default="outputs/genuine_documents")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_genuine_documents(a.n, a.out_dir, a.seed)
