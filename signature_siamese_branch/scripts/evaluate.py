"""
Evaluate a trained Siamese checkpoint on a held-out manifest and export per-pair forgery probabilities
for the fusion layer.

python scripts/evaluate.py --checkpoint checkpoints/best_siamese.pt --test_manifest <test.csv> \
    --output_json results/siamese_metrics.json --predictions_csv results/siamese_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import SignaturePairDataset
from model.siamese_cnn import build_model
from utils.metrics import compute_forgery_metrics, format_metrics_report


@torch.no_grad()
def get_predictions(model, loader, device) -> dict:
    ids, types, ys, ps = [], [], [], []
    model.eval()
    for b in loader:
        p = model.predict_proba(b["img_a"].to(device), b["img_b"].to(device)).cpu().numpy().ravel()
        ids += list(b["pair_id"])
        types += list(b["pair_type"])
        ys.append(b["label"].numpy().ravel())
        ps.append(p)
    return {"pair_id": ids, "pair_type": types, "y_true": np.concatenate(ys), "forgery_prob": np.concatenate(ps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--predictions_csv", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ck['epoch']} (val_auc={ck['val_auc']:.4f})")

    ds = SignaturePairDataset(a.test_manifest, a.data_root)
    pr = get_predictions(model, DataLoader(ds, batch_size=a.batch_size), device)
    m = compute_forgery_metrics(pr["y_true"], pr["forgery_prob"])
    print("\n" + format_metrics_report(m, "Siamese signature verification (test set)"))

    by_type = {}
    types = np.array(pr["pair_type"])
    for t in sorted(set(pr["pair_type"])):
        sel = types == t
        by_type[t] = {"n": int(sel.sum()), "mean_forgery_prob": float(pr["forgery_prob"][sel].mean()),
                      "flagged_rate@0.5": float((pr["forgery_prob"][sel] >= 0.5).mean())}
    print("\nBy pair type (flagged rate = share scored >= 0.5; for genuine pairs this is the false-positive rate):")
    for t, v in by_type.items():
        print(f"   {t:16s} n={v['n']:4d}  mean_p={v['mean_forgery_prob']:.3f}  flagged={v['flagged_rate@0.5']:.3f}")

    if a.output_json:
        Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.output_json, "w") as f:
            json.dump({"forgery": m, "by_pair_type": by_type}, f, indent=2)
    if a.predictions_csv:
        Path(a.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.predictions_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "y_true", "forgery_prob", "pair_type"])
            for i in range(len(pr["pair_id"])):
                w.writerow([pr["pair_id"][i], int(pr["y_true"][i]), float(pr["forgery_prob"][i]), pr["pair_type"][i]])
        print(f"\nSaved per-pair predictions to {a.predictions_csv} (feed into the fusion layer)")


if __name__ == "__main__":
    main()
