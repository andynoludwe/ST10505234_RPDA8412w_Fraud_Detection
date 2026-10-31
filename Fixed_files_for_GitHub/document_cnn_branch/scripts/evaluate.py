"""
Standalone evaluation of a trained DocumentTamperCNN checkpoint on a
held-out test manifest. Produces the full metrics report referenced in
Section 7.5: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix,
plus per-tamper-type breakdown (splicing/copy_move/inpainting).

Also exposes `get_predictions()`, which returns raw probabilities per
document — this is the function you'll call from the fusion layer /
meta-classifier to feed this branch's output into the multimodal model
(alongside the tabular model's and the Siamese signature model's outputs).

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_resnet50.pt \
        --test_manifest data/test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import DocumentTamperDataset
from data.synthetic_tamper import TAMPER_TYPES
from model.cnn_document_branch import build_model
from utils.metrics import compute_forgery_metrics, compute_tamper_type_metrics, format_metrics_report


def load_model(checkpoint_path: str, device: torch.device):
    # our own checkpoint (it stores metrics next to the weights); PyTorch >= 2.6 defaults to weights_only=True and refuses it
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def get_predictions(model, loader: DataLoader, device: torch.device) -> dict:
    """
    Returns raw predictions for every document in the loader — this is
    the function to call when producing the forgery-probability input
    for the fusion / stacking meta-classifier described in Section 6.2.
    """
    filepaths, y_true, forgery_prob, tamper_true, tamper_prob = [], [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        out = model.predict_proba(images)

        filepaths.extend(batch["filepath"])
        y_true.append(batch["forgery_label"].numpy())
        forgery_prob.append(out["forgery_prob"].cpu().numpy())
        tamper_true.append(batch["tamper_type_label"].numpy())
        tamper_prob.append(out["tamper_type_probs"].cpu().numpy())

    return {
        "filepaths": filepaths,
        "y_true": np.concatenate(y_true).ravel(),
        "forgery_prob": np.concatenate(forgery_prob).ravel(),
        "tamper_true": np.concatenate(tamper_true),
        "tamper_prob": np.concatenate(tamper_prob),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_root", default=".", help="manifest filepaths are relative to this folder")
    parser.add_argument("--output_json", default=None, help="Optional path to dump full metrics as JSON")
    parser.add_argument("--predictions_csv", default=None,
                         help="Optional path to dump per-document predictions (for the fusion layer)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (backbone={ckpt['backbone']}, val_auc={ckpt.get('val_auc', float('nan')):.4f}, val_f1={ckpt['val_f1']:.4f})")

    test_ds = DocumentTamperDataset(args.test_manifest, image_size=args.image_size, augment=False, data_root=args.data_root)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    preds = get_predictions(model, test_loader, device)

    forgery_metrics = compute_forgery_metrics(preds["y_true"], preds["forgery_prob"])
    print("\n" + format_metrics_report(forgery_metrics, "Overall forgery detection (test set)"))

    tamper_metrics = compute_tamper_type_metrics(preds["tamper_true"], preds["tamper_prob"], TAMPER_TYPES)
    print("\nPer-tamper-type breakdown:")
    for t, m in tamper_metrics.items():
        print(format_metrics_report(m, t))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"forgery": forgery_metrics, "tamper_types": tamper_metrics}, f, indent=2)
        print(f"\nSaved metrics JSON to {args.output_json}")

    if args.predictions_csv:
        import csv
        with open(args.predictions_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # doc_id + {type}_prob columns are what build_matched_cases.py / the fusion layer look for
            writer.writerow(["doc_id", "filepath", "y_true", "forgery_prob"] + [f"{t}_prob" for t in TAMPER_TYPES])
            for i, fp in enumerate(preds["filepaths"]):
                writer.writerow([Path(fp).stem, fp, int(preds["y_true"][i]), preds["forgery_prob"][i]] + list(preds["tamper_prob"][i]))
        print(f"Saved per-document predictions to {args.predictions_csv} (feed this into the fusion layer)")


if __name__ == "__main__":
    main()
