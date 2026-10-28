"""
Evaluate the trained Random Forest document branch on a held-out test set
and export per-document predictions for the fusion meta-classifier.

Usage
-----
python scripts/evaluate.py \
    --checkpoint checkpoints/rf_document_branch.joblib \
    --test_manifest data/rf_test.csv \
    --output_json results/rf_metrics.json \
    --predictions_csv results/rf_predictions.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_manifest
from data.synthetic_metadata import ANOMALY_TYPES
from model.rf_document_branch import DocumentTamperRF
from utils.metrics import compute_forgery_metrics


def load_model(path: str) -> DocumentTamperRF:
    return DocumentTamperRF.load(path)


def get_predictions(model: DocumentTamperRF, rows: list[dict]):
    forgery_prob, anomaly_probs = model.predict_proba(rows)
    return forgery_prob, anomaly_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_csv", required=True)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    rows = load_manifest(args.test_manifest)

    forgery_prob, anomaly_probs = get_predictions(model, rows)
    forgery_pred = (forgery_prob >= 0.5).astype(int)
    y_true = [row["label"] for row in rows]

    overall = compute_forgery_metrics(y_true, forgery_pred, forgery_prob)
    print("--- Overall forgery detection (test set) ---")
    print(f"   precision: {overall['precision']:.4f}   recall: {overall['recall']:.4f}   "
          f"f1: {overall['f1']:.4f}   roc_auc: {overall['roc_auc']}   pr_auc: {overall['pr_auc']}")
    print(f"   confusion_matrix (rows=true, cols=pred): {overall['confusion_matrix']}")

    per_type = {}
    print("\nPer-anomaly-type breakdown:")
    for i, atype in enumerate(ANOMALY_TYPES):
        y_true_type = [row[atype] for row in rows]
        y_pred_type = (anomaly_probs[:, i] >= 0.5).astype(int)
        m = compute_forgery_metrics(y_true_type, y_pred_type, anomaly_probs[:, i])
        per_type[atype] = m
        print(f"--- {atype} ---")
        print(f"   precision: {m['precision']:.4f}   recall: {m['recall']:.4f}   "
              f"f1: {m['f1']:.4f}   roc_auc: {m['roc_auc']}   pr_auc: {m['pr_auc']}")
        print(f"   confusion_matrix (rows=true, cols=pred): {m['confusion_matrix']}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"overall": overall, "per_anomaly_type": per_type}, f, indent=2)

    Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_csv, "w", newline="") as f:
        fieldnames = ["doc_id", "forgery_prob", "forgery_pred", "label"] + [f"{a}_prob" for a in ANOMALY_TYPES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = {
                "doc_id": row["doc_id"],
                "forgery_prob": float(forgery_prob[i]),
                "forgery_pred": int(forgery_pred[i]),
                "label": row["label"],
            }
            for j, atype in enumerate(ANOMALY_TYPES):
                out_row[f"{atype}_prob"] = float(anomaly_probs[i, j])
            writer.writerow(out_row)

    print(f"\nWrote metrics to {args.output_json} and predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
