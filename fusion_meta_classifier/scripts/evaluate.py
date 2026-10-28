"""
Evaluate the fusion meta-classifier on a held-out test set, compare it
against each individual branch via McNemar's test (RQ1's core comparison),
and produce SHAP explanations.

Usage
-----
python scripts/evaluate.py \
    --checkpoint checkpoints/fusion_meta_classifier.joblib \
    --train_manifest fusion_train.csv \
    --test_manifest fusion_test.csv \
    --output_json results/fusion_metrics.json \
    --predictions_csv results/fusion_predictions.csv \
    --shap_csv results/fusion_shap_values.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_manifest
from data.synthetic_cases import FEATURE_COLUMNS
from model.fusion_classifier import FusionMetaClassifier
from utils.explain import explain_model, global_feature_importance
from utils.metrics import compute_forgery_metrics, mcnemar_test


def load_model(path: str) -> FusionMetaClassifier:
    return FusionMetaClassifier.load(path)


def get_predictions(model: FusionMetaClassifier, rows: list[dict]):
    return model.predict_proba(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_manifest", required=True, help="used as SHAP background distribution")
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--shap_csv", required=True)
    parser.add_argument("--shap_sample_size", type=int, default=30,
                         help="SHAP is slow with KernelExplainer; cap explained rows for the smoke test")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    train_rows = load_manifest(args.train_manifest)
    test_rows = load_manifest(args.test_manifest)

    fusion_prob = get_predictions(model, test_rows)
    fusion_pred = (fusion_prob >= 0.5).astype(int)
    y_true = [row["label"] for row in test_rows]

    overall = compute_forgery_metrics(y_true, fusion_pred, fusion_prob)
    print("--- Fusion meta-classifier (test set) ---")
    print(f"   precision: {overall['precision']:.4f}   recall: {overall['recall']:.4f}   "
          f"f1: {overall['f1']:.4f}   roc_auc: {overall['roc_auc']}   pr_auc: {overall['pr_auc']}")
    print(f"   confusion_matrix (rows=true, cols=pred): {overall['confusion_matrix']}")

    # McNemar's test: fusion vs. each individual branch used alone
    # (thresholding each branch's own forgery_prob at 0.5, same rule as the branch's own evaluate.py)
    mcnemar_results = {}
    for branch_col in ["cnn_forgery_prob", "rf_forgery_prob", "siamese_forgery_prob"]:
        branch_pred = [1 if row[branch_col] >= 0.5 else 0 for row in test_rows]
        result = mcnemar_test(y_true, fusion_pred, branch_pred)
        mcnemar_results[branch_col] = result
        print(f"\nMcNemar's test, fusion vs. {branch_col}: "
              f"statistic={result['statistic']:.4f}, p={result['p_value']:.4f}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"overall": overall, "mcnemar_vs_branches": mcnemar_results}, f, indent=2)

    Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "fusion_prob", "fusion_pred", "label"])
        writer.writeheader()
        for row, prob, pred in zip(test_rows, fusion_prob, fusion_pred):
            writer.writerow({
                "case_id": row["case_id"],
                "fusion_prob": float(prob),
                "fusion_pred": int(pred),
                "label": row["label"],
            })

    # SHAP explainability
    print(f"\nComputing SHAP values for up to {args.shap_sample_size} test cases...")
    background = train_rows[:min(50, len(train_rows))]
    explain_rows = test_rows[:min(args.shap_sample_size, len(test_rows))]
    shap_values, feature_names = explain_model(model, background, explain_rows)

    Path(args.shap_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.shap_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id"] + list(feature_names))
        writer.writeheader()
        for row, values in zip(explain_rows, shap_values):
            out_row = {"case_id": row["case_id"]}
            for name, val in zip(feature_names, values):
                out_row[name] = float(val)
            writer.writerow(out_row)

    importance = global_feature_importance(shap_values, feature_names)
    print("\nGlobal feature importance (mean |SHAP value|):")
    for name, val in importance.items():
        print(f"   {name}: {val:.4f}")

    print(f"\nWrote metrics to {args.output_json}, predictions to {args.predictions_csv}, "
          f"SHAP values to {args.shap_csv}")


if __name__ == "__main__":
    main()
