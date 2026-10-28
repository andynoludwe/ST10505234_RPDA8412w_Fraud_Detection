"""
Train the Random Forest document branch.

Usage
-----
python scripts/train.py \
    --train_manifest data/rf_train.csv \
    --val_manifest data/rf_val.csv \
    --n_estimators 300 \
    --output_dir checkpoints/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_manifest
from model.rf_document_branch import build_model
from utils.metrics import compute_forgery_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    train_rows = load_manifest(args.train_manifest)
    val_rows = load_manifest(args.val_manifest)

    model = build_model(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(train_rows)

    val_prob, _ = model.predict_proba(val_rows)
    val_pred = (val_prob >= 0.5).astype(int)
    val_true = [row["label"] for row in val_rows]
    val_metrics = compute_forgery_metrics(val_true, val_pred, val_prob)

    print(f"Validation — precision: {val_metrics['precision']:.4f}  "
          f"recall: {val_metrics['recall']:.4f}  f1: {val_metrics['f1']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "rf_document_branch.joblib"
    model.save(str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path} (val_f1={val_metrics['f1']:.4f})")


if __name__ == "__main__":
    main()
