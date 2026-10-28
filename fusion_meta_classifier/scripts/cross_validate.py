"""
Repeated stratified cross-validation of the fusion meta-classifier against simple baselines.

Why this exists: with real branch outputs the held-out case set is small (tens of cases), so a single
train/test split and a McNemar test on ~50 cases are very noisy. Cross-validating over ALL cases and
repeating with different shuffles gives a mean and spread that can actually be compared.

Compared on identical folds (ROC-AUC, F1 at 0.5):
    fusion (MLP)         the learned meta-classifier
    logistic regression  a plain linear stacker over the same 10 inputs
    mean of 3 branches   mean(cnn_forgery_prob, rf_forgery_prob, siamese_forgery_prob), no learning
    max of 3 branches    flag if any branch is suspicious
    each branch alone    cnn / rf / siamese forgery_prob

python scripts/cross_validate.py --manifest fusion_manifest.csv --output_json results/fusion_cv.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_manifest
from data.synthetic_cases import FEATURE_COLUMNS
from model.fusion_classifier import FusionMetaClassifier

BRANCHES = ["cnn_forgery_prob", "rf_forgery_prob", "siamese_forgery_prob"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, nargs="+", help="one or more fusion manifests (all rows are pooled)")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = [r for m in a.manifest for r in load_manifest(m)]
    X = np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows])
    y = np.array([r["label"] for r in rows])
    col = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
    print(f"{len(rows)} cases ({int(y.sum())} fraudulent, {int((1 - y).sum())} genuine); {a.folds}-fold x {a.repeats} repeats")

    scores = {k: {"auc": [], "f1": []} for k in
              ["fusion_mlp", "logistic_regression", "mean_of_branches", "max_of_branches",
               "cnn_alone", "rf_alone", "siamese_alone"]}

    def log(name, yt, p):
        scores[name]["auc"].append(roc_auc_score(yt, p))
        scores[name]["f1"].append(f1_score(yt, (p >= 0.5).astype(int), zero_division=0))

    cv = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=a.seed)
    for tr, te in cv.split(X, y):
        tr_rows = [dict(rows[i]) for i in tr]
        te_rows = [dict(rows[i]) for i in te]
        log("fusion_mlp", y[te], FusionMetaClassifier(random_state=a.seed).fit(tr_rows).predict_proba(te_rows))
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        log("logistic_regression", y[te], lr.predict_proba(X[te])[:, 1])
        b = X[te][:, [col[c] for c in BRANCHES]]
        log("mean_of_branches", y[te], b.mean(1))
        log("max_of_branches", y[te], b.max(1))
        log("cnn_alone", y[te], X[te][:, col["cnn_forgery_prob"]])
        log("rf_alone", y[te], X[te][:, col["rf_forgery_prob"]])
        log("siamese_alone", y[te], X[te][:, col["siamese_forgery_prob"]])

    out = {"n_cases": len(rows), "folds": a.folds, "repeats": a.repeats, "results": {}}
    print(f"\n{'model':22s} {'ROC-AUC':>16s} {'F1@0.5':>16s}")
    for k, v in scores.items():
        r = {m: {"mean": float(np.mean(v[m])), "std": float(np.std(v[m]))} for m in ("auc", "f1")}
        out["results"][k] = r
        print(f"{k:22s} {r['auc']['mean']:.3f} +/- {r['auc']['std']:.3f}   {r['f1']['mean']:.3f} +/- {r['f1']['std']:.3f}")
    print("\n(std is across fold/repeat resamples of the SAME cases; it is not a confidence interval for new data)")
    Path(a.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output_json, "w"), indent=2)


if __name__ == "__main__":
    main()
