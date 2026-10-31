"""
Evaluation metrics matching Section 4.2 / 7.5 of the proposal:
Precision, Recall, F1 (fraud class), ROC-AUC, PR-AUC, plus confusion
matrix and McNemar's test for comparing two models' predictions
(used later to compare the fused multimodal model against each
single-modality baseline, per RQ1).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_forgery_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Core metrics for the binary forged/genuine head.

    Args:
        y_true: array of 0/1 ground-truth labels (1 = forged)
        y_prob: array of predicted probabilities (forged)
        threshold: decision threshold for precision/recall/F1

    Returns dict with precision, recall, f1, roc_auc, pr_auc, confusion_matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC-AUC / PR-AUC require both classes present
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def compute_tamper_type_metrics(y_true: np.ndarray, y_prob: np.ndarray, tamper_types: list[str], threshold: float = 0.5) -> dict:
    """
    Per-tamper-type metrics for the multi-label head (splicing/copy_move/inpainting).
    y_true, y_prob: shape (N, len(tamper_types))
    """
    results = {}
    for i, t in enumerate(tamper_types):
        results[t] = compute_forgery_metrics(y_true[:, i], y_prob[:, i], threshold)
    return results


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test for comparing two models' predictions on the same
    test set — used to test whether the fused multimodal model's
    performance difference vs. a single-modality baseline is
    statistically significant (Section 7.5).

    Returns the contingency counts and a chi-square statistic + p-value
    (uses the continuity-corrected version, standard for McNemar's test).
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong ; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {"b": b, "c": c, "statistic": float(statistic), "p_value": float(p_value)}


def format_metrics_report(metrics: dict, model_name: str = "Model") -> str:
    lines = [f"--- {model_name} ---"]
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if key in metrics:
            val = metrics[key]
            lines.append(f"  {key:>10}: {val:.4f}" if not np.isnan(val) else f"  {key:>10}: n/a")
    if "confusion_matrix" in metrics:
        lines.append(f"  confusion_matrix (rows=true, cols=pred): {metrics['confusion_matrix']}")
    return "\n".join(lines)
