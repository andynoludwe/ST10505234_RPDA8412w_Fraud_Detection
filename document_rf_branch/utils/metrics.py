"""
Metrics utilities for the RF document branch — mirrors the CNN branch's
utils/metrics.py so outputs are directly comparable (RQ1's metric list:
precision/recall/F1/ROC-AUC/PR-AUC, plus McNemar's test for statistical
comparison between branches/baselines).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_forgery_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """
    McNemar's test comparing two classifiers' predictions on the same
    test set (Section 7.5: statistical comparison against baselines).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    # b (0,1) = a correct, b wrong ; c (1,0) = a wrong, b correct
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))

    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}

    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # chi-square survival function, 1 dof, via a small helper to avoid a
    # scipy hard dependency (scipy is fine too — swap in scipy.stats.chi2
    # if you already depend on it elsewhere)
    p_value = _chi2_sf_1dof(statistic)

    return {"statistic": float(statistic), "p_value": float(p_value), "n01": n01, "n10": n10}


def _chi2_sf_1dof(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom."""
    import math
    return math.erfc(math.sqrt(x / 2))
