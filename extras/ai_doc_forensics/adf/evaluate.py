"""Metrics, paired comparison tests, bootstrap intervals and figures."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve)


def metrics(y, p, thr):
    pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(thr), precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred)),
                specificity=float(tn / (tn + fp)) if tn + fp else float("nan"), f1=float(f1_score(y, pred)),
                roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def mcnemar(y, pred_a, pred_b):
    """Paired test on the same cases. b = A right/B wrong, c = A wrong/B right. Exact binomial when b+c is small."""
    a_ok = pred_a == y; b_ok = pred_b == y
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum()); n = b + c
    if n == 0:
        return dict(b=b, c=c, statistic=0.0, p_value=1.0, method="none")
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue); return dict(b=b, c=c, statistic=float(min(b, c)), p_value=p, method="exact")
    s = (abs(b - c) - 1) ** 2 / n; return dict(b=b, c=c, statistic=float(s), p_value=float(stats.chi2.sf(s, 1)), method="chi2_cc")


def bootstrap_ci(y, p, thr, n_boot=500, seed=0, other=None):
    """95% CI for AUC and F1 (and, if `other` scores are given, the paired AUC difference p - other)."""
    rng = np.random.default_rng(seed); n = len(y); auc, f1, dd = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2: continue
        auc.append(roc_auc_score(y[i], p[i])); f1.append(f1_score(y[i], (p[i] >= thr).astype(int)))
        if other is not None: dd.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], other[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(auc_ci=q(auc), f1_ci=q(f1))
    if other is not None: out["auc_diff_ci"] = q(dd); out["auc_diff_mean"] = float(np.mean(dd))
    return out


def recall_by(meta_test, pred, col):
    d = meta_test.assign(pred=pred)
    return d.groupby(col).apply(lambda g: pd.Series({"n": len(g), "flagged_rate": float(g["pred"].mean())}), include_groups=False)
