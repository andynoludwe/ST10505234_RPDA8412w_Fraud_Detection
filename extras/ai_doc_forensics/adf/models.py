"""Branch models, out-of-fold prediction, and fusion. Branch targets are binary: AI-involved (generated or edited) vs genuine."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import precision_recall_curve


def make_branch_model(branch, seed):
    if branch == "metadata":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    if branch == "signature":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=seed)


def split_by_identity(meta, test_frac=0.3, seed=0):
    """Leakage-free split: every signature identity is wholly in train or wholly in test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(meta, meta["y"], groups=meta["identity_id"]))
    assert not set(meta.iloc[tr]["identity_id"]) & set(meta.iloc[te]["identity_id"])
    return meta.index[tr], meta.index[te]


def oof_predict(make_model, X, y, groups, n_splits=5, seed=0):
    oof = np.zeros(len(y)); cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[np.nanargmax(f1[:-1])]) if len(th) else 0.5


def make_fusion(kind, seed):
    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=1000)
    return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, l2_regularization=2.0, random_state=seed)
