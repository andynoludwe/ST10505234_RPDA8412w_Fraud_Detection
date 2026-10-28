"""
SHAP explainability for the fusion meta-classifier, per the proposal's
interpretability requirement (given FAIS disclosure and POPIA
processing-transparency considerations).

Produces per-case and global feature-attribution values across the 10
input features (each branch's forgery_prob plus each branch's
per-anomaly-type probabilities), answering: "which branch/anomaly-type
signal drove this case's final fraud score?"
"""

import numpy as np
import shap

from data.synthetic_cases import FEATURE_COLUMNS


def explain_model(model, background_rows: list[dict], explain_rows: list[dict]):
    """
    Parameters
    ----------
    model : a fitted FusionMetaClassifier (from model.fusion_classifier)
    background_rows : rows used to build the SHAP background distribution
        (e.g. a sample of the training set)
    explain_rows : rows to actually explain (e.g. the test set, or a
        specific case flagged as high-risk)

    Returns
    -------
    shap_values : np.ndarray, shape (n_explain_rows, n_features)
    feature_names : list[str]
    """
    X_background = np.array([[row[c] for c in FEATURE_COLUMNS] for row in background_rows])
    X_explain = np.array([[row[c] for c in FEATURE_COLUMNS] for row in explain_rows])

    # KernelExplainer works with any predict_proba-exposing model, which
    # keeps this branch-agnostic (works the same whether the meta-classifier
    # is later swapped for a different small model).
    predict_fn = lambda X: model.clf.predict_proba(X)[:, 1]
    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_values = explainer.shap_values(X_explain)

    return shap_values, FEATURE_COLUMNS


def global_feature_importance(shap_values: np.ndarray, feature_names: list) -> dict:
    """Mean absolute SHAP value per feature, across all explained cases."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: -x[1]))
