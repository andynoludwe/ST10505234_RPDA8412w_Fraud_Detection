"""
RandomForestDocumentBranch: the tabular metadata/OCR/font baseline.

Mirrors the CNN branch's dual-head design (overall forgery probability +
per-anomaly-type probabilities) so its output plugs into the fusion
meta-classifier the same way:

    forgery_prob        -> float, P(document is forged)
    anomaly_type_probs   -> dict[str, float], one per ANOMALY_TYPE

Two scikit-learn RandomForestClassifiers are trained internally:
    - `forgery_clf`  : binary genuine/forged classifier
    - `anomaly_clf`  : MultiOutputClassifier(RandomForestClassifier) over
                        the four anomaly-type flags
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from data.synthetic_metadata import ANOMALY_TYPES, FEATURE_NAMES


@dataclass
class DocumentTamperRF:
    n_estimators: int = 300
    max_depth: int | None = None
    class_weight: str | None = "balanced"
    random_state: int = 42

    forgery_clf: RandomForestClassifier = field(init=False, default=None)
    anomaly_clf: MultiOutputClassifier = field(init=False, default=None)

    def _build(self):
        self.forgery_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.anomaly_clf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        )

    def fit(self, rows: list[dict]) -> "DocumentTamperRF":
        self._build()
        X = np.array([[row[f] for f in FEATURE_NAMES] for row in rows])
        y_forgery = np.array([row["label"] for row in rows])
        y_anomaly = np.array([[row[a] for a in ANOMALY_TYPES] for row in rows])

        self.forgery_clf.fit(X, y_forgery)
        self.anomaly_clf.fit(X, y_anomaly)
        return self

    def predict_proba(self, rows: list[dict]):
        """
        Returns
        -------
        forgery_prob : np.ndarray, shape (n,)
        anomaly_probs : np.ndarray, shape (n, len(ANOMALY_TYPES))
        """
        X = np.array([[row[f] for f in FEATURE_NAMES] for row in rows])
        forgery_prob = self.forgery_clf.predict_proba(X)[:, 1]

        per_label_probs = self.anomaly_clf.predict_proba(X)
        # MultiOutputClassifier returns a list of (n, 2) arrays, one per label
        anomaly_probs = np.column_stack([p[:, 1] for p in per_label_probs])

        return forgery_prob, anomaly_probs

    def feature_importances(self) -> dict:
        return dict(zip(FEATURE_NAMES, self.forgery_clf.feature_importances_))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DocumentTamperRF":
        return joblib.load(path)


def build_model(**kwargs) -> DocumentTamperRF:
    return DocumentTamperRF(**kwargs)
