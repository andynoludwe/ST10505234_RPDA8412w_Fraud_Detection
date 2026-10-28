"""
FusionMetaClassifier: the "small dense layer" learned meta-classifier from
the proposal's methodology (Section on fusion architecture), which combines
the three branches' outputs into a single fraud probability -- replacing
rule-based aggregation (e.g. averaging or max-voting across branches).

Implemented as a small multi-layer perceptron (scikit-learn's MLPClassifier)
over the 9-dimensional input:
    [cnn_forgery_prob, cnn_splicing_prob, cnn_copy_move_prob, cnn_inpainting_prob,
     rf_forgery_prob, rf_tampered_text_prob, rf_altered_figures_prob,
     rf_metadata_anomaly_prob, rf_layout_inconsistency_prob,
     siamese_forgery_prob]

This matches the proposal's description of a small, learned dense layer
rather than a deep architecture -- the "learning" here is entirely about
weighting/combining the three branches' already-informative outputs, not
re-deriving features from raw documents/signatures again.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.synthetic_cases import FEATURE_COLUMNS


@dataclass
class FusionMetaClassifier:
    hidden_layer_sizes: tuple = (16, 8)
    max_iter: int = 2000
    random_state: int = 42

    clf: object = field(init=False, default=None)

    def _build(self):
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,
            alpha=1e-3,
        )
        # Standardize inputs -- MLPs converge far more reliably on scaled
        # features, even though these are already bounded [0, 1] probabilities.
        self.clf = make_pipeline(StandardScaler(), mlp)

    def fit(self, rows: list[dict]) -> "FusionMetaClassifier":
        self._build()
        X = np.array([[row[c] for c in FEATURE_COLUMNS] for row in rows])
        y = np.array([row["label"] for row in rows])
        self.clf.fit(X, y)
        return self

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[row[c] for c in FEATURE_COLUMNS] for row in rows])
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FusionMetaClassifier":
        return joblib.load(path)


def build_model(**kwargs) -> FusionMetaClassifier:
    return FusionMetaClassifier(**kwargs)
