"""Independent classical baselines, without leakage of preprocessing from validation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_learning import HORIZONS
from ensemble_learning.contracts import feature_columns
from ensemble_learning.splits import TemporalHoldout

CLASSICAL_MODELS = ("logreg", "random_forest", "hist_gradient_boosting")


def binary_metrics(y_true: np.ndarray, positive_proba: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(positive_proba, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y_true, dtype=int)
    predicted = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "f1_binary": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def make_classical_model(name: str, seed: int):
    """Fixed exploratory defaults only; selection must be done on temporal folds."""
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100, min_samples_leaf=3, class_weight="balanced_subsample",
                n_jobs=1, random_state=seed,
            ),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=15, min_samples_leaf=5, random_state=seed,
            ),
        )
    raise ValueError(f"Unknown classical model: {name}")


def run_independent_classical(
    split: TemporalHoldout, *, model_name: str = "logreg",
    seed: int = 42, evaluate_test: bool = False,
) -> dict:
    if model_name not in CLASSICAL_MODELS:
        raise ValueError(f"Choose one of {CLASSICAL_MODELS}")
    cols = feature_columns(split.train)
    outcome: dict = {
        "method": "independent_" + model_name, "seed": seed,
        "features": cols, "horizons": {},
    }
    for h in HORIZONS:
        label = f"y_{h}m"
        classes = set(split.train[label].unique())
        if classes != {0, 1}:
            raise ValueError(f"{label} training contains classes {sorted(classes)}; needs both")
        model = make_classical_model(model_name, seed)
        model.fit(split.train[cols], split.train[label])
        validation = model.predict_proba(split.validation[cols])[:, 1]
        metrics = {"validation": binary_metrics(split.validation[label].to_numpy(), validation)}
        if evaluate_test:
            prob_test = model.predict_proba(split.test[cols])[:, 1]
            metrics["test"] = binary_metrics(split.test[label].to_numpy(), prob_test)
        outcome["horizons"][str(h)] = metrics
    return outcome


def run_independent_logreg(
    split: TemporalHoldout, *, seed: int = 42, evaluate_test: bool = False,
) -> dict:
    """Existing entry point, retained for compatibility."""
    return run_independent_classical(
        split, model_name="logreg", seed=seed, evaluate_test=evaluate_test
    )
