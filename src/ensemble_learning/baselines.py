"""Independent logistic regression benchmark; preprocessing fitted on TRAIN only."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def run_independent_logreg(
    split: TemporalHoldout,
    *,
    seed: int = 42,
    evaluate_test: bool = False,
) -> dict:
    cols = feature_columns(split.train)
    outcome: dict = {"method": "independent_logreg", "seed": seed, "features": cols, "horizons": {}}
    for h in HORIZONS:
        label = f"y_{h}m"
        classes = set(split.train[label].unique())
        if classes != {0, 1}:
            raise ValueError(f"{label} training fold contains classes {sorted(classes)}; needs both")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        model.fit(split.train[cols], split.train[label])
        val_probability = model.predict_proba(split.validation[cols])[:, 1]
        per_horizon = {
            "validation": binary_metrics(split.validation[label].to_numpy(), val_probability)
        }
        if evaluate_test:
            test_probability = model.predict_proba(split.test[cols])[:, 1]
            per_horizon["test"] = binary_metrics(split.test[label].to_numpy(), test_probability)
        outcome["horizons"][str(h)] = per_horizon
    return outcome
