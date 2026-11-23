"""Chronological, out-of-fold stacking helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from ensemble_learning.ensembles import _normalize_predictions
from ensemble_learning.pairing import KEYS


def chronological_fold_schedule(
    chronological_folds: Sequence[str], prediction_folds: Sequence[str],
) -> list[dict[str, object]]:
    """Return prediction folds with strictly earlier folds available for meta fit."""
    order = {fold: index for index, fold in enumerate(chronological_folds)}
    if len(order) != len(chronological_folds) or len(set(prediction_folds)) != len(prediction_folds):
        raise ValueError("Fold names must be unique")
    if any(fold not in order for fold in prediction_folds):
        raise ValueError("Prediction folds must be included in chronological order")
    return [
        {"prediction_fold": fold,
         "meta_train_folds": [prior for prior in chronological_folds if order[prior] < order[fold]]}
        for fold in prediction_folds
        if any(order[prior] < order[fold] for prior in chronological_folds)
    ]


def stacking_feature_frame(members: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join base probabilities by exact prediction key and preserve true labels."""
    if not members:
        raise ValueError("At least one base prediction member is required")
    names = list(members)
    normalized = {name: _normalize_predictions(frame, name) for name, frame in members.items()}
    reference = normalized[names[0]]
    features = reference[KEYS + ["y_true"]].copy()
    for name, frame in normalized.items():
        if not reference[KEYS].equals(frame[KEYS]):
            raise ValueError(f"{name} has a different complete prediction population")
        if not reference.y_true.equals(frame.y_true):
            raise ValueError(f"{name} has conflicting y_true values")
        features[f"p_{name}"] = frame.p_up.to_numpy(dtype=float)
    return features
