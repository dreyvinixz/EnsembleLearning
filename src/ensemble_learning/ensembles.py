"""Leakage-safe probability aggregation for already recorded predictions."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ensemble_learning.contracts import timestamp_column
from ensemble_learning.pairing import KEYS


def _normalize_predictions(frame: pd.DataFrame, member: str) -> pd.DataFrame:
    required = set(KEYS + ["y_true", "p_up"])
    if missing := required - set(frame.columns):
        raise ValueError(f"{member} missing columns: {sorted(missing)}")
    result = frame[KEYS + ["y_true", "p_up"]].copy()
    if result.isna().any().any():
        raise ValueError(f"{member} contains null keys, labels or probabilities")
    result["decision_time"] = timestamp_column(result, "decision_time")
    result["session_date"] = result.session_date.astype(str)
    if result.duplicated(KEYS).any():
        raise ValueError(f"{member} contains duplicate prediction keys")
    if not result.y_true.isin([0, 1]).all():
        raise ValueError(f"{member} contains invalid B0 classes")
    if not result.p_up.between(0, 1).all():
        raise ValueError(f"{member} contains probabilities outside [0, 1]")
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def mean_probability_ensemble(
    members: Mapping[str, pd.DataFrame], *, ensemble_name: str,
) -> pd.DataFrame:
    """Average member probabilities after exact key and class equality checks."""
    if not members:
        raise ValueError("At least one prediction member is required")
    normalized = [(name, _normalize_predictions(frame, name))
                  for name, frame in members.items()]
    _, reference = normalized[0]
    probabilities = []
    for name, frame in normalized:
        if not reference[KEYS].equals(frame[KEYS]):
            raise ValueError(f"{name} has a different complete prediction population")
        if not reference.y_true.equals(frame.y_true):
            raise ValueError(f"{name} has conflicting y_true values")
        probabilities.append(frame.p_up.to_numpy(dtype=np.float64))
    probability = np.mean(np.stack(probabilities, axis=0), axis=0)
    result = reference[KEYS + ["y_true"]].copy()
    result["p_up"] = probability
    result["p_down"] = 1.0 - probability
    result["y_pred"] = (probability >= 0.5).astype(np.int8)
    result["ensemble"] = ensemble_name
    result["member_count"] = len(normalized)
    result["members"] = ",".join(name for name, _ in normalized)
    return result

