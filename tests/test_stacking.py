import pandas as pd
import pytest

from ensemble_learning.stacking import chronological_fold_schedule, stacking_feature_frame


def _frame():
    return pd.DataFrame({
        "instrument": ["WIN", "WIN"],
        "session_date": ["2026-02-02", "2026-02-02"],
        "decision_time": pd.to_datetime([
            "2026-02-02T13:00:00Z", "2026-02-02T13:01:00Z",
        ]),
        "scale": ["5m", "5m"], "fold": ["fold01", "fold01"],
        "y_true": [0, 1], "p_up": [0.2, 0.8],
    })


def test_meta_training_uses_only_strictly_earlier_oof_folds():
    folds = ("fold01", "fold02", "fold03", "fold04")
    schedule = chronological_fold_schedule(folds, folds)
    assert schedule == [
        {"prediction_fold": "fold02", "meta_train_folds": ["fold01"]},
        {"prediction_fold": "fold03", "meta_train_folds": ["fold01", "fold02"]},
        {"prediction_fold": "fold04", "meta_train_folds": ["fold01", "fold02", "fold03"]},
    ]


def test_stacking_features_require_exact_keys_and_classes():
    first = _frame()
    second = first.copy()
    second["p_up"] = [0.3, 0.7]
    features = stacking_feature_frame({"gru": first, "lstm": second})
    assert features.p_gru.tolist() == [0.2, 0.8]
    assert features.p_lstm.tolist() == [0.3, 0.7]
    with pytest.raises(ValueError, match="different complete prediction population"):
        stacking_feature_frame({"full": first, "partial": first.iloc[:1]})
    changed = second.copy()
    changed.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="conflicting y_true"):
        stacking_feature_frame({"base": first, "changed": changed})
