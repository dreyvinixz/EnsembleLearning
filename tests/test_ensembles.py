import pandas as pd
import pytest

from ensemble_learning.ensembles import mean_probability_ensemble


def _predictions():
    return pd.DataFrame({
        "instrument": ["WIN", "WIN"],
        "session_date": ["2026-02-02", "2026-02-02"],
        "decision_time": pd.to_datetime([
            "2026-02-02T13:00:00Z", "2026-02-02T13:01:00Z",
        ]),
        "scale": ["5m", "5m"], "fold": ["fold01", "fold01"],
        "y_true": [0, 1], "p_up": [0.2, 0.8],
    })


def test_seed_ensemble_averages_probabilities_on_identical_keys():
    first = _predictions()
    second = first.copy()
    second["p_up"] = [0.4, 0.6]
    result = mean_probability_ensemble(
        {"seed42": first, "seed43": second}, ensemble_name="seed_ensemble_gru_mtl"
    )
    assert result.p_up.tolist() == pytest.approx([0.3, 0.7])
    assert result.p_down.tolist() == pytest.approx([0.7, 0.3])
    assert result.y_pred.tolist() == [0, 1]
    assert result.member_count.eq(2).all()


def test_ensemble_rejects_partial_population_and_conflicting_labels():
    frame = _predictions()
    with pytest.raises(ValueError, match="different complete prediction population"):
        mean_probability_ensemble(
            {"full": frame, "partial": frame.iloc[:1]}, ensemble_name="bad"
        )
    changed = frame.copy()
    changed.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="conflicting y_true"):
        mean_probability_ensemble({"base": frame, "changed": changed}, ensemble_name="bad")


def test_ensemble_rejects_duplicate_keys_and_invalid_probabilities():
    frame = _predictions()
    with pytest.raises(ValueError, match="duplicate"):
        mean_probability_ensemble(
            {"duplicate": pd.concat([frame, frame])}, ensemble_name="bad"
        )
    invalid = frame.copy()
    invalid.loc[0, "p_up"] = 1.1
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        mean_probability_ensemble({"invalid": invalid}, ensemble_name="bad")
