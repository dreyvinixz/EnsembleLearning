import pandas as pd
import pytest

from ensemble_learning.pairing import paired_metrics
from ensemble_learning.sequences import build_causal_windows, summarize_eligibility


def test_pairing_intersection_and_conflicting_labels():
    base = pd.DataFrame({"instrument": ["WIN", "WIN"], "session_date": ["2026-01-05"] * 2,
                         "decision_time": ["2026-01-05T13:00:00Z", "2026-01-05T13:01:00Z"],
                         "scale": ["5m"] * 2, "fold": ["fold01"] * 2,
                         "y_true": [0, 1], "p_up": [.2, .8]})
    paired, metrics = paired_metrics(base, base.iloc[1:])
    assert len(paired) == 1 and metrics.n.tolist() == [1, 1]
    changed = base.copy()
    changed.loc[1, "y_true"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        paired_metrics(base, changed)
    with pytest.raises(ValueError, match="duplicate"):
        paired_metrics(pd.concat([base, base]), base)
    naive = base.copy()
    naive["decision_time"] = ["2026-01-05T13:00:00", "2026-01-05T13:01:00"]
    with pytest.raises(ValueError, match="timezone offset"):
        paired_metrics(naive, base)


def test_unlabeled_context_counts_and_coverage_boundary():
    times = pd.date_range("2026-01-05T12:01:00Z", periods=60, freq="min")
    frame = pd.DataFrame({"instrument": "WIN", "session_date": "2026-01-05",
                          "decision_time": times, "features_available_at": times,
                          "x_price": 1., "y_5m": [None] * 59 + [1]})
    frame = frame.drop(index=range(12)).reset_index(drop=True)
    windows = build_causal_windows(frame, feature_columns=["x_price"], target_columns=["y_5m"])
    assert len(windows.anchors) == 1 and windows.observed_candles.tolist() == [48]
    assert summarize_eligibility(frame, windows, ["y_5m"])["accepted"] == 1
    frame = frame.drop(index=0).reset_index(drop=True)
    windows = build_causal_windows(frame, feature_columns=["x_price"], target_columns=["y_5m"])
    assert windows.anchors.empty
    assert summarize_eligibility(frame, windows, ["y_5m"])["insufficient_coverage"] == 1
