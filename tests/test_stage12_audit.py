import pandas as pd
import pytest

from scripts.audit_stage11_results import paired_session_bootstrap


def test_session_bootstrap_is_paired_reproducible_and_counts_whole_sessions():
    rows = []
    for day, labels in (("2026-02-02", (0, 1)), ("2026-02-03", (1, 1)),
                        ("2026-02-04", (0, 0)), ("2026-02-05", (0, 1))):
        for minute, truth in enumerate(labels):
            rows.append({
                "instrument": "WIN", "session_date": day,
                "decision_time": pd.Timestamp(day, tz="America/Sao_Paulo")
                + pd.Timedelta(hours=13, minutes=minute),
                "scale": "5m", "fold": "fold01", "y_true": truth,
            })
    left = pd.DataFrame(rows)
    right = left.copy()
    left["p_up"] = [0.1, 0.9, 0.9, 0.6, 0.1, 0.2, 0.8, 0.4]
    right["p_up"] = [0.4, 0.6, 0.7, 0.55, 0.3, 0.4, 0.6, 0.45]

    first = paired_session_bootstrap(left, right, iterations=500, seed=7)
    second = paired_session_bootstrap(left, right, iterations=500, seed=7)

    assert first == second
    assert first["n"] == 8
    assert first["n_sessions"] == 4
    assert first["ci95_low"] <= first["ci95_high"]


def test_session_bootstrap_rejects_partial_or_label_mismatched_populations():
    frame = pd.DataFrame({
        "instrument": ["WIN"] * 4,
        "session_date": ["2026-02-02", "2026-02-02", "2026-02-03", "2026-02-03"],
        "decision_time": pd.date_range("2026-02-02T13:00:00-03:00", periods=4, freq="min"),
        "scale": "5m", "fold": "fold01", "y_true": [0, 1, 0, 1],
        "p_up": [0.2, 0.8, 0.3, 0.7],
    })
    with pytest.raises(ValueError, match="identical complete key populations"):
        paired_session_bootstrap(frame, frame.iloc[:-1])

    changed = frame.copy()
    changed.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="equal y_true"):
        paired_session_bootstrap(frame, changed)
