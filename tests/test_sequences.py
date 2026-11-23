from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ensemble_learning.sequences import build_causal_windows

TZ = ZoneInfo("America/Sao_Paulo")


def _frame(session: str, start: str, minutes: int, missing: set[int] | None = None):
    base = datetime.fromisoformat(f"{session}T{start}:00").replace(tzinfo=TZ)
    missing = missing or set()
    rows = []
    for minute in range(minutes):
        if minute in missing:
            continue
        decision = base + timedelta(minutes=minute)
        rows.append({
            "instrument": "SYNTH", "session_date": session,
            "decision_time": decision, "features_available_at": decision,
            "x_signal": float(minute), "y_5m": minute % 2,
        })
    return pd.DataFrame(rows)


def _build(frame):
    return build_causal_windows(
        frame, feature_columns=["x_signal"], target_columns=["y_5m"],
    )


def test_window_at_10am_uses_exact_previous_60_closed_minutes():
    frame = _frame("2026-01-05", "09:01", 61)
    windows = _build(frame)
    first = windows.anchors.iloc[0]
    assert first.decision_time == pd.Timestamp("2026-01-05T10:00:00-03:00")
    first_window = windows.row_indices[0]
    assert len(first_window) == 60
    assert first_window[0] == 0
    assert frame.iloc[first_window[-1]].decision_time == first.decision_time
    assert windows.observed_candles[0] == 60


def test_window_slots_normalize_microsecond_timestamp_units():
    frame = _frame("2026-01-05", "09:01", 61)
    for column in ("decision_time", "features_available_at"):
        frame[column] = pd.to_datetime(frame[column]).dt.as_unit("us")
    windows = _build(frame)
    assert len(windows.anchors) == 2
    assert windows.observed_candles.tolist() == [60, 60]


def test_missing_minutes_are_not_filled_and_80_percent_gate_is_exact():
    frame = _frame("2026-01-05", "09:01", 80, set(range(13)))
    windows = _build(frame)
    first = windows.anchors.iloc[0]
    assert first.decision_time == pd.Timestamp("2026-01-05T10:01:00-03:00")
    assert windows.observed_candles[0] == 48
    assert (windows.row_indices[0] == -1).sum() == 12


def test_history_never_crosses_a_session_boundary():
    yesterday = _frame("2026-01-05", "09:01", 60)
    today = _frame("2026-01-06", "09:41", 20)
    windows = _build(pd.concat([yesterday, today], ignore_index=True))
    today_anchors = windows.anchors.loc[windows.anchors.session_date == "2026-01-06"]
    assert today_anchors.empty


def test_windows_never_reference_rows_after_the_anchor():
    frame = _frame("2026-01-05", "09:01", 80)
    windows = _build(frame)
    for row_ids, anchor in zip(windows.row_indices, windows.anchors.itertuples(), strict=True):
        valid_rows = row_ids[row_ids >= 0]
        assert frame.iloc[valid_rows].decision_time.max() <= anchor.decision_time
