import importlib.util
from pathlib import Path

import pandas as pd

from ensemble_learning.contracts import feature_columns, validate_sequence_frame
from ensemble_learning.sequences import build_causal_windows


def _snapshot_builder():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_b0_baseline_snapshot.py"
    spec = importlib.util.spec_from_file_location("sequence_snapshot_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_snapshot_retains_candles_before_first_prediction_anchor():
    stamps = pd.date_range("2026-01-05 09:00", periods=90, freq="min")
    rows = []
    for index, stamp in enumerate(stamps):
        close = 100 + index / 10
        row = {
            "instrument": "WIN", "asset_id": 1, "session_date": "2026-01-05",
            "candle_start_local": stamp, "open": close, "high": close + 1,
            "low": close - 1, "close": close, "average": close,
            "volume": index + 1, "business": index + 1, "amount_stock": index + 1,
        }
        row.update({f"y_b0_{h}m": index % 2 for h in (5, 10, 15)})
        rows.append(row)
    source = pd.DataFrame(rows)
    market = source.drop(columns=[f"y_b0_{h}m" for h in (5, 10, 15)])
    labels = source[["instrument", "asset_id", "session_date", "candle_start_local",
                     *[f"y_b0_{h}m" for h in (5, 10, 15)]]]

    context = _snapshot_builder().make_snapshot(market, labels, include_context_bars=True)
    validated = validate_sequence_frame(context)
    windows = build_causal_windows(
        validated, feature_columns=feature_columns(validated),
        target_columns=["y_5m", "y_10m", "y_15m"],
    )

    assert len(validated) == 90
    assert validated.decision_time.min() == pd.Timestamp("2026-01-05T12:01:00Z")
    assert windows.anchors.iloc[0].decision_time == pd.Timestamp("2026-01-05T13:00:00Z")
    assert windows.observed_candles[0] == 60
