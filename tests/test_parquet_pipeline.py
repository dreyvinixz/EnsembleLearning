"""Synthetic integration coverage; no private data or real June labels."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ensemble_learning.baselines import run_independent_logreg
from ensemble_learning.contracts import validate_frame
from ensemble_learning.splits import dated_holdout


def test_parquet_to_baseline_and_session_boundary(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_b0_baseline_snapshot.py"
    spec = importlib.util.spec_from_file_location("snapshot_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    rows = []
    for day in ("2026-01-02", "2026-04-30", "2026-05-04", "2026-06-01"):
        for index, stamp in enumerate(pd.date_range(day + " 16:00", periods=120, freq="min")):
            price = 100 + index * 0.1
            rows.append({"instrument": "WIN", "asset_id": 1, "session_date": day,
                         "candle_start_local": stamp, "open": price, "high": price + 1,
                         "low": price - 1, "close": price + 0.2, "average": price,
                         "volume": index + 1, "business": index + 1, "amount_stock": index + 2,
                         "y_b0_5m": index % 2, "y_b0_10m": index % 2,
                         "y_b0_15m": index % 2})
    source = pd.DataFrame(rows)
    labels = ["y_b0_5m", "y_b0_10m", "y_b0_15m"]
    market = source.drop(columns=labels)
    label_data = source[["instrument", "asset_id", "session_date", "candle_start_local"] + labels]
    market.to_parquet(tmp_path / "market_candles.parquet")
    label_data.to_parquet(tmp_path / "b0_targets_retrospective.parquet")
    monkeypatch.setattr(builder, "BASE", tmp_path)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["prepare", "--output", str(tmp_path / "snapshot.csv")])
    builder.main()
    before = (tmp_path / "snapshot.csv").read_bytes()
    with pytest.raises(FileExistsError):
        builder.main()
    assert (tmp_path / "snapshot.csv").read_bytes() == before
    raw = pd.read_csv(tmp_path / "snapshot.csv")
    assert raw.decision_time.str.endswith("-03:00").all()
    assert raw.decision_time.str.contains("T18:00:00-03:00").sum() == 4
    validated = validate_frame(raw)
    split = dated_holdout(validated)
    assert split.train_sessions == ("2026-01-02", "2026-04-30")
    assert split.validation_sessions == ("2026-05-04",)
    assert split.test_sessions == ("2026-06-01",)
    results = run_independent_logreg(split)
    assert all("test" not in result for result in results["horizons"].values())
    with pytest.raises(ValueError, match="outside"):
        dated_holdout(pd.concat([validated, validated.assign(session_date="2026-07-01")]))
