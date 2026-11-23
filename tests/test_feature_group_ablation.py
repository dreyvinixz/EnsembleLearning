from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "run_feature_group_ablation", ROOT / "scripts/run_feature_group_ablation.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_taxonomy_is_disjoint_exhaustive_and_matches_21_current_features():
    names = [
        "x_return_1bar", "x_return_5bars", "x_return_15bars", "x_return_30bars", "x_return_60bars",
        "x_range_fraction", "x_body_fraction", "x_close_location", "x_average_close_gap",
        "x_return_volatility_5bars", "x_return_volatility_15bars",
        "x_return_volatility_30bars", "x_return_volatility_60bars",
        "x_close_vs_sma_5bars", "x_close_vs_sma_15bars",
        "x_close_vs_sma_30bars", "x_close_vs_sma_60bars",
        "x_log_volume", "x_log_business", "x_log_amount", "x_log_volume_zscore_30bars",
    ]
    groups = MODULE.assign_feature_groups(names)
    assert {name: len(features) for name, features in groups.items()} == {
        "returns": 5, "candles_price_shape": 4, "volatility": 4,
        "trend_vs_sma": 4, "volume_activity": 4,
    }
    assert sorted(sum(groups.values(), [])) == sorted(names)


def test_feature_taxonomy_rejects_unclassified_future_features():
    names = [name for values in {
        "x_return_1bar": ["x_return_1bar"],
        "x_range_fraction": ["x_range_fraction"],
        "x_return_volatility_5bars": ["x_return_volatility_5bars"],
        "x_close_vs_sma_5bars": ["x_close_vs_sma_5bars"],
        "x_log_volume": ["x_log_volume"],
    }.values() for name in values]
    with pytest.raises(ValueError, match="exhaustive and disjoint"):
        MODULE.assign_feature_groups(names + ["x_new_feature"])


def test_feature_variants_preserve_common_causal_anchor_eligibility():
    columns = [
        "x_return_1bar", "x_return_5bars", "x_return_15bars", "x_return_30bars", "x_return_60bars",
        "x_range_fraction", "x_body_fraction", "x_close_location", "x_average_close_gap",
        "x_return_volatility_5bars", "x_return_volatility_15bars",
        "x_return_volatility_30bars", "x_return_volatility_60bars",
        "x_close_vs_sma_5bars", "x_close_vs_sma_15bars",
        "x_close_vs_sma_30bars", "x_close_vs_sma_60bars",
        "x_log_volume", "x_log_business", "x_log_amount", "x_log_volume_zscore_30bars",
    ]
    group_defs = MODULE.assign_feature_groups(columns)
    times = pd.date_range("2026-01-05 09:02", periods=62, freq="min", tz="America/Sao_Paulo")
    frame = pd.DataFrame({
        "instrument": "WING26", "session_date": "2026-01-05",
        "decision_time": times, "features_available_at": times,
        **{column: range(62) for column in columns},
        "y_5m": [0] * 62, "y_10m": [1] * 62, "y_15m": [0] * 62,
    })
    accepted = MODULE.verify_common_eligibility(frame, columns, group_defs)
    assert accepted == 4
