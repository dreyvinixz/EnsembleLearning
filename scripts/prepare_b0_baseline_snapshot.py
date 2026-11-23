"""Prepare a causal, validation-only CSV snapshot from the approved WIN Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/win_2026_v1"
DEFAULT_OUT = ROOT / "data/processed/win_2026_baseline_input_v1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_snapshot(
    market: pd.DataFrame, labels: pd.DataFrame, *, include_context_bars: bool = False,
) -> pd.DataFrame:
    keys = ["instrument", "asset_id", "session_date", "candle_start_local"]
    frame = market.merge(labels, on=keys, how="inner", validate="one_to_one")
    frame["candle_start_local"] = pd.to_datetime(frame["candle_start_local"])
    frame = frame.sort_values(keys).reset_index(drop=True)
    group_keys = ["instrument", "session_date"]
    grouped = frame.groupby(group_keys, sort=False, observed=True)

    # Every feature uses the current bar (available when that one-minute candle
    # closes) and/or earlier bars in the same instrument-session only.
    previous_close = grouped["close"].shift(1)
    one_bar_return = frame["close"].div(previous_close).sub(1)
    features: dict[str, pd.Series] = {
        "x_return_1bar": one_bar_return,
        "x_range_fraction": frame["high"].sub(frame["low"]).div(frame["close"]),
        "x_body_fraction": frame["close"].sub(frame["open"]).div(frame["open"]),
        "x_close_location": frame["close"].sub(frame["low"]).div(
            frame["high"].sub(frame["low"]).replace(0, np.nan)
        ),
        "x_log_volume": np.log1p(frame["volume"].clip(lower=0)),
        "x_log_business": np.log1p(frame["business"].clip(lower=0)),
        "x_average_close_gap": frame["average"].div(frame["close"]).sub(1),
        "x_log_amount": np.log1p(frame["amount_stock"].clip(lower=0)),
    }
    for lag in (5, 15, 30, 60):
        lagged = grouped["close"].shift(lag)
        features[f"x_return_{lag}bars"] = frame["close"].div(lagged).sub(1)
    for window in (5, 15, 30, 60):
        features[f"x_return_volatility_{window}bars"] = one_bar_return.groupby(
            [frame[k] for k in group_keys], sort=False, observed=True
        ).transform(
            lambda values, rolling_window=window: values.rolling(
                rolling_window, min_periods=max(3, rolling_window // 2)
            ).std()
        )
        rolling_mean = grouped["close"].transform(
            lambda values, rolling_window=window: values.rolling(
                rolling_window, min_periods=max(3, rolling_window // 2)
            ).mean()
        )
        features[f"x_close_vs_sma_{window}bars"] = frame["close"].div(rolling_mean).sub(1)
    log_volume = features["x_log_volume"]
    volume_mean = log_volume.groupby(
        [frame[k] for k in group_keys], sort=False, observed=True
    ).transform(lambda values: values.rolling(30, min_periods=10).mean())
    volume_std = log_volume.groupby(
        [frame[k] for k in group_keys], sort=False, observed=True
    ).transform(lambda values: values.rolling(30, min_periods=10).std())
    features["x_log_volume_zscore_30bars"] = log_volume.sub(volume_mean).div(volume_std)
    for name, values in features.items():
        frame[name] = values.replace([np.inf, -np.inf], np.nan)

    # At decision time = candle start + 1 minute, count source candles in
    # [decision-60m, decision), grouped by contract and session.
    eligible = np.zeros(len(frame), dtype=bool)
    for indexes in frame.groupby(group_keys, sort=False, observed=True).groups.values():
        positions = np.asarray(list(indexes), dtype=int)
        starts = frame.loc[positions, "candle_start_local"].to_numpy(dtype="datetime64[ns]").astype("int64")
        left = np.searchsorted(starts, starts - 59 * 60 * 1_000_000_000, side="left")
        # Current candle is included: its start is one minute before decision.
        counts = np.arange(len(starts)) + 1 - left
        full_window = frame.loc[positions, "candle_start_local"].dt.hour.ge(10).to_numpy()
        eligible[positions] = (counts >= 48) & full_window
    frame["_coverage_80"] = eligible

    frame["decision_time"] = frame["candle_start_local"] + pd.Timedelta(minutes=1)
    frame["features_available_at"] = frame["decision_time"]
    session_start = pd.to_datetime(frame["session_date"]) + pd.Timedelta(hours=9)
    conservative_end = pd.to_datetime(frame["session_date"]) + pd.Timedelta(hours=18, minutes=1)
    simulated_known = pd.to_datetime(frame["session_date"]) + pd.Timedelta(hours=18, minutes=2)
    frame["label_info_start"] = session_start
    frame["label_info_end"] = conservative_end
    frame["label_known_at"] = simulated_known

    if not include_context_bars:
        frame = frame.loc[frame["_coverage_80"]].copy()
        frame = frame.dropna(subset=["y_b0_5m", "y_b0_10m", "y_b0_15m"])
    frame = frame.rename(
        columns={"y_b0_5m": "y_5m", "y_b0_10m": "y_10m", "y_b0_15m": "y_15m"}
    )
    feature_cols = [column for column in frame if column.startswith("x_")]
    snapshot = frame[
        ["instrument", "session_date", "decision_time", "features_available_at"]
        + feature_cols
        + ["y_5m", "y_10m", "y_15m"]
    ].copy()
    for horizon in (5, 10, 15):
        snapshot[f"label_info_start_{horizon}m"] = frame["label_info_start"]
        snapshot[f"label_info_end_{horizon}m"] = frame["label_info_end"]
        snapshot[f"label_known_at_{horizon}m"] = frame["label_known_at"]
    snapshot = snapshot.sort_values(["decision_time", "instrument"]).reset_index(drop=True)
    time_columns = [column for column in snapshot if column in {
        "decision_time", "features_available_at",
        *[f"label_info_start_{h}m" for h in (5, 10, 15)],
        *[f"label_info_end_{h}m" for h in (5, 10, 15)],
        *[f"label_known_at_{h}m" for h in (5, 10, 15)],
    }]
    for column in time_columns:
        snapshot[column] = pd.to_datetime(snapshot[column]).dt.tz_localize("America/Sao_Paulo")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest_path}")
    market_path = BASE / "market_candles.parquet"
    labels_path = BASE / "b0_targets_retrospective.parquet"
    market = pd.read_parquet(market_path)
    labels = pd.read_parquet(labels_path)
    snapshot = make_snapshot(market, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    for column in snapshot.select_dtypes(include=["datetimetz"]).columns:
        snapshot[column] = snapshot[column].map(lambda timestamp: timestamp.isoformat())
    snapshot.to_csv(output, index=False)
    summary = {
        "snapshot": output.name,
        "snapshot_sha256": sha256(output),
        "market_source_sha256": sha256(market_path),
        "label_source_sha256": sha256(labels_path),
        "rows": len(snapshot),
        "sessions": int(snapshot.session_date.nunique()),
        "instruments": sorted(snapshot.instrument.unique().tolist()),
        "features": [column for column in snapshot if column.startswith("x_")],
        "coverage_gate": "At prediction time, at least 48 observed closed candles in the preceding 60 minutes; same contract and session.",
        "decision_time": "One minute after candle_start_local; current candle is closed and available.",
        "label_classes": {f"y_{h}m": sorted(snapshot[f"y_{h}m"].unique().astype(int).tolist()) for h in (5, 10, 15)},
        "label_population": "Common complete cases: all three retrospective B0 labels non-null.",
        "label_information_interval": "Conservative whole session, 09:00 through 18:01 America/Sao_Paulo.",
        "simulated_label_release": "18:02 America/Sao_Paulo, after the latest possible candle close; processing latency/finalization is an explicit simulation assumption, not historically observed.",
        "feature_policy": "Causal lag/rolling OHLCV features reset for each contract-session; no future bars, labels, SIDEWAYS, trend_mean, or segment endpoints.",
        "test_policy": "January-April train, May validation, June reserved as final test.",
        "source_manifest": str((BASE / "manifest.json").relative_to(ROOT)),
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
