"""Strict provisional CSV data contract; no inferred timestamps or label windows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ensemble_learning import HORIZONS

IDENTIFIERS = ("instrument", "session_date", "decision_time", "features_available_at")
TIMEZONE_RE = re.compile(r"(Z|[+-][0-9]{2}:?[0-9]{2})$", re.IGNORECASE)


def timestamp_column(df: pd.DataFrame, name: str) -> pd.Series:
    values = df[name]
    if values.isna().any():
        raise ValueError(f"{name}: null timestamps are not allowed")
    # Explicit offsets prevent accidental timezone assumptions about B3 sessions.
    if not values.map(lambda x: bool(TIMEZONE_RE.search(str(x)))).all():
        raise ValueError(f"{name}: require ISO-8601 timestamps with explicit timezone offset")
    return pd.to_datetime(values, utc=True, errors="raise")


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns = sorted(c for c in df.columns if c.startswith("x_"))
    if not columns:
        raise ValueError("At least one numeric feature prefixed with x_ is required")
    return columns


def validate_frame(frame: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Validate one approved binary-label snapshot; does not construct any B0 labels.

    The accepted class domain {0, 1} is provisional. If the source uses
    {-1, 0, +1}, define a separate, versioned mapping before training.
    """
    df = frame.copy(deep=True)
    required = set(IDENTIFIERS)
    for h in horizons:
        required.update(
            (f"y_{h}m", f"label_info_start_{h}m",
             f"label_info_end_{h}m", f"label_known_at_{h}m")
        )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input snapshot is empty")
    if df["instrument"].isna().any() or df["session_date"].isna().any():
        raise ValueError("Instrument and session_date must be populated")
    if not df["session_date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("session_date must use YYYY-MM-DD (exchange-local session)")
    xcols = feature_columns(df)
    for col in xcols:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"{col}: infinite values are forbidden")
        if df[col].isna().all():
            raise ValueError(f"{col}: entirely missing feature")
    for col in ("decision_time", "features_available_at"):
        df[col] = timestamp_column(df, col)
    if (df["features_available_at"] > df["decision_time"]).any():
        raise ValueError("A feature is marked available after its decision_time")
    for h in horizons:
        label = f"y_{h}m"
        if df[label].isna().any() or not df[label].isin([0, 1]).all():
            raise ValueError(f"{label}: explicitly confirmed binary labels 0/1 required")
        df[label] = df[label].astype(int)
        start = f"label_info_start_{h}m"
        end = f"label_info_end_{h}m"
        known = f"label_known_at_{h}m"
        for col in (start, end, known):
            df[col] = timestamp_column(df, col)
        if ((df[start] > df[end]) | (df[end] > df[known])).any():
            raise ValueError(f"{h}m: invalid label information/confirmation interval")
    if df.duplicated(["instrument", "decision_time"]).any():
        raise ValueError("Duplicate instrument x decision_time observations")
    # Never infer session from UTC calendar date: B3 session_date is authoritative.
    return df.sort_values(["decision_time", "instrument"]).reset_index(drop=True)


def validate_sequence_frame(frame: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Validate a full candle context grid whose retrospective targets may be null.

    Null targets are allowed only on context candles; any populated B0 target
    must remain binary and carry explicit, ordered information/availability
    timestamps. Target and label metadata columns are never model features.
    """
    df = frame.copy(deep=True)
    required = set(IDENTIFIERS)
    for horizon in horizons:
        required.update((
            f"y_{horizon}m", f"label_info_start_{horizon}m",
            f"label_info_end_{horizon}m", f"label_known_at_{horizon}m",
        ))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing sequence context columns: {missing}")
    if df.empty:
        raise ValueError("Input sequence context is empty")
    if df[["instrument", "session_date"]].isna().any().any():
        raise ValueError("Instrument and session_date must be populated")
    if not df["session_date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("session_date must use YYYY-MM-DD")

    features = feature_columns(df)
    for column in features:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if np.isinf(df[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{column}: infinite values are forbidden")
        if df[column].isna().all():
            raise ValueError(f"{column}: entirely missing feature")
    for column in ("decision_time", "features_available_at"):
        df[column] = timestamp_column(df, column)
    if (df["features_available_at"] > df["decision_time"]).any():
        raise ValueError("A feature is marked available after its decision_time")

    for horizon in horizons:
        target = f"y_{horizon}m"
        populated = df[target].notna()
        if not df.loc[populated, target].isin([0, 1]).all():
            raise ValueError(f"{target}: populated B0 targets must be binary 0/1")
        df.loc[populated, target] = df.loc[populated, target].astype(int)
        columns = (
            f"label_info_start_{horizon}m", f"label_info_end_{horizon}m",
            f"label_known_at_{horizon}m",
        )
        for column in columns:
            nonnull = df[column].notna()
            df.loc[nonnull, column] = timestamp_column(df.loc[nonnull], column)
        has_metadata = df[list(columns)].notna().all(axis=1)
        if (populated & ~has_metadata).any():
            raise ValueError(f"{target}: populated targets require label interval and availability")
        valid = populated & has_metadata
        start, end, known = (df.loc[valid, column] for column in columns)
        if ((start > end) | (end > known)).any():
            raise ValueError(f"{horizon}m: invalid label information/confirmation interval")

    if df.duplicated(["instrument", "session_date", "decision_time"]).any():
        raise ValueError("Duplicate instrument x session x decision observations")
    return df.sort_values(["decision_time", "instrument"]).reset_index(drop=True)


def latest_label_confirmation(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.Series:
    """Maximum target availability across all tasks at each decision."""
    return df[[f"label_known_at_{h}m" for h in horizons]].max(axis=1)
