"""Causal fixed-clock sequence windows with explicit missing-candle masks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalWindows:
    """Index windows into the full context frame; -1 denotes an absent candle."""

    row_indices: np.ndarray
    anchors: pd.DataFrame
    observed_candles: np.ndarray


def summarize_eligibility(frame: pd.DataFrame, windows: CausalWindows,
                          target_columns: list[str]) -> dict[str, int]:
    """Count mutually exclusive reasons for excluding context rows."""
    labeled = frame[target_columns].notna().all(axis=1)
    before = pd.to_datetime(frame.decision_time, utc=True).dt.tz_convert(
        "America/Sao_Paulo"
    ).dt.time < time(10, 0)
    candidates = labeled & ~before
    return {"context_rows": len(frame), "missing_target": int((~labeled).sum()),
            "before_10": int((labeled & before).sum()),
            "insufficient_coverage": int(candidates.sum()) - len(windows.anchors),
            "accepted": len(windows.anchors)}


def build_causal_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    window_minutes: int = 60,
    min_observed: int = 48,
    first_decision: time = time(10, 0),
    market_timezone: str = "America/Sao_Paulo",
) -> CausalWindows:
    """Build windows ending at each labeled decision, never crossing a session.

    Rows in ``frame`` form the context grid and may have null targets. Only rows
    with all requested targets are returned as prediction anchors. Decision
    timestamps are candle close/availability times. At least ``min_observed``
    of the exact clock-minute slots must have candles. Missing slots stay -1 so
    the neural dataset can zero-fill features and append a presence mask.
    """
    required = {
        "instrument", "session_date", "decision_time", "features_available_at",
        *feature_columns, *target_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sequence frame is missing columns: {missing}")
    if not feature_columns or not target_columns:
        raise ValueError("At least one feature and one target are required")
    if window_minutes < 1 or not 1 <= min_observed <= window_minutes:
        raise ValueError("Invalid sequence window or minimum observed candle count")
    if frame.empty:
        raise ValueError("Sequence context is empty")
    if frame[["instrument", "session_date", "decision_time"]].isna().any().any():
        raise ValueError("Instrument, session and decision timestamps must be populated")

    decisions = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    available = pd.to_datetime(frame["features_available_at"], utc=True, errors="raise")
    if (available > decisions).any():
        raise ValueError("Sequence frame contains features unavailable at decision time")
    if frame.duplicated(["instrument", "session_date", "decision_time"]).any():
        raise ValueError("Duplicate instrument x session x decision timestamp")

    local_clock = decisions.dt.tz_convert(ZoneInfo(market_timezone)).dt.time
    labeled = frame[target_columns].notna().all(axis=1).to_numpy()
    candidates = np.flatnonzero(labeled & (local_clock >= first_decision).to_numpy())
    frame_index = frame.reset_index(drop=True)
    # Pandas may preserve a non-nanosecond datetime unit (for example us).
    # Normalize explicitly because slot arithmetic below is expressed in ns.
    decision_ns = decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    candidate_set = set(candidates.tolist())
    built: list[tuple[int, np.ndarray, int]] = []

    for _, group in frame_index.groupby(["instrument", "session_date"], sort=True, observed=True):
        positions = group.index.to_numpy(dtype=np.int64)
        times = decision_ns[positions]
        ordering = np.argsort(times, kind="stable")
        positions, times = positions[ordering], times[ordering]
        by_time = dict(zip(times.tolist(), positions.tolist(), strict=True))
        for anchor_position, anchor_ns in zip(positions, times, strict=True):
            if int(anchor_position) not in candidate_set:
                continue
            slot_times = anchor_ns - np.arange(window_minutes - 1, -1, -1, dtype=np.int64) * 60_000_000_000
            row_ids = np.fromiter(
                (by_time.get(int(slot_time), -1) for slot_time in slot_times),
                dtype=np.int64,
                count=window_minutes,
            )
            observed = int(np.count_nonzero(row_ids >= 0))
            if observed >= min_observed:
                built.append((int(anchor_position), row_ids, observed))

    built.sort(key=lambda item: item[0])
    if not built:
        return CausalWindows(
            row_indices=np.empty((0, window_minutes), dtype=np.int64),
            anchors=frame_index.iloc[0:0].copy(),
            observed_candles=np.empty(0, dtype=np.int16),
        )
    anchor_positions = np.fromiter((item[0] for item in built), dtype=np.int64)
    windows = np.stack([item[1] for item in built])
    counts = np.fromiter((item[2] for item in built), dtype=np.int16)
    return CausalWindows(windows, frame_index.iloc[anchor_positions].copy(), counts)
