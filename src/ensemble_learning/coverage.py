"""Causal coverage check over completed one-minute bars from one session."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil


@dataclass(frozen=True)
class CoverageResult:
    eligible: bool
    observed: int
    required: int
    longest_gap: int
    reason: str


def recent_coverage(
    candle_starts: Iterable[datetime],
    *,
    decision_time: datetime,
    session_start: datetime,
    window_minutes: int = 60,
    minimum_fraction: float = 0.8,
    max_gap_minutes: int | None = None,
) -> CoverageResult:
    """Assess [decision-window, decision) using only completed bars.

    Inputs must all belong to one instrument and session. Decision and session
    start must be timezone-aware, minute-aligned, and use the same timezone.
    A full window must fit within the session; prior-session bars never count.
    max_gap_minutes is optional until a pre-test protocol specifies a value.
    """
    if window_minutes <= 0 or not 0 < minimum_fraction <= 1:
        raise ValueError('Invalid window or minimum coverage')
    if max_gap_minutes is not None and max_gap_minutes < 0:
        raise ValueError('max_gap_minutes must be nonnegative')
    for timestamp in (decision_time, session_start):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError('Timestamps must be timezone-aware')
        if timestamp.second or timestamp.microsecond:
            raise ValueError('Timestamps must be minute-aligned')
    if decision_time.utcoffset() != session_start.utcoffset():
        raise ValueError('Decision and session must use the same UTC offset')
    required = ceil(window_minutes * minimum_fraction)
    start = decision_time - timedelta(minutes=window_minutes)
    if start < session_start:
        return CoverageResult(False, 0, required, 0, 'insufficient_session_history')
    available = set()
    for timestamp in candle_starts:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError('Candle timestamps must be timezone-aware')
        if timestamp.second or timestamp.microsecond:
            raise ValueError('Candle timestamps must be minute-aligned')
        if timestamp.utcoffset() != decision_time.utcoffset():
            raise ValueError('Candle timestamps must use the same UTC offset')
        if start <= timestamp and timestamp + timedelta(minutes=1) <= decision_time:
            available.add(timestamp)
    count = len(available)
    run = longest = 0
    for minute in range(window_minutes):
        if start + timedelta(minutes=minute) in available:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    if count < required:
        return CoverageResult(False, count, required, longest, 'insufficient_coverage')
    if max_gap_minutes is not None and longest > max_gap_minutes:
        return CoverageResult(False, count, required, longest, 'gap_too_long')
    return CoverageResult(True, count, required, longest, 'eligible')
