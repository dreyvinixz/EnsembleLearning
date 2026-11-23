"""Temporal session-block bootstrap sampling for bagged estimators."""
from __future__ import annotations

import numpy as np


def sample_session_blocks(
    session_dates: list[str], *, block_size: int, seed: int,
) -> list[str]:
    """Sample contiguous blocks of observed sessions with replacement.

    Returns exactly ``len(session_dates)`` session draws. Blocks follow the
    input's chronological order; repeated sessions intentionally repeat their
    training rows and never alter prediction keys.
    """
    if not session_dates:
        raise ValueError("At least one training session is required")
    if len(set(session_dates)) != len(session_dates):
        raise ValueError("session_dates must be unique and chronologically deduplicated")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    size = min(block_size, len(session_dates))
    rng = np.random.default_rng(seed)
    starts = np.arange(len(session_dates) - size + 1)
    sampled: list[str] = []
    while len(sampled) < len(session_dates):
        start = int(rng.choice(starts))
        sampled.extend(session_dates[start:start + size])
    return sampled[:len(session_dates)]
