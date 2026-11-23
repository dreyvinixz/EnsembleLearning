"""Chronological, session-based holdout with explicit target-availability purging."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ensemble_learning.contracts import latest_label_confirmation


@dataclass(frozen=True)
class TemporalHoldout:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged_train: int
    purged_validation: int
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def chronological_holdout(
    df: pd.DataFrame, *, validation_sessions: int, test_sessions: int
) -> TemporalHoldout:
    """Split whole exchange sessions across every instrument.

    Requires already validated data. Purges any development row whose
    latest label confirmation reaches into the next split. The test
    block is never used for training, preprocessing, or model selection.
    """
    if validation_sessions < 1 or test_sessions < 1:
        raise ValueError("Both validation_sessions and test_sessions must be positive")
    ordered = sorted(df["session_date"].astype(str).unique())
    if len(ordered) <= validation_sessions + test_sessions:
        raise ValueError("Not enough sessions for train/validation/test")
    cut_train = len(ordered) - validation_sessions - test_sessions
    train_sessions = tuple(ordered[:cut_train])
    val_sessions = tuple(ordered[cut_train:-test_sessions])
    test_sessions_seq = tuple(ordered[-test_sessions:])
    train_raw = df.loc[df["session_date"].isin(train_sessions)].copy()
    val_raw = df.loc[df["session_date"].isin(val_sessions)].copy()
    test = df.loc[df["session_date"].isin(test_sessions_seq)].copy()
    if train_raw.empty or val_raw.empty or test.empty:
        raise ValueError("Empty split after session grouping")
    validation_start = val_raw["decision_time"].min()
    test_start = test["decision_time"].min()
    if not train_raw["decision_time"].max() < validation_start < test_start:
        raise ValueError("Chronological session boundaries overlap: inspect session dates")
    train = train_raw.loc[latest_label_confirmation(train_raw) < validation_start].copy()
    validation = val_raw.loc[latest_label_confirmation(val_raw) < test_start].copy()
    if train.empty or validation.empty:
        raise ValueError("Purging removed an entire development split")
    return TemporalHoldout(
        train=train,
        validation=validation,
        test=test,
        purged_train=len(train_raw) - len(train),
        purged_validation=len(val_raw) - len(validation),
        train_sessions=train_sessions,
        validation_sessions=val_sessions,
        test_sessions=test_sessions_seq,
    )


def dated_holdout(df: pd.DataFrame) -> TemporalHoldout:
    """2026 fixed calendar partitions, with whole-session availability purge."""
    dates = df["session_date"].astype(str)
    train_raw = df.loc[dates.between("2026-01-01", "2026-04-30")].copy()
    val_raw = df.loc[dates.between("2026-05-01", "2026-05-31")].copy()
    test = df.loc[dates.between("2026-06-01", "2026-06-30")].copy()
    if len(train_raw) + len(val_raw) + len(test) != len(df):
        raise ValueError("Rows outside the declared January–June 2026 boundaries")
    if train_raw.empty or val_raw.empty:
        raise ValueError("Training and validation must both contain data")
    # Boundary is the start of the next calendar month in B3 time, independent
    # of the dates present in the dataset or whether the test labels are loaded.
    validation_start = pd.Timestamp("2026-05-01T00:00:00-03:00")
    test_start = pd.Timestamp("2026-06-01T00:00:00-03:00")
    train = train_raw.loc[latest_label_confirmation(train_raw) < validation_start].copy()
    validation = val_raw.loc[latest_label_confirmation(val_raw) < test_start].copy()
    if train.empty or validation.empty:
        raise ValueError("Purging removed an entire development split")
    return TemporalHoldout(
        train, validation, test, len(train_raw) - len(train), len(val_raw) - len(validation),
        tuple(sorted(train_raw.session_date.unique())),
        tuple(sorted(val_raw.session_date.unique())),
        tuple(sorted(test.session_date.unique())),
    )
