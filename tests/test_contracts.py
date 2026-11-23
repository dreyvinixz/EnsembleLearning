from datetime import timedelta

import pytest

from ensemble_learning.contracts import validate_frame


def test_valid_snapshot(sample_frame):
    validated = validate_frame(sample_frame)
    assert len(validated) == 96
    assert str(validated["decision_time"].dt.tz) == "UTC"


def test_rejects_future_features(sample_frame):
    row = sample_frame.copy()
    t = row.loc[0, "decision_time"]
    row.loc[0, "features_available_at"] = (
        __import__("pandas").Timestamp(t) + timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValueError, match="available after"):
        validate_frame(row)


def test_rejects_trinary_implicit_mapping(sample_frame):
    row = sample_frame.copy()
    row.loc[0, "y_5m"] = -1
    with pytest.raises(ValueError, match="binary labels"):
        validate_frame(row)


def test_rejects_naive_timezone(sample_frame):
    row = sample_frame.copy()
    row.loc[0, "decision_time"] = "2026-01-05T14:00:00"
    with pytest.raises(ValueError, match="explicit timezone"):
        validate_frame(row)


def test_accepts_iso8601_basic_offset(sample_frame):
    row = sample_frame.copy()
    row["features_available_at"] = row["features_available_at"].str.replace(
        "-03:00", "-0300", regex=False
    )
    validate_frame(row)


def test_rejects_duplicate_instrument_timestamp(sample_frame):
    row = __import__("pandas").concat([sample_frame, sample_frame.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frame(row)
