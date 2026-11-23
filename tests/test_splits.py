import pytest

from ensemble_learning.contracts import latest_label_confirmation, validate_frame
from ensemble_learning.splits import chronological_holdout


def test_session_boundaries(sample_frame):
    split = chronological_holdout(validate_frame(sample_frame), validation_sessions=3, test_sessions=3)
    assert len(split.train_sessions) == 6
    assert len(split.validation_sessions) == 3
    assert len(split.test_sessions) == 3
    assert latest_label_confirmation(split.train).max() < split.validation["decision_time"].min()
    assert latest_label_confirmation(split.validation).max() < split.test["decision_time"].min()


def test_purge_label_reaching_validation(sample_frame):
    source = sample_frame.copy()
    # Last training session label is confirmed during the first validation session.
    first_validation_time = source.loc[source["session_date"] == "2026-01-11", "decision_time"].min()
    train_last = source["session_date"] == "2026-01-10"
    for h in (5, 10, 15):
        source.loc[train_last, f"label_info_end_{h}m"] = first_validation_time
        source.loc[train_last, f"label_known_at_{h}m"] = first_validation_time
    split = chronological_holdout(validate_frame(source), validation_sessions=3, test_sessions=3)
    assert split.purged_train == 8
    assert latest_label_confirmation(split.train).max() < split.validation["decision_time"].min()


def test_impossible_split(sample_frame):
    with pytest.raises(ValueError, match="Not enough sessions"):
        chronological_holdout(validate_frame(sample_frame), validation_sessions=8, test_sessions=5)
