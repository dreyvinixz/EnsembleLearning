from datetime import datetime, timedelta, timezone

import pytest

from ensemble_learning.coverage import recent_coverage

TZ = timezone(timedelta(hours=-3))
START = datetime(2026, 1, 5, 9, tzinfo=TZ)
DECISION = START + timedelta(hours=1)


def test_completed_minutes_only_and_future_never_changes_result():
    bars = [START + timedelta(minutes=i) for i in range(48)]
    result = recent_coverage(bars, decision_time=DECISION, session_start=START)
    assert result.eligible and result.observed == 48 and result.longest_gap == 12
    assert result == recent_coverage(
        bars + [DECISION, DECISION + timedelta(minutes=1)],
        decision_time=DECISION, session_start=START,
    )
    assert not recent_coverage(bars[:-1], decision_time=DECISION, session_start=START).eligible


def test_history_stays_in_session_and_duplicates_do_not_count():
    bars = [START - timedelta(minutes=1)] * 20 + [START] * 60
    assert recent_coverage(bars, decision_time=DECISION, session_start=START).observed == 1
    assert recent_coverage(bars, decision_time=START + timedelta(minutes=59),
                           session_start=START).reason == 'insufficient_session_history'


def test_gap_limit_and_90_percent_sensitivity():
    bars = [START + timedelta(minutes=i) for i in range(48)]
    assert recent_coverage(bars, decision_time=DECISION, session_start=START,
                           max_gap_minutes=10).reason == 'gap_too_long'
    assert recent_coverage(bars, decision_time=DECISION, session_start=START,
                           minimum_fraction=.9).required == 54
    assert not recent_coverage(bars, decision_time=DECISION, session_start=START,
                               minimum_fraction=.9).eligible


def test_requires_timezone():
    with pytest.raises(ValueError, match='timezone-aware'):
        recent_coverage([], decision_time=DECISION.replace(tzinfo=None), session_start=START)
