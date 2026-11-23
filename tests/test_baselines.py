from ensemble_learning.baselines import run_independent_logreg
from ensemble_learning.contracts import validate_frame
from ensemble_learning.splits import chronological_holdout


def test_independent_runs_validation_only(sample_frame):
    split = chronological_holdout(validate_frame(sample_frame), validation_sessions=3, test_sessions=3)
    result = run_independent_logreg(split, seed=13)
    assert result["method"] == "independent_logreg"
    assert set(result["horizons"]) == {"5", "10", "15"}
    for task in result["horizons"].values():
        assert "test" not in task
        assert -1 <= task["validation"]["mcc"] <= 1
        assert 0 <= task["validation"]["brier"] <= 1
