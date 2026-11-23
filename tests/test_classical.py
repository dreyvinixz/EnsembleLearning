from ensemble_learning.baselines import CLASSICAL_MODELS, run_independent_classical
from ensemble_learning.contracts import validate_frame
from ensemble_learning.splits import chronological_holdout


def test_classical_models_only_evaluate_validation(sample_frame):
    split = chronological_holdout(
        validate_frame(sample_frame), validation_sessions=3, test_sessions=3
    )
    for model_name in CLASSICAL_MODELS:
        result = run_independent_classical(split, model_name=model_name, seed=7)
        assert result["method"] == "independent_" + model_name
        assert set(result["horizons"]) == {"5", "10", "15"}
        for metrics in result["horizons"].values():
            assert "test" not in metrics
            assert 0 <= metrics["validation"]["brier"] <= 1
