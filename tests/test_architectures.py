"""Architecture smoke tests; skipped without optional PyTorch."""
import pytest

torch = pytest.importorskip("torch")
from ensemble_learning.sequence_models import (  # noqa: I001
    IndependentSequenceClassifier,
    MultiTaskSequenceClassifier,
    NEURAL_ARCHITECTURES,
)


@pytest.mark.parametrize("architecture", NEURAL_ARCHITECTURES)
def test_forward_shapes(architecture):
    settings = {"sequence_length": 12} if architecture == "mlp" else {}
    x = torch.zeros((4, 12, 5))
    individual = IndependentSequenceClassifier(architecture, input_size=5, **settings)
    joint = MultiTaskSequenceClassifier(architecture, input_size=5, **settings)
    assert individual(x).shape == (4, 2)
    assert set(joint(x)) == {5, 10, 15}
    assert all(v.shape == (4, 2) for v in joint(x).values())


def test_mlp_rejects_different_window_length():
    model = IndependentSequenceClassifier("mlp", input_size=5, sequence_length=12)
    with pytest.raises(ValueError, match="history differs"):
        model(torch.zeros((2, 10, 5)))
