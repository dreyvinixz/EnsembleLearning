"""Architecture smoke tests; skipped without optional PyTorch."""
import pytest

torch = pytest.importorskip("torch")
from ensemble_learning.sequence_models import (
    NEURAL_ARCHITECTURES,
    IndependentSequenceClassifier,
    MultiTaskSequenceClassifier,
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


@pytest.mark.parametrize("architecture", NEURAL_ARCHITECTURES)
def test_gradients_reach_independent_and_all_task_heads(architecture):
    settings = {"sequence_length": 12} if architecture == "mlp" else {}
    x = torch.randn((4, 12, 5))
    individual = IndependentSequenceClassifier(architecture, input_size=5, **settings)
    individual(x).square().mean().backward()
    assert individual.head.weight.grad is not None
    joint = MultiTaskSequenceClassifier(architecture, input_size=5, **settings)
    sum(output.square().mean() for output in joint(x).values()).backward()
    assert all(head.weight.grad is not None for head in joint.heads.values())
