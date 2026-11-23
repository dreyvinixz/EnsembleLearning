import pytest

torch = pytest.importorskip("torch")
from ensemble_learning.gru import IndependentGRU, MultiTaskGRU  # noqa: E402


def test_gru_outputs_have_expected_shapes():
    x = torch.zeros((3, 12, 4), dtype=torch.float32)
    assert IndependentGRU(input_size=4)(x).shape == (3, 2)
    outputs = MultiTaskGRU(input_size=4)(x)
    assert set(outputs) == {5, 10, 15}
    assert all(value.shape == (3, 2) for value in outputs.values())
