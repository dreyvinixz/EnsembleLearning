from itertools import pairwise

import pytest

from ensemble_learning.block_bagging import sample_session_blocks


def test_session_block_bootstrap_is_reproducible_and_keeps_contiguous_blocks():
    sessions = [f"2026-02-{day:02d}" for day in range(2, 12)]
    first = sample_session_blocks(sessions, block_size=3, seed=17)
    second = sample_session_blocks(sessions, block_size=3, seed=17)
    assert first == second
    assert len(first) == len(sessions)
    positions = {session: index for index, session in enumerate(sessions)}
    for index in range(0, len(first), 3):
        block = first[index:index + 3]
        assert len(block) <= 3
        assert all(positions[right] == positions[left] + 1
                   for left, right in pairwise(block))


def test_session_block_bootstrap_validates_inputs_and_handles_short_history():
    assert sample_session_blocks(["2026-02-02", "2026-02-03"],
                                 block_size=5, seed=1) == [
        "2026-02-02", "2026-02-03",
    ]
    with pytest.raises(ValueError, match="[Aa]t least one"):
        sample_session_blocks([], block_size=2, seed=1)
    with pytest.raises(ValueError, match="unique"):
        sample_session_blocks(["2026-02-02", "2026-02-02"], block_size=1, seed=1)
    with pytest.raises(ValueError, match="positive"):
        sample_session_blocks(["2026-02-02"], block_size=0, seed=1)
