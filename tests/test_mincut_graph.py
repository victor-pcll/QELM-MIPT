"""Tests of the space-time min cut with partition (QELM.evaluation.mincut_graph)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from QELM.evaluation import fourier_framework as ff
from QELM.evaluation import r2_budget as rb
from QELM.evaluation.effect_rank import mincut_bits
from QELM.evaluation.mincut_graph import (
    SINK,
    SOURCE,
    hartley_mincut,
    input_bound,
    mincut,
    record_bound,
)


def _mask(depth: int, n: int, cells: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((depth, n), dtype=bool)
    for t, s in cells:
        m[t, s] = True
    return m


def test_matches_library_with_final_readout() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(2, 7))
        masks = rng.random((n, n)) < rng.random()
        assert mincut(masks, final=True).value == mincut_bits(masks)


def test_cut_edges_sum_to_value_and_separate() -> None:
    rng = np.random.default_rng(1)
    for final in (True, False):
        for _ in range(40):
            masks = rng.random((8, 4)) < 0.3
            cut = mincut(masks, final=final)
            assert sum(e.bits for e in cut.cut) == cut.value
            assert SOURCE in cut.source_side
            assert SINK not in cut.source_side or cut.value == 0


def test_bounds_hold() -> None:
    rng = np.random.default_rng(2)
    for final in (True, False):
        for _ in range(60):
            masks = rng.random((8, 4)) < rng.random()
            c = mincut(masks, final=final).value
            assert c <= record_bound(masks, final)
            assert c <= input_bound(masks)
            assert c <= 8


@pytest.mark.parametrize(
    ("cells", "final", "expected"),
    [
        ([], False, 0),
        ([(4, 2)], False, 1),
        ([(1, 1), (5, 3)], False, 2),
        ([(3, 0), (3, 1), (4, 0)], False, 2),  # three clicks, two bits
        ([], True, 4),
        ([(7, 0)], True, 4),  # last layer repeats the readout
        ([(1, 2)], True, 5),
    ],
)
def test_known_cases(cells: list[tuple[int, int]], final: bool, expected: int) -> None:
    assert mincut(_mask(8, 4, cells), final=final).value == expected


def test_zeno() -> None:
    for final in (True, False):
        assert mincut(np.ones((8, 4), dtype=bool), final=final).value == 4


def test_pruning_needed_without_readout() -> None:
    masks = _mask(8, 4, [(3, 0), (3, 1), (4, 0)])
    assert mincut(masks, final=False, prune=False).value == 3
    assert mincut(masks, final=False).value == 2


def test_single_and_pair_ranks_without_readout() -> None:
    """Exact Haar ranks: every single click gives 2, every pair gives 4."""
    rng = np.random.default_rng(3)
    n, depth = 3, 4
    base = ff.draw_circuit(n, depth, 0.0, "haar", rng)
    cells = [(t, s) for t in range(depth) for s in range(n)]
    for k, rank in ((1, 2), (2, 4)):
        for chosen in itertools.combinations(cells, k):
            masks = _mask(depth, n, list(chosen))
            basis = rb.effect_basis(ff.with_mask(base, masks), final=False)
            assert basis.shape[0] == rank == 2 ** mincut(masks, final=False).value


def test_rank_bounded_by_cut() -> None:
    rng = np.random.default_rng(4)
    for final in (True, False):
        for _ in range(30):
            c = ff.draw_circuit(3, 4, float(rng.uniform(0.1, 0.7)), "haar", rng)
            rank = rb.effect_basis(c, final=final).shape[0]
            cut = mincut(c.mask, final=final).value
            assert rank <= 2**cut
            if final:
                assert rank == 2**cut


def test_hartley_ends() -> None:
    rng = np.random.default_rng(5)
    assert hartley_mincut(8, 16, 0.0, rng) == 4
    assert hartley_mincut(8, 16, 1.0, rng) == 0
    assert hartley_mincut(8, 16, 1.0, rng, half_layers=False) == 0
