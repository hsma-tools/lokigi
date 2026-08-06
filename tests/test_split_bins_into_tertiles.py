"""Direct unit tests for `lokigi.utils._split_bins_into_tertiles()`.

Regression coverage for the fix that replaced `np.array_split` (which puts
any remainder bins in the FIRST chunks, so the two ends being compared were
often unequal in size -- e.g. 10 deciles gave 4/3/3, not 3/4/3) with an
equal-ends split that puts the remainder in the ignored middle chunk
instead.
"""

import pytest

from lokigi.utils import _split_bins_into_tertiles


def test_fewer_than_three_bins_returns_all_none():
    assert _split_bins_into_tertiles([1, 2]) == (None, None, None)


@pytest.mark.parametrize(
    "n_bins,expected_lens",
    [
        (3, (1, 1, 1)),
        (4, (1, 2, 1)),
        (5, (1, 3, 1)),
        (6, (2, 2, 2)),
        (7, (2, 3, 2)),
        (8, (2, 4, 2)),
        (9, (3, 3, 3)),
        (10, (3, 4, 3)),
        (11, (3, 5, 3)),
        (12, (4, 4, 4)),
    ],
)
def test_end_chunks_are_always_equal_sized(n_bins, expected_lens):
    bins = list(range(1, n_bins + 1))
    lower, middle, upper = _split_bins_into_tertiles(bins)

    assert len(lower) == len(upper)
    assert (len(lower), len(middle), len(upper)) == expected_lens


def test_chunks_concatenate_back_to_the_input_in_order():
    bins = list(range(1, 12))
    lower, middle, upper = _split_bins_into_tertiles(bins)
    assert lower + middle + upper == bins


def test_decile_split_is_1_3_4_7_8_10():
    lower, middle, upper = _split_bins_into_tertiles(range(1, 11))
    assert lower == [1, 2, 3]
    assert middle == [4, 5, 6, 7]
    assert upper == [8, 9, 10]


def test_disadvantaged_end_high_reverses_the_end_chunks():
    low = _split_bins_into_tertiles(range(1, 11), disadvantaged_end="low")
    high = _split_bins_into_tertiles(range(1, 11), disadvantaged_end="high")

    assert high == (low[2], low[1], low[0])
