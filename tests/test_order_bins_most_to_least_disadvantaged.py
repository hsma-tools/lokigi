"""Direct unit tests for `lokigi.utils._order_bins_most_to_least_disadvantaged()`
-- the shared helper behind `population_impact_by_equity_group()` and
`check_solution_equity()`'s "most to least disadvantaged" bar ordering.
Built on top of `_split_bins_into_tertiles()`'s own tertile split, with a
fallback for fewer than 3 distinct bands (which `_split_bins_into_tertiles`
can't split into thirds at all).
"""

from lokigi.utils import _order_bins_most_to_least_disadvantaged


def test_disadvantaged_end_low_keeps_ascending_order():
    result = _order_bins_most_to_least_disadvantaged([1, 2, 3, 4, 5, 6], "low")
    assert result == [1, 2, 3, 4, 5, 6]


def test_disadvantaged_end_high_reverses_tertile_order():
    """3+ bins go through _split_bins_into_tertiles, which reverses the
    order of the THREE CHUNKS but keeps each chunk's own ascending order
    (ties within a tertile stay in raw-bin order) -- so this is a
    tertile-level reversal, not a full element-wise one: [5,6] (most
    disadvantaged tertile), then [3,4], then [1,2]."""
    result = _order_bins_most_to_least_disadvantaged([1, 2, 3, 4, 5, 6], "high")
    assert result == [5, 6, 3, 4, 1, 2]


def test_none_defaults_to_low_behaviour():
    result = _order_bins_most_to_least_disadvantaged([1, 2, 3, 4, 5, 6], None)
    assert result == [1, 2, 3, 4, 5, 6]


def test_fewer_than_three_bins_disadvantaged_end_high_still_reverses():
    """_split_bins_into_tertiles can't split 2 bins into thirds (returns
    (None, None, None)) -- without an explicit fallback here, this case
    would silently keep ascending (least-disadvantaged-first) order even
    though disadvantaged_end="high" was requested."""
    result = _order_bins_most_to_least_disadvantaged([1, 10], "high")
    assert result == [10, 1]


def test_fewer_than_three_bins_disadvantaged_end_low_keeps_ascending():
    result = _order_bins_most_to_least_disadvantaged([1, 10], "low")
    assert result == [1, 10]


def test_matches_split_bins_into_tertiles_concatenation():
    """For 3+ bins, the result is exactly the tertile split concatenated
    back together -- this helper doesn't reimplement the tertile logic,
    it just chains _split_bins_into_tertiles()'s own three chunks."""
    from lokigi.utils import _split_bins_into_tertiles

    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    lower, middle, upper = _split_bins_into_tertiles(bins, "high")
    result = _order_bins_most_to_least_disadvantaged(bins, "high")
    assert result == lower + middle + upper
