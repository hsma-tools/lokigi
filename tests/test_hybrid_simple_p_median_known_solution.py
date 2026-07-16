import pytest


def test_hybrid_simple_p_median_with_a_non_binding_cutoff_matches_plain_simple_p_median(
    hybrid_p_median_problem,
):
    """Reuses `hybrid_p_median_problem` -- its demand column is equal (100)
    across all four LSOAs, so unweighted_average and weighted_average are
    numerically identical here, letting the same fixture and numbers serve
    both the p_median/hybrid_p_median and simple_p_median/hybrid_simple_p_median
    golden tests.

    With a max_value_cutoff higher than every combination's worst-case travel
    time, the filter is a no-op, so hybrid_simple_p_median should reduce to
    plain simple_p_median and pick the combination with the lowest
    unweighted_average: {Site_1, Site_3} at 5.5 (mins [2, 2, 2, 16] over
    LSOA_1-4), even though its worst-case travel time (16) is high."""
    result = hybrid_p_median_problem.solve(
        p=2,
        objectives="hybrid_simple_p_median",
        search_strategy="brute-force",
        show_progress=False,
        max_value_cutoff=1000,
    )

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_1", "Site_3"})
    assert best["unweighted_average"] == pytest.approx(5.5)
    assert best["max"] == pytest.approx(16)


def test_hybrid_simple_p_median_cutoff_excludes_the_unconstrained_optimum(
    hybrid_p_median_problem,
):
    """
    Hand-computed unweighted_average / max for all three p=2 combinations:
      {Site_1, Site_3}: mins [2, 2, 2, 16]  -> avg 5.5, max 16 (best avg, but breaches any cutoff < 16)
      {Site_1, Site_2}: mins [2, 8, 10, 8]  -> avg 7.0, max 10 (feasible under cutoff=10)
      {Site_2, Site_3}: mins [16, 2, 2, 8]  -> avg 7.0, max 16 (also breaches cutoff=10)

    With max_value_cutoff=10, {Site_1, Site_3} and {Site_2, Site_3} are both
    excluded (max=16 > 10), leaving {Site_1, Site_2} as the sole feasible --
    and therefore best -- combination, despite it NOT being the unconstrained
    simple_p_median optimum.
    """
    result = hybrid_p_median_problem.solve(
        p=2,
        objectives="hybrid_simple_p_median",
        search_strategy="brute-force",
        show_progress=False,
        max_value_cutoff=10,
    )

    assert len(result.solution_df) == 1
    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_1", "Site_2"})
    assert best["unweighted_average"] == pytest.approx(7.0)
    assert best["max"] == pytest.approx(10)
