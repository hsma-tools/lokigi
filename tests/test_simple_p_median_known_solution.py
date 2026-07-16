import pytest


def test_simple_p_median_solve_finds_the_known_optimal_combination(loaded_problem):
    """Golden test: on `loaded_problem`'s 3-site/3-demand-location toy problem,
    hand-computing the *unweighted* (plain, not demand-weighted) mean of
    min-cost for all three p=2 combinations gives a known, non-tied optimum.
    simple_p_median ranks on `unweighted_average`, ignoring demand entirely --
    this is deliberately checked against loaded_problem's unequal demand
    values ([100, 200, 150]) to confirm the demand column has no bearing on
    the result (contrast with test_p_median_known_solution.py, which uses
    the same travel matrix but ranks on the demand-weighted average and
    gets different numbers for the same combinations).

    Unweighted average of min(travel cost to selected sites):
      {Site_A, Site_B}: mean(10, 5,  15) = 30/3 = 10.0
      {Site_A, Site_C}: mean(10, 10, 8)  = 28/3 ~= 9.333  <- optimal
      {Site_B, Site_C}: mean(25, 5,  8)  = 38/3 ~= 12.667
    """
    expected_by_combo = {
        frozenset({"Site_A", "Site_B"}): 30 / 3,
        frozenset({"Site_A", "Site_C"}): 28 / 3,
        frozenset({"Site_B", "Site_C"}): 38 / 3,
    }

    result = loaded_problem.solve(
        p=2,
        objectives="simple_p_median",
        search_strategy="brute-force",
        show_progress=False,
    )

    assert len(result.solution_df) == len(expected_by_combo)
    for _, row in result.solution_df.iterrows():
        combo = frozenset(row["site_names"])
        assert combo in expected_by_combo
        assert row["unweighted_average"] == pytest.approx(expected_by_combo[combo])

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_A", "Site_C"})
    assert best["unweighted_average"] == pytest.approx(28 / 3)
