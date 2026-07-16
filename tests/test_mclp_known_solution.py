import pytest


def test_mclp_solve_finds_the_known_optimal_combination(loaded_problem):
    """Golden test: on `loaded_problem`'s 3-site/3-demand-location toy problem,
    hand-computing the coverage proportion (share of demand *locations*, not
    demand-weighted -- see site.py's `within_threshold` / site_solutions.py's
    `proportion_within_coverage_threshold`) for all three p=2 combinations
    at threshold=12 gives a known, non-tied optimum.

    within_threshold is `min_cost < threshold_for_coverage` (strict), and
    12 sits strictly between the distinct min-cost values that occur here
    (5, 8, 10 vs. 15, 25), so there's no boundary ambiguity.

    Coverage (count of demand points with min-cost < 12, out of 3):
      {Site_A, Site_B}: mins [10, 5,  15] -> covered [T, T, F] -> 2/3
      {Site_A, Site_C}: mins [10, 10, 8]  -> covered [T, T, T] -> 3/3  <- optimal
      {Site_B, Site_C}: mins [25, 5,  8]  -> covered [F, T, T] -> 2/3
    """
    expected_by_combo = {
        frozenset({"Site_A", "Site_B"}): 2 / 3,
        frozenset({"Site_A", "Site_C"}): 3 / 3,
        frozenset({"Site_B", "Site_C"}): 2 / 3,
    }

    result = loaded_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=12,
    )

    assert len(result.solution_df) == len(expected_by_combo)
    for _, row in result.solution_df.iterrows():
        combo = frozenset(row["site_names"])
        assert combo in expected_by_combo
        assert row["proportion_within_coverage_threshold"] == pytest.approx(
            expected_by_combo[combo]
        )

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_A", "Site_C"})
    assert best["proportion_within_coverage_threshold"] == pytest.approx(1.0)
