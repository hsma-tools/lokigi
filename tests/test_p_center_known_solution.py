import pytest


def test_p_center_solve_finds_the_known_optimal_combination(loaded_problem):
    """Golden test: on `loaded_problem`'s 3-site/3-demand-location toy problem,
    hand-computing the worst-case (max) min-cost for all three p=2
    combinations gives a known, non-tied optimum. This checks solve()'s
    result against those independently computed numbers, not just against
    another solver's output.

    p_center minimises the maximum min-cost across demand points (weights
    have no effect -- see test_solve_weights.py::test_p_center_rejects_custom_weights),
    so no `weights` argument is passed here.

    Worst-case (max) of min(travel cost to selected sites):
      {Site_A, Site_B}: max(10, 5,  15) = 15
      {Site_A, Site_C}: max(10, 10, 8)  = 10  <- optimal
      {Site_B, Site_C}: max(25, 5,  8)  = 25
    """
    expected_by_combo = {
        frozenset({"Site_A", "Site_B"}): 15,
        frozenset({"Site_A", "Site_C"}): 10,
        frozenset({"Site_B", "Site_C"}): 25,
    }

    result = loaded_problem.solve(
        p=2, objectives="p_center", search_strategy="brute-force", show_progress=False
    )

    assert len(result.solution_df) == len(expected_by_combo)
    for _, row in result.solution_df.iterrows():
        combo = frozenset(row["site_names"])
        assert combo in expected_by_combo
        assert row["max"] == pytest.approx(expected_by_combo[combo])

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_A", "Site_C"})
    assert best["max"] == pytest.approx(10)
