import pytest


def test_p_median_solve_finds_the_known_optimal_combination(loaded_problem):
    """Golden test: on `loaded_problem`'s 3-site/3-demand-location toy problem,
    hand-computing the demand-weighted mean of min-cost for all three p=2
    combinations gives a known, non-tied optimum. This checks solve()'s
    result against those independently computed numbers, not just against
    another solver's output.

    Demand-weighted average of min(travel cost to selected sites):
      {Site_A, Site_B}: (100*10 + 200*5  + 150*15) / 450 = 4250/450
      {Site_A, Site_C}: (100*10 + 200*10 + 150*8)  / 450 = 4200/450  <- optimal
      {Site_B, Site_C}: (100*25 + 200*5  + 150*8)  / 450 = 4700/450
    """
    expected_by_combo = {
        frozenset({"Site_A", "Site_B"}): 4250 / 450,
        frozenset({"Site_A", "Site_C"}): 4200 / 450,
        frozenset({"Site_B", "Site_C"}): 4700 / 450,
    }

    result = loaded_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    assert len(result.solution_df) == len(expected_by_combo)
    for _, row in result.solution_df.iterrows():
        combo = frozenset(row["site_names"])
        assert combo in expected_by_combo
        assert row["weighted_average"] == pytest.approx(expected_by_combo[combo])

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_A", "Site_C"})
    assert best["weighted_average"] == pytest.approx(4200 / 450)
