import pandas as pd
import pytest

import lokigi


def test_mclp_solve_finds_the_known_optimal_combination(loaded_problem):
    """Golden test: on `loaded_problem`'s 3-site/3-demand-location toy problem,
    hand-computing coverage for all three p=2 combinations at threshold=12
    gives a known, non-tied optimum.

    Since v0.7.0 `proportion_within_coverage_threshold` is demand-weighted
    (demand here is 100/200/150, total 450), and the region count is preserved
    as `proportion_regions_within_coverage_threshold`. Both are asserted --
    note {Site_B, Site_C} is where they disagree, since the region it misses
    is the smallest one.

    within_threshold is `min_cost < threshold_for_coverage` (strict), and
    12 sits strictly between the distinct min-cost values that occur here
    (5, 8, 10 vs. 15, 25), so there's no boundary ambiguity.

    Coverage (demand points with min-cost < 12):
      {Site_A, Site_B}: mins [10, 5,  15] -> [T, T, F] -> 2/3 regions, 300/450 demand
      {Site_A, Site_C}: mins [10, 10, 8]  -> [T, T, T] -> 3/3 regions, 450/450 demand  <- optimal
      {Site_B, Site_C}: mins [25, 5,  8]  -> [F, T, T] -> 2/3 regions, 350/450 demand
    """
    expected_by_combo = {
        # combo: (demand-weighted, regions)
        frozenset({"Site_A", "Site_B"}): (300 / 450, 2 / 3),
        frozenset({"Site_A", "Site_C"}): (450 / 450, 3 / 3),
        frozenset({"Site_B", "Site_C"}): (350 / 450, 2 / 3),
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
        expected_demand, expected_regions = expected_by_combo[combo]
        assert row["proportion_within_coverage_threshold"] == pytest.approx(
            expected_demand
        )
        assert row["proportion_regions_within_coverage_threshold"] == pytest.approx(
            expected_regions
        )

    best = result.solution_df.iloc[0]
    assert frozenset(best["site_names"]) == frozenset({"Site_A", "Site_C"})
    assert best["proportion_within_coverage_threshold"] == pytest.approx(1.0)


def test_mclp_maximises_covered_demand_not_covered_regions():
    """The v0.7.0 behaviour change, on a problem built so the two definitions
    of coverage disagree about the winner.

    Site_X reaches three tiny regions (demand 1 each); Site_Y reaches the one
    large region (demand 100). Counting regions, Site_X wins 3/4 to 1/4.
    Counting demand, Site_Y wins 100/103 to 3/103. MCLP must pick Site_Y,
    matching the textbook Maximal Covering Location Problem.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_BIG"],
            "demand": [1, 1, 1, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_X", "Site_Y"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_BIG"],
            "Site_X": [5.0, 5.0, 5.0, 50.0],
            "Site_Y": [50.0, 50.0, 50.0, 5.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.solve(
        p=1,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=10.0,
    )

    by_site = {
        row["site_names"][0]: row for _, row in result.solution_df.iterrows()
    }

    assert by_site["Site_Y"]["proportion_within_coverage_threshold"] == pytest.approx(
        100 / 103
    )
    assert by_site["Site_X"]["proportion_within_coverage_threshold"] == pytest.approx(
        3 / 103
    )
    # The region count still prefers Site_X -- it is reported, just not optimised.
    assert by_site["Site_X"][
        "proportion_regions_within_coverage_threshold"
    ] == pytest.approx(0.75)
    assert by_site["Site_Y"][
        "proportion_regions_within_coverage_threshold"
    ] == pytest.approx(0.25)

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_Y"]
