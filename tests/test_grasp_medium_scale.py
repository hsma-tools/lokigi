"""
Medium-scale GRASP regression coverage.

The existing GRASP tests in test_search_strategies.py and test_backtests.py
use tiny fixtures (4-5 sites, grasp_max_attempts=10) chosen for speed and to
isolate a single behaviour each. None of them exercise GRASP at a scale
close to real usage -- dozens of candidate sites and hundreds of attempts,
where the construction/local-search loop in `_grasp` (site_solvers.py) runs
long enough for its lru_cache to actually matter and for diversity rejection
to kick in repeatedly.

That gap used to be covered incidentally by examples/location/greedy_and_grasp
whenever it was executed (CI had no quarto freeze cache, so every render
re-ran the notebook's grasp_max_attempts=50000 cells end-to-end). That
notebook now has execute: {enabled: false} -- see its front matter -- so it
no longer re-executes in CI at all. This file is the deliberate replacement:
a real backtest at a size that would actually surface a regression in the
construction phase, RCL selection, or local search, without paying for the
notebook's 50000-attempt cells.

Pinned via `assert_backtest` the same way as test_backtests.py: expected
values live in tests/backtest_snapshots.json. Regenerate with
`pytest tests/test_grasp_medium_scale.py --update-backtests` after a
deliberate solver change, and review the diff before committing it.

Marked `slow_grasp` so CI only runs it on one leg of the OS/Python matrix
(see .github/workflows/tests.yml) rather than on every combination.
"""

import pandas as pd
import pytest

import lokigi

pytestmark = pytest.mark.slow_grasp


def _fingerprint(result, cols=("weighted_average", "max", "unweighted_average")):
    """Order-preserving, tuple-based snapshot of a SiteSolutionSet.solution_df."""
    rows = []
    for _, row in result.solution_df.iterrows():
        entry = [tuple(sorted(row["site_names"]))]
        for col in cols:
            value = row[col]
            entry.append(round(float(value), 6) if pd.notna(value) else None)
        rows.append(tuple(entry))
    return rows


@pytest.fixture
def medium_problem():
    """
    A 25-site / 20-demand-point problem, built from a fixed grid rather than
    random sampling so it's fully deterministic without depending on numpy's
    RNG. Travel times are Euclidean grid-distance plus a small index-derived
    jitter, just enough to avoid exact ties dominating the RCL.
    """
    site_rows = [
        {"site_id": f"Site_{i}_{j}", "lat": 51.0 + i * 0.1, "long": -0.1 - j * 0.1}
        for i in range(5)
        for j in range(5)
    ]
    candidate_df = pd.DataFrame(site_rows)

    demand_rows = [
        {"location_id": f"LSOA_{i}_{j}", "demand": 50 + (i * 5 + j * 3) % 40}
        for i in range(4)
        for j in range(5)
    ]
    demand_df = pd.DataFrame(demand_rows)

    travel_cols = {"source_id": [row["location_id"] for row in demand_rows]}
    for site_index, site_row in enumerate(site_rows):
        times = []
        for demand_index, _ in enumerate(demand_rows):
            demand_lat = 51.0 + (demand_index // 5) * 0.1
            demand_long = -0.1 - (demand_index % 5) * 0.1
            distance = abs(site_row["lat"] - demand_lat) + abs(
                site_row["long"] - demand_long
            )
            jitter = ((site_index * 7 + demand_index * 3) % 5) * 0.5
            times.append(round(distance * 100 + jitter, 2))
        travel_cols[site_row["site_id"]] = times
    travel_df = pd.DataFrame(travel_cols)

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def test_grasp_medium_scale_p_median(medium_problem, assert_backtest):
    result = medium_problem.solve(
        p=8,
        objectives="p_median",
        search_strategy="grasp",
        grasp_alpha=0.3,
        grasp_max_attempts=500,
        grasp_num_solutions=10,
        show_progress=False,
    )
    assert len(result.solution_df) == 10
    assert_backtest(_fingerprint(result))
