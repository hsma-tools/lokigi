"""Tests for evaluate_single_solution_single_objective()'s per-row lowest-
cost selection (`min_cost`/`selected_site`, via `_safe_idxmin` in site.py)
in the ordinary, non-degenerate case.

Existing coverage only exercises the two edge cases `_safe_idxmin` was
written to handle: an all-NaN row (test_missing_travel_values.py) and an
exact tie between two sites (test_site_allocation_summary.py). Neither
proves the everyday row-wise minimum -- distinct, always-reachable travel
times, no ties -- actually picks the right site for the right row.
"""

import pandas as pd
import pytest

import lokigi


@pytest.fixture
def demand_df():
    return pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )


@pytest.fixture
def candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
        }
    )


@pytest.fixture
def travel_df():
    """Distinct, non-tied, always-reachable travel times for every (site,
    demand) pair, so each row has exactly one unambiguous minimum among the
    two selected sites (Site_A, Site_B):
      LSOA_1: Site_A=12, Site_B=7  -> min=7,  Site_B
      LSOA_2: Site_A=9,  Site_B=15 -> min=9,  Site_A
      LSOA_3: Site_A=25, Site_B=6  -> min=6,  Site_B
    Site_C is a third candidate present in the matrix but not selected
    below, matching the usual "pick a subset of candidates" pattern."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [12.0, 9.0, 25.0],
            "Site_B": [7.0, 15.0, 6.0],
            "Site_C": [20.0, 3.0, 11.0],
        }
    )


@pytest.fixture
def secondary_travel_df():
    """A 'public_transport' secondary matrix with its own distinct,
    always-reachable values, deliberately choosing the OTHER selected site
    as the minimum on two of the three rows relative to the primary
    matrix -- so a bug that accidentally reused the primary matrix's
    selection (rather than computing its own) would be visible:
      LSOA_1: Site_A=30, Site_B=18 -> min=18, Site_B
      LSOA_2: Site_A=22, Site_B=40 -> min=22, Site_A
      LSOA_3: Site_A=14, Site_B=27 -> min=14, Site_A
    """
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [30.0, 22.0, 14.0],
            "Site_B": [18.0, 40.0, 27.0],
            "Site_C": [50.0, 50.0, 50.0],
        }
    )


@pytest.fixture
def problem(demand_df, candidate_df, travel_df):
    site_problem = lokigi.site.SiteProblem(debug_mode=False)
    site_problem.add_demand(
        demand_df, demand_col="demand", location_id_col="location_id"
    )
    site_problem.add_sites(candidate_df, candidate_id_col="site_id")
    site_problem.add_travel_matrix(travel_df, source_col="source_id")
    return site_problem


def test_min_cost_is_the_true_row_wise_minimum(problem):
    df = problem.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"]
    ).evaluated_combination_df

    assert df["min_cost"].tolist() == pytest.approx([7.0, 9.0, 6.0])


def test_selected_site_matches_the_site_that_produced_the_minimum(problem):
    df = problem.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"]
    ).evaluated_combination_df

    assert df["selected_site"].tolist() == ["Site_B", "Site_A", "Site_B"]


def test_secondary_matrix_selection_is_independent_of_the_primary(
    problem, secondary_travel_df
):
    """The secondary-matrix columns (min_cost__<label>/selected_site__
    <label>, site.py:444-450) run the same _safe_idxmin selection logic a
    second time, over the secondary matrix's own values -- this has zero
    direct coverage of its normal case either."""
    problem.add_secondary_travel_matrix(
        secondary_travel_df, source_col="source_id", label="public_transport"
    )

    df = problem.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"]
    ).evaluated_combination_df

    assert df["min_cost__public_transport"].tolist() == pytest.approx(
        [18.0, 22.0, 14.0]
    )
    assert df["selected_site__public_transport"].tolist() == [
        "Site_B",
        "Site_A",
        "Site_A",
    ]
    # The secondary selection genuinely differs from the primary's on two
    # of three rows -- proof it wasn't just copied across.
    assert df["selected_site"].tolist() != df["selected_site__public_transport"].tolist()
