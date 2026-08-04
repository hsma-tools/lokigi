"""Tests for `SiteSolutionSet.site_capacity_summary()` -- comparing a
chosen solution's allocated demand (from `site_allocation_summary()`)
against registered site capacity, answering "does the allocation fit?"

Fixture arithmetic used below (hand-derived):

- `capacitated_problem`: `loaded_problem`'s demand/travel data (Site_A/B/C,
  LSOA_1/2/3, demand=[100, 200, 150]) plus a `capacity` column
  ([200, 250, 999] for A/B/C). At p=2 {Site_A, Site_B}: A gets LSOA_1
  (100), B gets LSOA_2+LSOA_3 (350) -- see `test_site_allocation_summary.py`
  for the underlying nearest-site derivation. So:
    Site_A: allocated_demand=100, capacity=200 -> ratio=0.5 (under)
    Site_B: allocated_demand=350, capacity=250 -> ratio=1.4 (over)
- `capacitated_raw_counts_problem`: same, plus `current_load`=[50, 300, 0].
    Site_A: headroom = 200-50=150, incremental_headroom_ratio=100/150=0.667
    Site_B: headroom = 250-300=-50 (already over capacity today),
      incremental_headroom_ratio=350/-50=-7.0 (not clipped)
- `capacitated_ratio_problem`: same, plus `utilisation` (precomputed ratio)
  =[0.25, 1.2, 0.0] instead of current_load.
    Site_A: headroom = 200*(1-0.25)=150 (matches raw-counts path above)
    Site_B: headroom = 250*(1-1.2)=-50 (matches raw-counts path above)
- `capacitated_five_site_problem`: `five_site_problem`'s demand/travel data
  (uniform demand=100 each) plus capacity=[150, 50, 250, None, None] for
  Site_1..5. At p=3 {Site_1, Site_2, Site_3}: Site_1=100 (LSOA_3),
  Site_2=0 (closest to nothing -- the headline zero-allocation case),
  Site_3=300 (LSOA_1, LSOA_2, LSOA_4).
    Site_1: 100/150 = 0.667
    Site_2: 0/50 = 0.0 (explicit zero row, not NaN)
    Site_3: 300/250 = 1.2 (over)
"""

import numpy as np
import pandas as pd
import pytest

import lokigi


def _build_capacitated_problem(demand_df, candidate_df, travel_df, capacity_col_kwargs):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", **capacity_col_kwargs)
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def base_demand_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 200, 150],
        }
    )


@pytest.fixture
def base_travel_df():
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )


@pytest.fixture
def capacity_candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
            "capacity": [200, 250, 999],
        }
    )


@pytest.fixture
def capacitated_solution(base_demand_df, capacity_candidate_df, base_travel_df):
    """capacity_col registered, no baseline load data."""
    problem = _build_capacitated_problem(
        base_demand_df, capacity_candidate_df, base_travel_df, {"capacity_col": "capacity"}
    )
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def raw_counts_candidate_df(capacity_candidate_df):
    df = capacity_candidate_df.copy()
    df["current_load"] = [50, 300, 0]
    return df


@pytest.fixture
def capacitated_raw_counts_solution(base_demand_df, raw_counts_candidate_df, base_travel_df):
    problem = _build_capacitated_problem(
        base_demand_df,
        raw_counts_candidate_df,
        base_travel_df,
        {"capacity_col": "capacity", "current_load_col": "current_load"},
    )
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def ratio_candidate_df(capacity_candidate_df):
    df = capacity_candidate_df.copy()
    df["utilisation"] = [0.25, 1.2, 0.0]
    return df


@pytest.fixture
def capacitated_ratio_solution(base_demand_df, ratio_candidate_df, base_travel_df):
    problem = _build_capacitated_problem(
        base_demand_df,
        ratio_candidate_df,
        base_travel_df,
        {"capacity_col": "capacity", "utilisation_col": "utilisation"},
    )
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def five_site_capacity_candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"],
            "lat": [51.1, 51.2, 51.3, 51.4, 51.5],
            "long": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "capacity": [150, 50, 250, 300, 300],
        }
    )


@pytest.fixture
def capacitated_five_site_solution(five_site_capacity_candidate_df):
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [38.0, 18.0, 10.0, 28.0],
            "Site_2": [25.0, 40.0, 11.0, 31.0],
            "Site_3": [24.0, 13.0, 29.0, 13.0],
            "Site_4": [29.0, 16.0, 17.0, 16.0],
            "Site_5": [17.0, 15.0, 17.0, 36.0],
        }
    )
    problem = _build_capacitated_problem(
        demand_df, five_site_capacity_candidate_df, travel_df, {"capacity_col": "capacity"}
    )
    return problem.solve(p=3, search_strategy="brute-force", show_progress=False)


# --- core arithmetic --------------------------------------------------------


def test_allocated_utilisation_ratio_matches_hand_computed_values(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    assert summary.loc["Site_A", "allocated_utilisation_ratio"] == pytest.approx(0.5)
    assert summary.loc["Site_B", "allocated_utilisation_ratio"] == pytest.approx(1.4)


def test_allocated_demand_matches_site_allocation_summary(capacitated_solution):
    """Anti-drift: site_capacity_summary() must reuse
    site_allocation_summary() rather than reimplementing it."""
    capacity_summary = capacitated_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    allocation_summary = capacitated_solution.site_allocation_summary(
        by="demand", site_names=["Site_A", "Site_B"]
    )
    pd.testing.assert_series_equal(
        capacity_summary["allocated_demand"],
        allocation_summary["allocated_demand"],
    )
    assert capacity_summary.loc["Site_A", "n_regions"] == allocation_summary.loc[
        "Site_A", "n_regions"
    ]


def test_index_is_site_named_and_in_canonical_order_unsorted(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_B", "Site_A"])
    assert summary.index.name == "site"
    # Canonical order preserved regardless of site_names' given order --
    # matches site_allocation_summary()'s own reindex-against-solution
    # behaviour.
    assert list(summary.index) == ["Site_A", "Site_B"]


def test_columns_are_in_documented_order(capacitated_raw_counts_solution):
    summary = capacitated_raw_counts_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    assert list(summary.columns) == [
        "n_regions",
        "allocated_demand",
        "allocated_load",
        "capacity",
        "allocated_utilisation_ratio",
        "current_load",
        "headroom",
        "incremental_headroom_ratio",
        "residual_headroom",
    ]


# --- conversion rate ---------------------------------------------------------


def test_conversion_rate_scales_load_and_ratio_but_not_demand_or_capacity(
    capacitated_solution,
):
    summary = capacitated_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"], demand_to_capacity_rate=2.0
    )
    assert summary.loc["Site_A", "allocated_demand"] == 100
    assert summary.loc["Site_A", "capacity"] == 200
    assert summary.loc["Site_A", "allocated_load"] == pytest.approx(200.0)
    assert summary.loc["Site_A", "allocated_utilisation_ratio"] == pytest.approx(1.0)


def test_default_rate_leaves_allocated_load_equal_to_allocated_demand(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    pd.testing.assert_series_equal(
        summary["allocated_load"], summary["allocated_demand"].astype(float), check_names=False
    )


@pytest.mark.parametrize("bad_rate", [0, -1.0])
def test_non_positive_conversion_rate_raises(capacitated_solution, bad_rate):
    with pytest.raises(ValueError, match="demand_to_capacity_rate"):
        capacitated_solution.site_capacity_summary(
            site_names=["Site_A", "Site_B"], demand_to_capacity_rate=bad_rate
        )


def test_non_numeric_conversion_rate_raises(capacitated_solution):
    with pytest.raises(ValueError, match="demand_to_capacity_rate"):
        capacitated_solution.site_capacity_summary(
            site_names=["Site_A", "Site_B"], demand_to_capacity_rate="fast"
        )


# --- conditional columns -----------------------------------------------------


def test_capacity_only_registration_omits_all_headroom_columns(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    assert list(summary.columns) == [
        "n_regions",
        "allocated_demand",
        "allocated_load",
        "capacity",
        "allocated_utilisation_ratio",
    ]


def test_raw_counts_registration_adds_current_load_headroom_ratio_and_residual(
    capacitated_raw_counts_solution,
):
    summary = capacitated_raw_counts_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    assert summary.loc["Site_A", "current_load"] == 50
    assert summary.loc["Site_A", "headroom"] == pytest.approx(150.0)
    assert summary.loc["Site_A", "incremental_headroom_ratio"] == pytest.approx(100 / 150)
    assert summary.loc["Site_A", "residual_headroom"] == pytest.approx(50.0)
    assert "baseline_utilisation_ratio" not in summary.columns


def test_precomputed_ratio_registration_adds_baseline_ratio_but_not_current_load(
    capacitated_ratio_solution,
):
    summary = capacitated_ratio_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    assert summary.loc["Site_A", "baseline_utilisation_ratio"] == pytest.approx(0.25)
    assert "current_load" not in summary.columns


def test_precomputed_ratio_path_derives_headroom_from_capacity_times_one_minus_ratio(
    capacitated_ratio_solution,
):
    summary = capacitated_ratio_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    # Matches the raw-counts fixture's headroom exactly (200*0.75=150,
    # 250*-0.2=-50), by construction of the two fixtures.
    assert summary.loc["Site_A", "headroom"] == pytest.approx(150.0)
    assert summary.loc["Site_B", "headroom"] == pytest.approx(-50.0)


def test_headroom_ratio_negative_when_site_already_over_capacity_not_clipped(
    capacitated_raw_counts_solution,
):
    summary = capacitated_raw_counts_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    assert summary.loc["Site_B", "incremental_headroom_ratio"] == pytest.approx(-7.0)


def test_residual_headroom_negative_signals_shortfall(capacitated_raw_counts_solution):
    summary = capacitated_raw_counts_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    )
    assert summary.loc["Site_B", "residual_headroom"] == pytest.approx(-400.0)


# --- NaN / 0 / inf discipline -------------------------------------------------


def test_zero_allocation_site_gets_explicit_zero_row_not_dropped(
    capacitated_five_site_solution,
):
    summary = capacitated_five_site_solution.site_capacity_summary(
        site_names=["Site_1", "Site_2", "Site_3"]
    )
    assert list(summary.index) == ["Site_1", "Site_2", "Site_3"]
    assert summary.loc["Site_2", "n_regions"] == 0
    assert summary.loc["Site_2", "allocated_demand"] == 0


def test_zero_allocation_ratio_is_zero_not_nan(capacitated_five_site_solution):
    summary = capacitated_five_site_solution.site_capacity_summary(
        site_names=["Site_1", "Site_2", "Site_3"]
    )
    assert summary.loc["Site_2", "allocated_utilisation_ratio"] == 0.0
    assert not pd.isna(summary.loc["Site_2", "allocated_utilisation_ratio"])


def test_nan_capacity_gives_nan_ratio_and_warns_naming_the_site(
    base_demand_df, capacity_candidate_df, base_travel_df
):
    candidate_df = capacity_candidate_df.copy()
    candidate_df.loc[candidate_df["site_id"] == "Site_B", "capacity"] = np.nan
    problem = _build_capacitated_problem(
        base_demand_df, candidate_df, base_travel_df, {"capacity_col": "capacity"}
    )
    result = problem.solve(p=2, search_strategy="brute-force", show_progress=False)

    with pytest.warns(UserWarning, match="Site_B"):
        summary = result.site_capacity_summary(site_names=["Site_A", "Site_B"])

    assert pd.isna(summary.loc["Site_B", "capacity"])
    assert pd.isna(summary.loc["Site_B", "allocated_utilisation_ratio"])
    assert summary.loc["Site_A", "allocated_utilisation_ratio"] == pytest.approx(0.5)


def test_zero_capacity_with_allocation_gives_inf_not_clipped(
    base_demand_df, capacity_candidate_df, base_travel_df
):
    candidate_df = capacity_candidate_df.copy()
    candidate_df.loc[candidate_df["site_id"] == "Site_B", "capacity"] = 0
    problem = _build_capacitated_problem(
        base_demand_df, candidate_df, base_travel_df, {"capacity_col": "capacity"}
    )
    result = problem.solve(p=2, search_strategy="brute-force", show_progress=False)
    summary = result.site_capacity_summary(site_names=["Site_A", "Site_B"])
    assert summary.loc["Site_B", "allocated_utilisation_ratio"] == float("inf")


def test_zero_capacity_and_zero_allocation_gives_nan(capacitated_five_site_solution):
    """Site_2 (0 allocated demand) with 0 capacity would be 0/0 -- a known
    ambiguity between "no capacity" and "undefined", documented as such."""
    result = capacitated_five_site_solution
    result.site_problem.candidate_sites.loc[
        result.site_problem.candidate_sites["site_id"] == "Site_2", "capacity"
    ] = 0
    summary = result.site_capacity_summary(site_names=["Site_1", "Site_2", "Site_3"])
    assert pd.isna(summary.loc["Site_2", "allocated_utilisation_ratio"])


def test_ratio_above_one_is_not_clipped(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    assert summary.loc["Site_B", "allocated_utilisation_ratio"] == pytest.approx(1.4)


def test_unreachable_region_excluded_makes_allocated_demand_sum_below_total_and_warns():
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.5, 51.6],
            "long": [-0.1, -0.2],
            "capacity": [500, 500],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 12.0, 14.0, np.nan],
            "Site_B": [20.0, 22.0, 24.0, np.nan],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", capacity_col="capacity")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)
    result = problem.solve(p=1, unreachable_cost=1000, show_progress=False)

    with pytest.warns(UserWarning, match="region"):
        summary = result.site_capacity_summary()

    assert summary["allocated_demand"].sum() < 400


# --- capacity resolution / errors --------------------------------------------


def test_call_time_capacity_col_overrides_registered_column(capacitated_solution):
    candidate = capacitated_solution.site_problem.candidate_sites.copy()
    candidate["alt_capacity"] = [1000, 1000, 1000]
    capacitated_solution.site_problem.candidate_sites = candidate

    summary = capacitated_solution.site_capacity_summary(
        capacity_col="alt_capacity", site_names=["Site_A", "Site_B"]
    )
    assert summary.loc["Site_A", "capacity"] == 1000
    assert summary.loc["Site_A", "allocated_utilisation_ratio"] == pytest.approx(100 / 1000)


def test_falls_back_to_registered_capacity_col(capacitated_solution):
    summary = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    assert summary.loc["Site_A", "capacity"] == 200


def test_no_capacity_anywhere_raises_naming_both_routes(loaded_problem):
    result = loaded_problem.solve(p=2, search_strategy="brute-force", show_progress=False)
    with pytest.raises(ValueError, match="capacity_col"):
        result.site_capacity_summary(site_names=["Site_A", "Site_B"])


def test_unknown_capacity_col_raises_listing_available_columns(capacitated_solution):
    with pytest.raises(ValueError, match="not found in candidate_sites"):
        capacitated_solution.site_capacity_summary(
            capacity_col="does_not_exist", site_names=["Site_A", "Site_B"]
        )


def test_non_numeric_capacity_raises_typeerror(capacitated_solution):
    """add_sites() itself already validates capacity_col is numeric, so a
    non-numeric column can only reach site_capacity_summary()'s own
    resolver via a call-time capacity_col naming a different column that
    was never validated at registration time."""
    candidate = capacitated_solution.site_problem.candidate_sites.copy()
    candidate["alt_capacity"] = ["a", "b", "c"]
    capacitated_solution.site_problem.candidate_sites = candidate

    with pytest.raises(TypeError, match="numeric"):
        capacitated_solution.site_capacity_summary(
            capacity_col="alt_capacity", site_names=["Site_A", "Site_B"]
        )


def test_negative_capacity_raises_naming_sites(
    base_demand_df, capacity_candidate_df, base_travel_df
):
    candidate_df = capacity_candidate_df.copy()
    candidate_df.loc[candidate_df["site_id"] == "Site_B", "capacity"] = -10
    problem = _build_capacitated_problem(
        base_demand_df, candidate_df, base_travel_df, {"capacity_col": "capacity"}
    )
    result = problem.solve(p=2, search_strategy="brute-force", show_progress=False)
    with pytest.raises(ValueError, match="Site_B"):
        result.site_capacity_summary(site_names=["Site_A", "Site_B"])


def test_no_demand_data_raises_naming_add_demand(capacitated_solution, monkeypatch):
    """Unreachable via solve() (which installs a uniform-demand fallback),
    matching site_allocation_summary()'s own equivalent test."""
    monkeypatch.setattr(
        capacitated_solution.site_problem, "_demand_data_demand_col", None
    )
    with pytest.raises(ValueError, match="add_demand"):
        capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])


# --- selection / secondary data -----------------------------------------------


def test_secondary_matrix_reallocates_demand(loaded_problem_with_secondary_matrix):
    """With all three sites open, every region's closest site on the
    'public_transport' matrix is Site_C (uniform per-site cost, Site_C
    cheapest) -- see test_site_allocation_summary.py."""
    candidate = loaded_problem_with_secondary_matrix.candidate_sites.copy()
    candidate["capacity"] = candidate["site_id"].map(
        {"Site_A": 200, "Site_B": 250, "Site_C": 999}
    )
    loaded_problem_with_secondary_matrix.candidate_sites = candidate
    loaded_problem_with_secondary_matrix._candidate_sites_capacity_col = "capacity"

    result = loaded_problem_with_secondary_matrix.solve(p=3)
    summary = result.site_capacity_summary(
        matrix="public_transport", site_names=["Site_A", "Site_B", "Site_C"]
    )
    assert summary.loc["Site_A", "allocated_demand"] == 0
    assert summary.loc["Site_B", "allocated_demand"] == 0
    assert summary.loc["Site_C", "allocated_demand"] == 450


def test_secondary_demand_scenario_changes_allocated_demand(
    loaded_problem_with_secondary_demand,
):
    candidate = loaded_problem_with_secondary_demand.candidate_sites.copy()
    candidate["capacity"] = candidate["site_id"].map(
        {"Site_A": 200, "Site_B": 250, "Site_C": 999}
    )
    loaded_problem_with_secondary_demand.candidate_sites = candidate
    loaded_problem_with_secondary_demand._candidate_sites_capacity_col = "capacity"

    result = loaded_problem_with_secondary_demand.solve(p=1)  # Site_B wins primary
    default_summary = result.site_capacity_summary(demand=None)
    future_summary = result.site_capacity_summary(demand="future_demand")
    assert default_summary.loc["Site_B", "allocated_demand"] == 450
    assert future_summary.loc["Site_B", "allocated_demand"] == 400


def test_site_names_and_site_indices_select_the_same_solution(capacitated_solution):
    by_names = capacitated_solution.site_capacity_summary(site_names=["Site_A", "Site_B"])
    by_indices = capacitated_solution.site_capacity_summary(site_indices=[0, 1])
    pd.testing.assert_frame_equal(by_names, by_indices)
