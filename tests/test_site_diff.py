"""Tests for `show_solutions_summary()`'s `diff_against` parameter --
per-row site-name diff columns (`Sites added`/`Sites removed`/`Sites
changed (vs <reference>)`) that make near-identical top-N rows (differing
by only one or two sites out of several) distinguishable without a
baseline, unlike `sites_closed_vs_baseline`/`sites_added_vs_baseline`
(which require `solve(baseline=...)`).

Uses a self-contained 4-site/3-demand-location problem (Site_A/B/C/D,
L1/L2/L3), brute-forced at p=2 (6 combinations), so every rank's
site_names and hence every diff is hand-derivable:

    L1: A=10, B=25, C=30, D=12 -> best pair overall is {B, D} (rank 1)
    L2: A=20, B=5,  C=10, D=22
    L3: A=30, B=15, C=8,  D=9

Brute-force ranking (verified by running the scenario, not assumed):
    Rank 1: {Site_B, Site_D}
    Rank 2: {Site_A, Site_C}
    Rank 3 (tie): {Site_A, Site_B}, {Site_C, Site_D}
    Rank 4: {Site_B, Site_C}
    Rank 5: {Site_A, Site_D}
"""

import pandas as pd
import pytest

import lokigi.site


@pytest.fixture
def four_site_problem():
    demand_df = pd.DataFrame({"location_id": ["L1", "L2", "L3"], "demand": [100, 100, 100]})
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "lat": [51.1, 51.2, 51.3, 51.4],
            "long": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["L1", "L2", "L3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
            "Site_D": [12.0, 22.0, 9.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def four_site_problem_with_required(four_site_problem):
    """Same as `four_site_problem`, but Site_A is flagged as required, via
    a fresh add_sites() call (required_sites_col can only be set there)."""
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "lat": [51.1, 51.2, 51.3, 51.4],
            "long": [-0.1, -0.2, -0.3, -0.4],
            "required": ["Y", "N", "N", "N"],
        }
    )
    four_site_problem.add_sites(
        candidate_df, candidate_id_col="site_id", required_sites_col="required"
    )
    return four_site_problem


@pytest.fixture
def six_combos(four_site_problem):
    return four_site_problem.solve(p=2, search_strategy="brute-force", show_progress=False)


def test_rank_1_is_the_expected_pair(six_combos):
    """Sanity check the fixture's own claim before relying on it below."""
    top = six_combos.show_solutions(rounding=None).iloc[0]
    assert set(top["site_names"]) == {"Site_B", "Site_D"}


def test_default_falls_back_to_rank_1_without_required_sites(six_combos):
    df = six_combos.show_solutions_summary()
    assert "Sites added (vs rank 1)" in df.columns
    assert "Sites removed (vs rank 1)" in df.columns
    assert "Sites changed (vs rank 1)" in df.columns


def test_rank_1_row_has_zero_diff_against_itself(six_combos):
    df = six_combos.show_solutions_summary(diff_against="rank_1")
    rank_1_row = df[df["Rank"] == 1].iloc[0]
    assert rank_1_row["Sites added (vs rank 1)"] == ""
    assert rank_1_row["Sites removed (vs rank 1)"] == ""
    assert rank_1_row["Sites changed (vs rank 1)"] == 0


def test_rank_1_diff_matches_hand_derived_site_names(six_combos):
    df = six_combos.show_solutions_summary(diff_against="rank_1")
    # Rank 2 is {Site_A, Site_C} vs rank 1's {Site_B, Site_D} -- every site differs.
    rank_2_row = df[df["Rank"] == 2].iloc[0]
    assert set(rank_2_row["Sites added (vs rank 1)"].split(", ")) == {"Site_A", "Site_C"}
    assert set(rank_2_row["Sites removed (vs rank 1)"].split(", ")) == {"Site_B", "Site_D"}
    assert rank_2_row["Sites changed (vs rank 1)"] == 4


def test_previous_rank_first_row_has_zero_diff(six_combos):
    df = six_combos.show_solutions_summary(diff_against="previous_rank")
    rank_1_row = df[df["Rank"] == 1].iloc[0]
    assert rank_1_row["Sites added (vs previous rank)"] == ""
    assert rank_1_row["Sites removed (vs previous rank)"] == ""
    assert rank_1_row["Sites changed (vs previous rank)"] == 0


def test_previous_rank_diffs_each_row_against_the_one_above_it(six_combos):
    df = six_combos.show_solutions_summary(diff_against="previous_rank")
    raw = six_combos.show_solutions(rounding=None)

    for i in range(1, len(df)):
        current_sites = set(raw.iloc[i]["site_names"])
        previous_sites = set(raw.iloc[i - 1]["site_names"])
        row = df.iloc[i]
        assert set(
            s for s in row["Sites added (vs previous rank)"].split(", ") if s
        ) == current_sites - previous_sites
        assert set(
            s for s in row["Sites removed (vs previous rank)"].split(", ") if s
        ) == previous_sites - current_sites


def test_required_sites_used_by_default_when_configured(four_site_problem_with_required):
    result = four_site_problem_with_required.solve(
        p=2, search_strategy="brute-force", show_progress=False
    )
    df = result.show_solutions_summary()
    assert "Sites added (vs required sites)" in df.columns
    assert "Sites removed (vs required sites)" in df.columns
    assert "Sites changed (vs required sites)" in df.columns
    # Every combination includes required Site_A (via required_sites_col),
    # so it should never appear in "added" or "removed" for any row.
    assert not df["Sites added (vs required sites)"].str.contains("Site_A").any()
    assert not df["Sites removed (vs required sites)"].str.contains("Site_A").any()


def test_required_sites_explicit_raises_when_none_configured(six_combos):
    with pytest.raises(ValueError, match="requires at least one site"):
        six_combos.show_solutions_summary(diff_against="required_sites")


def test_invalid_diff_against_raises(six_combos):
    with pytest.raises(ValueError, match="diff_against must be one of"):
        six_combos.show_solutions_summary(diff_against="sideways")


def test_single_solution_result_has_no_diff_columns(four_site_problem):
    baseline = four_site_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C", "Site_D"]
    )
    df = baseline.show_solutions_summary()
    assert not any(col.startswith("Sites added") for col in df.columns)
    assert not any(col.startswith("Sites removed") for col in df.columns)
    assert not any(col.startswith("Sites changed") for col in df.columns)


def test_diff_against_default_is_the_default_parameter_value(six_combos):
    explicit = six_combos.show_solutions_summary(diff_against="default")
    implicit = six_combos.show_solutions_summary()
    assert list(explicit.columns) == list(implicit.columns)
