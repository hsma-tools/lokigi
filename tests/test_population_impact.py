"""
Tests for the population-impact-vs-baseline feature: how many people's
journey actually changed relative to a baseline, and by how much, rather
than only the region-wide `weighted_average` shift.

Covers the shared arithmetic (`lokigi.utils._population_impact_metrics`),
v1 (`SiteProblem.evaluate_baseline()` +
`SolutionComparator.population_impact_summary()`), and v2
(`solve(baseline=...)`), including that the two paths agree exactly on the
same data.
"""

import numpy as np
import pandas as pd
import pytest

import lokigi
from lokigi.site_solutions import EvaluatedCombination, SiteSolutionSet, SolutionComparator
from lokigi.utils import _population_impact_metrics


# --- Devon fixture: real regression data, matching notes/2026-07-29-
# alternative-headline-metric-plan.md's repro exactly ---


@pytest.fixture(scope="module")
def devon_problem():
    """729 LSOAs, cohort MF50-84, 4 existing sites flagged as required via
    the 'Existing' column, 18 candidate sites total. p=5 with 4 required
    sites is only C(14,1)=14 brute-force combinations -- fast enough for
    the default test run."""
    sites_df = pd.read_csv("sample_data/devon_cdcs.csv")
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(
        sites_df,
        candidate_id_col="Facility_Name",
        vertical_geometry_col="Latitude",
        horizontal_geometry_col="Longitude",
        required_sites_col="Existing",
    )
    problem.add_travel_matrix(
        travel_matrix_df="sample_data/travel_matrix_car_devon_cdcs.csv",
        source_col="from_id",
        unit="minutes",
    )
    problem.add_demand(
        "sample_data/demand_MF_50_84.csv",
        demand_col="MF50-84",
        location_id_col="LSOA 2021 Name",
    )
    return problem


@pytest.fixture(scope="module")
def devon_existing_sites(devon_problem):
    sites_df = pd.read_csv("sample_data/devon_cdcs.csv")
    return sites_df[sites_df["Existing"] == "Yes"]["Facility_Name"].tolist()


@pytest.fixture(scope="module")
def devon_baseline(devon_problem):
    return devon_problem.evaluate_baseline(threshold_for_coverage=30)


# --- Golden regression: exact figures from the handover note's repro ---


def test_devon_golden_regression_v1(devon_problem, devon_baseline, devon_existing_sites):
    candidate = devon_problem.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
    )
    comparator = SolutionComparator(devon_baseline, candidate, labels=("Current", "Proposed"))
    result = comparator.population_impact_summary()

    assert devon_baseline.solution_df.iloc[0]["weighted_average"] == pytest.approx(23.39, abs=0.01)
    assert candidate.solution_df.iloc[0]["weighted_average"] == pytest.approx(21.93, abs=0.01)

    assert result["regions_improved"] == 61
    assert result["regions_worsened"] == 0
    assert result["regions_unchanged"] == 668
    assert result["demand_improved"] == pytest.approx(46907)
    assert result["demand_worsened"] == pytest.approx(0)
    assert result["demand_unchanged"] == pytest.approx(472586)
    assert result["mean_reduction_among_improved"] == pytest.approx(16.15, abs=0.01)
    assert result["max_reduction"] == pytest.approx(20.50, abs=0.01)


def test_devon_golden_regression_v2(devon_problem, devon_baseline, devon_existing_sites):
    solved = devon_problem.solve(
        p=5,
        threshold_for_coverage=30,
        baseline=devon_baseline,
        show_progress=False,
    )
    candidate_sites = set(devon_existing_sites + ["Barnstaple - Archwood Retail Park"])
    row = solved.solution_df[
        solved.solution_df["site_names"].apply(lambda names: set(names) == candidate_sites)
    ].iloc[0]

    assert row["regions_improved"] == 61
    assert row["regions_worsened"] == 0
    assert row["regions_unchanged"] == 668
    assert row["demand_improved"] == pytest.approx(46907)
    assert row["mean_reduction_among_improved"] == pytest.approx(16.15, abs=0.01)
    assert row["max_reduction"] == pytest.approx(20.50, abs=0.01)


def test_v1_and_v2_agree_exactly(devon_problem, devon_baseline, devon_existing_sites):
    """The two paths share the same underlying _population_impact_metrics
    call, so they must produce byte-identical numbers on the same data --
    not just approximately equal."""
    candidate_sites = devon_existing_sites + ["Barnstaple - Archwood Retail Park"]
    candidate = devon_problem.evaluate_baseline(
        site_names=candidate_sites, threshold_for_coverage=30
    )
    v1 = SolutionComparator(devon_baseline, candidate).population_impact_summary()

    solved = devon_problem.solve(
        p=5, threshold_for_coverage=30, baseline=devon_baseline, show_progress=False
    )
    v2_row = solved.solution_df[
        solved.solution_df["site_names"].apply(lambda names: set(names) == set(candidate_sites))
    ].iloc[0]

    for key in [
        "demand_improved",
        "demand_worsened",
        "demand_unchanged",
        "regions_improved",
        "regions_worsened",
        "regions_unchanged",
        "mean_reduction_among_improved",
        "mean_increase_among_worsened",
        "max_reduction",
        "max_increase",
    ]:
        v2_value = v2_row[key]
        if isinstance(v2_value, float) and np.isnan(v2_value):
            assert v1[key] != v1[key] or np.isnan(v1[key])
        else:
            assert v1[key] == v2_value, key


def test_solve_baseline_true_matches_explicit_baseline(devon_problem, devon_baseline):
    explicit = devon_problem.solve(
        p=5, threshold_for_coverage=30, baseline=devon_baseline, show_progress=False
    )
    implicit = devon_problem.solve(
        p=5, threshold_for_coverage=30, baseline=True, show_progress=False
    )
    assert (
        explicit.solution_df.sort_values("site_indices", key=lambda s: s.astype(str))[
            "demand_improved"
        ].tolist()
        == implicit.solution_df.sort_values("site_indices", key=lambda s: s.astype(str))[
            "demand_improved"
        ].tolist()
    )


def test_solve_without_baseline_is_purely_additive(devon_problem):
    """solve() with no baseline must produce exactly the same solution_df
    column set as before this feature existed -- no stray always-on
    columns leaking in."""
    solved = devon_problem.solve(p=5, threshold_for_coverage=30, show_progress=False)
    expected_columns = {
        "solution_rank",
        "site_names",
        "site_indices",
        "coverage_threshold",
        "weighted_average",
        "unweighted_average",
        "90th_percentile",
        "max",
        "total_cost",
        "proportion_within_coverage_threshold",
        "proportion_regions_within_coverage_threshold",
        "weighted_by_equity_group",
        "unweighted_by_equity_group",
        "coverage_by_equity_group",
        "coverage_regions_by_equity_group",
        "max_cost_by_equity_group",
        "gap_absolute_weighted",
        "gap_relative_weighted",
        "avg_lower_third_bins",
        "avg_middle_third_bins",
        "avg_upper_third_bins",
        "inter_tertile_ratio",
        "gap_absolute_description",
        "gap_relative_description",
        "inter_tertile_description",
        "problem_df",
    }
    assert set(solved.solution_df.columns) == expected_columns


# --- Non-superset comparison: both improved AND worsened buckets non-zero ---


def _swap_problem():
    """3 demand points, 3 sites (A/B/C). Baseline = {A, B}, candidate =
    {A, C} -- a genuine swap, not a superset, so the improved/worsened
    split is hand-computable in both directions:

      D1 (demand=100): A=10, B=50, C=50 -> baseline picks A (10),
        candidate also picks A (10) -- unchanged.
      D2 (demand=50):  A=50, B=10, C=50 -> baseline picks B (10),
        candidate picks A (50) -- worsened by 40.
      D3 (demand=200): A=50, B=20, C=5  -> baseline picks B (20),
        candidate picks C (5) -- improved by 15.
    """
    demand_df = pd.DataFrame(
        {"location_id": ["D1", "D2", "D3"], "demand": [100, 50, 200]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["D1", "D2", "D3"],
            "A": [10.0, 50.0, 50.0],
            "B": [50.0, 10.0, 20.0],
            "C": [50.0, 50.0, 5.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def test_non_superset_comparison_has_both_buckets_populated():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    result = SolutionComparator(baseline, candidate).population_impact_summary()

    assert result["regions_improved"] == 1
    assert result["regions_worsened"] == 1
    assert result["regions_unchanged"] == 1
    assert result["demand_improved"] == pytest.approx(200)
    assert result["demand_worsened"] == pytest.approx(50)
    assert result["demand_unchanged"] == pytest.approx(100)
    assert result["mean_reduction_among_improved"] == pytest.approx(15)
    assert result["mean_increase_among_worsened"] == pytest.approx(40)
    assert result["max_reduction"] == pytest.approx(15)
    assert result["max_increase"] == pytest.approx(40)
    assert result["total_demand"] == pytest.approx(350)
    assert result["proportion_demand_improved"] == pytest.approx(200 / 350)
    assert result["proportion_demand_worsened"] == pytest.approx(50 / 350)


def test_reversing_set_a_and_set_b_swaps_the_buckets_exactly():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    forward = SolutionComparator(baseline, candidate).population_impact_summary()
    reversed_ = SolutionComparator(candidate, baseline).population_impact_summary()

    assert reversed_["demand_improved"] == forward["demand_worsened"]
    assert reversed_["demand_worsened"] == forward["demand_improved"]
    assert reversed_["regions_improved"] == forward["regions_worsened"]
    assert reversed_["regions_worsened"] == forward["regions_improved"]
    assert reversed_["mean_reduction_among_improved"] == forward["mean_increase_among_worsened"]
    assert reversed_["mean_increase_among_worsened"] == forward["mean_reduction_among_improved"]


def test_return_per_region_matches_summary_buckets():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    summary, per_region = SolutionComparator(baseline, candidate).population_impact_summary(
        return_per_region=True
    )

    assert per_region.loc["D1", "bucket"] == "unchanged"
    assert per_region.loc["D2", "bucket"] == "worsened"
    assert per_region.loc["D3", "bucket"] == "improved"
    assert per_region.loc["D2", "delta"] == pytest.approx(40)
    assert per_region.loc["D3", "delta"] == pytest.approx(-15)
    assert (per_region["bucket"] == "improved").sum() == summary["regions_improved"]
    assert (per_region["bucket"] == "worsened").sum() == summary["regions_worsened"]


# --- meaningful_change_threshold ---


def test_meaningful_change_threshold_at_exact_boundary_is_unchanged():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    # D3's delta is exactly -15. A threshold of 15 must NOT count it as
    # improved (strictly-greater-than is required).
    at_boundary = SolutionComparator(baseline, candidate).population_impact_summary(
        meaningful_change_threshold=15
    )
    assert at_boundary["regions_improved"] == 0
    assert at_boundary["regions_unchanged"] == 2  # D1 and D3
    assert at_boundary["regions_worsened"] == 1  # D2 (delta +40) still worsened

    just_under = SolutionComparator(baseline, candidate).population_impact_summary(
        meaningful_change_threshold=14.9
    )
    assert just_under["regions_improved"] == 1


def test_default_threshold_absorbs_floating_point_noise():
    current = np.array([5.0 + 1e-10])
    baseline = np.array([5.0])
    result = _population_impact_metrics(current, baseline, demand=None)
    assert result["regions_improved"] == 0
    assert result["regions_unchanged"] == 1


# --- Edge cases on the shared helper ---


def test_zero_demand_region_counts_in_regions_not_demand():
    current = np.array([5.0, 5.0])
    baseline = np.array([10.0, 10.0])
    demand = np.array([0.0, 100.0])
    result = _population_impact_metrics(current, baseline, demand=demand)
    assert result["regions_improved"] == 2
    assert result["demand_improved"] == pytest.approx(100.0)


def test_nan_demand_weight_propagates_to_demand_totals_not_region_counts():
    current = np.array([5.0, 5.0])
    baseline = np.array([10.0, 10.0])
    demand = np.array([np.nan, 100.0])
    result = _population_impact_metrics(current, baseline, demand=demand)
    assert result["regions_improved"] == 2
    assert np.isnan(result["demand_improved"])
    # NaN demand poisons the weighted mean for the bucket it's in too,
    # mirroring how the rest of the codebase (weighted_average etc.)
    # propagates rather than silently drops a NaN demand weight.
    assert np.isnan(result["mean_reduction_among_improved"])


def test_mismatched_demand_locations_between_sets_raises_clear_error():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    # Tamper with candidate's problem_df to drop a demand location, so it
    # no longer shares the exact same ID set as the baseline.
    tampered_metrics = dict(candidate.solution_df.iloc[0])
    tampered_metrics["problem_df"] = tampered_metrics["problem_df"].iloc[:-1].copy()
    tampered = SiteSolutionSet(
        solution_df=pd.DataFrame([tampered_metrics]),
        site_problem=problem,
        objectives="p_median",
        n_sites=2,
    )

    with pytest.raises(ValueError, match="different demand locations"):
        SolutionComparator(baseline, tampered).population_impact_summary()


def test_evaluated_combination_raises_on_missing_baseline_ids():
    problem = _swap_problem()
    problem.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["A", "B"]
    )
    bad_baseline = pd.Series([1.0, 2.0], index=["D1", "D2"])  # missing D3

    with pytest.raises(ValueError, match="missing"):
        problem.evaluate_single_solution_single_objective(
            objective="p_median",
            site_names=["A", "C"],
            baseline_costs={"min_cost": bad_baseline},
        )


# --- Secondary travel matrix / secondary demand scenario support ---


def test_population_impact_summary_with_secondary_matrix_and_demand(
    loaded_problem_with_secondary_demand_and_travel,
):
    problem = loaded_problem_with_secondary_demand_and_travel
    baseline = problem.evaluate_baseline(site_names=["Site_A"])
    candidate = problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    default = comparator.population_impact_summary()
    assert default["regions_improved"] == 2
    assert default["regions_unchanged"] == 1
    assert default["regions_worsened"] == 0
    assert default["demand_improved"] == pytest.approx(350)  # 200 + 150
    assert default["mean_reduction_among_improved"] == pytest.approx(15)
    assert default["max_reduction"] == pytest.approx(15)

    via_matrix = comparator.population_impact_summary(matrix="public_transport")
    assert via_matrix["regions_improved"] == 3
    assert via_matrix["regions_worsened"] == 0
    assert via_matrix["mean_reduction_among_improved"] == pytest.approx(40)
    assert via_matrix["max_reduction"] == pytest.approx(40)

    via_demand = comparator.population_impact_summary(demand="future_demand")
    # Region membership doesn't depend on demand weighting -- same regions
    # improved as the default primary-matrix comparison above.
    assert via_demand["regions_improved"] == default["regions_improved"]
    assert via_demand["regions_worsened"] == default["regions_worsened"]
    assert via_demand["demand_improved"] == pytest.approx(100)  # 50 + 50
    assert via_demand["mean_reduction_among_improved"] == pytest.approx(15)


def test_population_impact_summary_unknown_demand_label_raises(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A"])
    candidate = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    with pytest.raises(ValueError, match="Unknown secondary demand scenario"):
        SolutionComparator(baseline, candidate).population_impact_summary(
            demand="not_a_real_label"
        )


# --- evaluate_baseline() ---


def _required_sites_problem():
    demand_df = pd.DataFrame(
        {"location_id": ["D1", "D2"], "demand": [100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
            "existing": ["yes", "no", "no"],
        }
    )
    travel_df = pd.DataFrame(
        {"source_id": ["D1", "D2"], "A": [10.0, 20.0], "B": [15.0, 5.0], "C": [30.0, 30.0]}
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", required_sites_col="existing")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def test_evaluate_baseline_defaults_to_required_sites():
    problem = _required_sites_problem()
    baseline = problem.evaluate_baseline()
    assert baseline.solution_df.iloc[0]["site_names"] == ["A"]


def test_evaluate_baseline_site_names_and_site_indices_agree():
    problem = _required_sites_problem()
    by_name = problem.evaluate_baseline(site_names=["A", "B"])
    by_index = problem.evaluate_baseline(site_indices=[0, 1])
    assert by_name.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        by_index.solution_df.iloc[0]["weighted_average"]
    )


def test_evaluate_baseline_both_site_names_and_indices_raises():
    problem = _required_sites_problem()
    with pytest.raises(ValueError, match="but not both"):
        problem.evaluate_baseline(site_names=["A"], site_indices=[0])


def test_evaluate_baseline_no_required_sites_configured_raises(loaded_problem):
    with pytest.raises(ValueError, match="required_sites_col"):
        loaded_problem.evaluate_baseline()


def test_solve_baseline_true_without_required_sites_raises(loaded_problem):
    with pytest.raises(ValueError, match="required_sites_col"):
        loaded_problem.solve(p=2, baseline=True, show_progress=False)


def test_solve_baseline_invalid_type_raises(loaded_problem):
    with pytest.raises(TypeError, match="None, True, or a SiteSolutionSet"):
        loaded_problem.solve(p=2, baseline="not-a-valid-baseline", show_progress=False)


# --- Solver parity: brute-force n_jobs, greedy, grasp all carry the columns ---


def test_brute_force_n_jobs_parity_with_baseline(five_site_problem):
    baseline = five_site_problem.evaluate_baseline(site_names=["Site_1"])

    serial = five_site_problem.solve(
        p=2, search_strategy="brute-force", n_jobs=1, baseline=baseline, show_progress=False
    )
    parallel = five_site_problem.solve(
        p=2, search_strategy="brute-force", n_jobs=2, baseline=baseline, show_progress=False
    )

    serial_sorted = serial.solution_df.sort_values(
        "site_indices", key=lambda s: s.astype(str)
    ).reset_index(drop=True)
    parallel_sorted = parallel.solution_df.sort_values(
        "site_indices", key=lambda s: s.astype(str)
    ).reset_index(drop=True)

    for col in ["demand_improved", "demand_worsened", "regions_improved", "max_reduction"]:
        assert serial_sorted[col].tolist() == parallel_sorted[col].tolist()


def test_greedy_and_grasp_carry_population_impact_columns(five_site_problem):
    baseline = five_site_problem.evaluate_baseline(site_names=["Site_1"])

    greedy = five_site_problem.solve(
        p=2, search_strategy="greedy", baseline=baseline, show_progress=False
    )
    assert "demand_improved" in greedy.solution_df.columns
    assert greedy.solution_df["demand_improved"].notna().all()

    grasp = five_site_problem.solve(
        p=2,
        search_strategy="grasp",
        baseline=baseline,
        grasp_num_solutions=2,
        show_progress=False,
    )
    assert "demand_improved" in grasp.solution_df.columns
    assert grasp.solution_df["demand_improved"].notna().all()
