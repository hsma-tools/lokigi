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

import warnings

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
    result = comparator.population_impact_summary(as_dict=True)

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
    v1 = SolutionComparator(devon_baseline, candidate).population_impact_summary(as_dict=True)

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
    """solve() with no baseline and no beyond_thresholds must produce
    exactly the same solution_df column set as before those features
    existed, plus the always-on absolute coverage headcounts (demand_
    within_coverage_threshold / regions_within_coverage_threshold),
    unselected_site_names (always present), and additional_site_names
    (devon_problem registers required_sites_col="Existing", so this is
    always present too) -- no stray baseline- or beyond_thresholds-only
    columns leaking in."""
    solved = devon_problem.solve(p=5, threshold_for_coverage=30, show_progress=False)
    expected_columns = {
        "solution_rank",
        "site_names",
        "site_indices",
        "unselected_site_names",
        "additional_site_names",
        "coverage_threshold",
        "weighted_average",
        "unweighted_average",
        "90th_percentile",
        "max",
        "total_cost",
        "proportion_within_coverage_threshold",
        "proportion_regions_within_coverage_threshold",
        "demand_within_coverage_threshold",
        "regions_within_coverage_threshold",
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

    result = SolutionComparator(baseline, candidate).population_impact_summary(as_dict=True)

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

    forward = SolutionComparator(baseline, candidate).population_impact_summary(as_dict=True)
    reversed_ = SolutionComparator(candidate, baseline).population_impact_summary(as_dict=True)

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
        return_per_region=True, as_dict=True
    )

    assert per_region.loc["D1", "bucket"] == "unchanged"
    assert per_region.loc["D2", "bucket"] == "worsened"
    assert per_region.loc["D3", "bucket"] == "improved"
    assert per_region.loc["D2", "delta"] == pytest.approx(40)
    assert per_region.loc["D3", "delta"] == pytest.approx(-15)
    assert (per_region["bucket"] == "improved").sum() == summary["regions_improved"]
    assert (per_region["bucket"] == "worsened").sum() == summary["regions_worsened"]


# --- population_impact_summary DataFrame default / as_dict / native types ---


def test_population_impact_summary_default_returns_dataframe():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    result = SolutionComparator(baseline, candidate).population_impact_summary()

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["value"]
    assert result.index.name == "metric"
    assert result.loc["regions_improved", "value"] == 1
    assert result.loc["demand_improved", "value"] == pytest.approx(200)
    assert result.loc["mean_reduction_among_improved", "value"] == pytest.approx(15)


def test_population_impact_summary_dataframe_keeps_region_counts_as_ints():
    """Region counts must not be silently upcast to float64 (e.g. "61.0")
    just because other metrics in the same column are floats -- the
    object-dtype column keeps each metric's own native type."""
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    result = SolutionComparator(baseline, candidate).population_impact_summary()

    assert isinstance(result.loc["regions_improved", "value"], int)
    assert isinstance(result.loc["demand_improved", "value"], float)


def test_population_impact_summary_as_dict_true_returns_plain_dict_with_native_types():
    """Regression guard for numpy>=2.0's np.float64(...) repr wrapper --
    as_dict=True must hand back native Python floats/ints, not numpy
    scalar types, so a bare dict repr in a notebook cell stays readable."""
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    result = SolutionComparator(baseline, candidate).population_impact_summary(as_dict=True)

    assert isinstance(result, dict)
    assert type(result["regions_improved"]) is int
    assert type(result["demand_improved"]) is float
    assert "np.float64" not in repr(result)


def test_population_impact_summary_return_per_region_with_dataframe_default():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    summary, per_region = SolutionComparator(baseline, candidate).population_impact_summary(
        return_per_region=True
    )

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(per_region, pd.DataFrame)
    assert summary.loc["regions_improved", "value"] == 1


# --- population_impact_phrase ---


def test_population_impact_phrase_superset_omits_worsened_clause():
    """The common superset case (add a site, keep every existing one) has
    demand_worsened == 0 -- the phrase must not include a "0 people ...,
    averaging nan minutes more" clause for it."""
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A"])
    candidate = problem.evaluate_baseline(site_names=["A", "B", "C"])

    phrase = SolutionComparator(baseline, candidate).population_impact_phrase()

    assert "shorter journey" in phrase
    assert "longer journey" not in phrase
    assert "nan" not in phrase
    assert "regions improved" in phrase


def test_population_impact_phrase_includes_both_clauses_when_both_nonzero():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    phrase = SolutionComparator(baseline, candidate).population_impact_phrase()

    assert "200 people" in phrase
    assert "shorter journey" in phrase
    assert "50 people" in phrase
    assert "longer journey" in phrase
    assert "1 of 3 regions improved; 1 worsened; 1 unchanged." in phrase


def test_population_impact_phrase_devon_matches_headline_numbers(
    devon_problem, devon_baseline, devon_existing_sites
):
    candidate = devon_problem.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
    )
    phrase = SolutionComparator(devon_baseline, candidate).population_impact_phrase()

    assert "46,907 people" in phrase
    assert "16.1 minutes off" in phrase
    assert "61 of 729 regions improved; 0 worsened; 668 unchanged." in phrase


# --- meaningful_change_threshold ---


def test_meaningful_change_threshold_at_exact_boundary_is_unchanged():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])

    # D3's delta is exactly -15. A threshold of 15 must NOT count it as
    # improved (strictly-greater-than is required).
    at_boundary = SolutionComparator(baseline, candidate).population_impact_summary(
        meaningful_change_threshold=15, as_dict=True
    )
    assert at_boundary["regions_improved"] == 0
    assert at_boundary["regions_unchanged"] == 2  # D1 and D3
    assert at_boundary["regions_worsened"] == 1  # D2 (delta +40) still worsened

    just_under = SolutionComparator(baseline, candidate).population_impact_summary(
        meaningful_change_threshold=14.9, as_dict=True
    )
    assert just_under["regions_improved"] == 1


def test_default_threshold_absorbs_floating_point_noise():
    current = np.array([5.0 + 1e-10])
    baseline = np.array([5.0])
    result = _population_impact_metrics(current, baseline, demand=None)
    assert result["regions_improved"] == 0
    assert result["regions_unchanged"] == 1


def test_population_impact_metrics_returns_native_python_types_not_numpy_scalars():
    current = np.array([1.0, 5.0])
    baseline = np.array([5.0, 5.0])
    demand = np.array([10.0, 20.0])
    result = _population_impact_metrics(current, baseline, demand=demand)

    float_keys = [
        "demand_improved",
        "demand_worsened",
        "demand_unchanged",
        "proportion_demand_improved",
        "proportion_demand_worsened",
        "total_demand",
        "mean_reduction_among_improved",
        "mean_increase_among_worsened",
        "max_reduction",
        "max_increase",
    ]
    for key in float_keys:
        assert type(result[key]) is float, key


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

    default = comparator.population_impact_summary(as_dict=True)
    assert default["regions_improved"] == 2
    assert default["regions_unchanged"] == 1
    assert default["regions_worsened"] == 0
    assert default["demand_improved"] == pytest.approx(350)  # 200 + 150
    assert default["mean_reduction_among_improved"] == pytest.approx(15)
    assert default["max_reduction"] == pytest.approx(15)

    via_matrix = comparator.population_impact_summary(matrix="public_transport", as_dict=True)
    assert via_matrix["regions_improved"] == 3
    assert via_matrix["regions_worsened"] == 0
    assert via_matrix["mean_reduction_among_improved"] == pytest.approx(40)
    assert via_matrix["max_reduction"] == pytest.approx(40)

    via_demand = comparator.population_impact_summary(demand="future_demand", as_dict=True)
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


# --- Metric C: absolute coverage headcounts + total_demand -----------------


def test_total_demand_property(five_site_problem):
    assert five_site_problem.total_demand == pytest.approx(400.0)


def test_total_demand_none_without_demand_data():
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame({"source_id": ["D1"], "Site_A": [5.0]})
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    assert problem.total_demand is None


def test_demand_within_coverage_threshold_matches_proportion_times_total_demand(
    devon_problem, devon_existing_sites
):
    """demand_within_coverage_threshold is explicitly NOT meant to be an
    independent measurement -- it's the same proportion_within_coverage_
    threshold, just in headcount units."""
    result = devon_problem.evaluate_baseline(
        site_names=devon_existing_sites, threshold_for_coverage=30
    )
    row = result.solution_df.iloc[0]
    assert row["demand_within_coverage_threshold"] == pytest.approx(370510, abs=1)
    assert row["proportion_within_coverage_threshold"] == pytest.approx(0.7132146, abs=1e-5)
    assert row["demand_within_coverage_threshold"] / devon_problem.total_demand == pytest.approx(
        row["proportion_within_coverage_threshold"]
    )


def test_coverage_headcounts_nan_without_threshold(five_site_problem):
    result = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )
    metrics = result.return_solution_metrics()
    assert pd.isna(metrics["demand_within_coverage_threshold"])
    assert pd.isna(metrics["proportion_within_coverage_threshold"])


def test_regions_within_coverage_threshold_counts_regions_not_demand():
    """4 regions, one with much higher demand than the rest -- regions_
    within_coverage_threshold must count regions equally regardless of
    demand, unlike the demand-weighted headcount."""
    demand_df = pd.DataFrame(
        {"location_id": ["D1", "D2", "D3", "D4"], "demand": [10, 10, 10, 1000]}
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]}
    )
    travel_df = pd.DataFrame(
        {"source_id": ["D1", "D2", "D3", "D4"], "Site_A": [5.0, 5.0, 25.0, 25.0]}
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0], threshold_for_coverage=10
    )
    metrics = result.return_solution_metrics()
    assert metrics["regions_within_coverage_threshold"] == 2
    assert metrics["demand_within_coverage_threshold"] == pytest.approx(20.0)


def _coverage_transition_problem():
    """Baseline = {A}, candidate = {A, B}, threshold=10: D1 stays covered
    (unaffected by B), D2 gains coverage (was 20 via A, now 5 via B), D3
    loses coverage relative to a wider network story is impossible here
    (adding a site can only ever help), so a genuine BOTH-directions test
    needs baseline={A} vs candidate={B} (a swap, not a superset): D1
    (demand=100) goes from covered (A=5) to uncovered (B=20) -- a real
    "newly uncovered" case; D2 (demand=50) goes from uncovered (A=20) to
    covered (B=5) -- "newly covered"; D3 (demand=200) stays uncovered
    either way (A=30, B=25, threshold=10)."""
    demand_df = pd.DataFrame(
        {"location_id": ["D1", "D2", "D3"], "demand": [100, 50, 200]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["A", "B"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["D1", "D2", "D3"],
            "A": [5.0, 20.0, 30.0],
            "B": [20.0, 5.0, 25.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def test_gross_coverage_transitions_both_directions():
    problem = _coverage_transition_problem()
    baseline = problem.evaluate_baseline(site_names=["A"], threshold_for_coverage=10)
    candidate = problem.evaluate_baseline(site_names=["B"], threshold_for_coverage=10)

    impact = SolutionComparator(baseline, candidate).population_impact_summary(as_dict=True)

    assert impact["regions_newly_covered"] == 1
    assert impact["regions_newly_uncovered"] == 1
    assert impact["demand_newly_covered"] == pytest.approx(50.0)
    assert impact["demand_newly_uncovered"] == pytest.approx(100.0)


def test_gross_coverage_transitions_absent_without_threshold():
    problem = _coverage_transition_problem()
    baseline = problem.evaluate_baseline(site_names=["A"])
    candidate = problem.evaluate_baseline(site_names=["B"])

    impact = SolutionComparator(baseline, candidate).population_impact_summary(as_dict=True)

    assert "regions_newly_covered" not in impact
    assert "demand_newly_covered" not in impact


# --- Metric B: "left behind" headcounts beyond one or more thresholds ------


def test_devon_beyond_thresholds_matches_repro(devon_problem, devon_existing_sites):
    """Golden regression matching notes/2026-07-29-handover-additional-
    metrics.md's repro exactly: baseline (existing 4 sites) vs candidate
    (existing + Barnstaple), at 30/45/60 minutes. The >60min tier is
    deliberately unchanged by this candidate -- that's the actual finding,
    not a bug in the fixture."""
    baseline = devon_problem.evaluate_baseline(
        threshold_for_coverage=30, beyond_thresholds=[30, 45, 60]
    )
    candidate = devon_problem.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
        beyond_thresholds=[30, 45, 60],
    )
    base_row = baseline.solution_df.iloc[0]
    cand_row = candidate.solution_df.iloc[0]

    assert base_row["demand_beyond_threshold_30"] == pytest.approx(148983, abs=1)
    assert base_row["demand_beyond_threshold_45"] == pytest.approx(52145, abs=1)
    assert base_row["demand_beyond_threshold_60"] == pytest.approx(2049, abs=1)
    assert base_row["regions_beyond_threshold_30"] == 171
    assert base_row["regions_beyond_threshold_45"] == 58
    assert base_row["regions_beyond_threshold_60"] == 2

    assert cand_row["demand_beyond_threshold_30"] == pytest.approx(127670, abs=1)
    assert cand_row["demand_beyond_threshold_45"] == pytest.approx(39798, abs=1)
    # The extreme tail is untouched by this candidate.
    assert cand_row["demand_beyond_threshold_60"] == pytest.approx(2049, abs=1)
    assert cand_row["regions_beyond_threshold_30"] == 145
    assert cand_row["regions_beyond_threshold_45"] == 43
    assert cand_row["regions_beyond_threshold_60"] == 2


def test_beyond_thresholds_scalar_equals_single_element_list(five_site_problem):
    scalar = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=20
    ).return_solution_metrics()
    as_list = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=[20]
    ).return_solution_metrics()

    assert scalar["demand_beyond_threshold_20"] == as_list["demand_beyond_threshold_20"]
    assert scalar["regions_beyond_threshold_20"] == as_list["regions_beyond_threshold_20"]


def test_beyond_threshold_column_naming_for_float_threshold(five_site_problem):
    metrics = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=[42.5]
    ).return_solution_metrics()
    assert "demand_beyond_threshold_42.5" in metrics
    assert "regions_beyond_threshold_42.5" in metrics


def test_beyond_thresholds_absent_by_default(five_site_problem):
    metrics = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    ).return_solution_metrics()
    assert not any(key.startswith("demand_beyond_threshold_") for key in metrics)


def test_nan_cost_counts_as_beyond_every_threshold():
    """A demand location with no reachable site (NaN min_cost) must count
    as beyond every threshold -- consistent with within_threshold's own
    "NaN cost -> not covered" convention, just applied to the opposite
    (bad) direction. Built directly via EvaluatedCombination (bypassing
    evaluate_single_solution_single_objective()) since a genuinely all-NaN
    row for the selected sites isn't reachable through the public solve
    path -- pandas' idxmin() raises before EvaluatedCombination is ever
    constructed in that case."""

    class _StubSiteProblem:
        _demand_data_demand_col = "demand"

    df = pd.DataFrame(
        {
            "min_cost": [5.0, np.nan],
            "demand": [100, 100],
            "within_threshold": [True, np.nan],
        }
    )
    result = EvaluatedCombination(
        solution_type="p_median",
        site_names=["A"],
        site_indices=[0],
        evaluated_combination_df=df,
        weights=None,
        site_problem=_StubSiteProblem(),
        beyond_thresholds=[100],
    )
    metrics = result.return_solution_metrics()

    assert metrics["regions_beyond_threshold_100"] == 1
    assert metrics["demand_beyond_threshold_100"] == pytest.approx(100.0)


def test_beyond_threshold_by_equity_group_present_iff_equity_data(
    five_site_problem,
):
    without_equity = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=[20]
    ).return_solution_metrics()
    assert "demand_beyond_threshold_20_by_equity_group" not in without_equity

    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [1, 5, 10, 10],
        }
    )
    five_site_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    with_equity = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=[20]
    ).return_solution_metrics()
    assert "demand_beyond_threshold_20_by_equity_group" in with_equity
    assert "regions_beyond_threshold_20_by_equity_group" in with_equity


def test_beyond_threshold_is_minimise_metric_by_default():
    from lokigi.utils import _is_maximise_metric

    assert _is_maximise_metric("demand_beyond_threshold_30") is False
    assert _is_maximise_metric("regions_beyond_threshold_45") is False
    assert _is_maximise_metric("demand_beyond_threshold_30__public_transport") is False


# --- Metric A: population impact split by equity band -----------------------


@pytest.fixture(scope="module")
def devon_problem_with_equity(devon_problem):
    """`devon_problem` with IMD decile equity data registered on a deep
    copy -- kept separate from the plain `devon_problem` fixture (also
    module-scoped and shared across many other tests in this file) so that
    registering equity data here can never leak into them."""
    problem = devon_problem.copy()
    imd = pd.read_csv("sample_data/devon_imd_2025_2021_LSOAs.csv")
    problem.add_equity_data(
        imd,
        equity_col=(
            "Index of Multiple Deprivation (IMD) Decile "
            "(where 1 is most deprived 10% of LSOA"
        ),
        common_col="LSOA name (2021)",
        label="IMD decile",
        disadvantaged_end="low",
    )
    return problem


def test_devon_population_impact_by_equity_group_golden_regression(
    devon_problem_with_equity, devon_existing_sites
):
    """Golden regression: 10 IMD deciles split via lokigi's own
    array_split-based tertile convention (chunks of 4/3/3, remainder in
    the first chunk) -- deciles 1-4 = most-deprived tertile, 8-10 =
    least-deprived. These are NOT the handover note's manually-computed
    1-3/8-10 thirds (a different, non-numpy split); they are the numbers
    lokigi's own bucketing convention actually produces, confirmed by
    direct computation against this exact fixture."""
    baseline = devon_problem_with_equity.evaluate_baseline(threshold_for_coverage=30)
    candidate = devon_problem_with_equity.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
    )
    comparator = SolutionComparator(baseline, candidate, labels=("baseline", "candidate"))

    by_band = comparator.population_impact_by_equity_group()

    assert list(by_band.index) == list(range(1, 11))
    assert by_band["demand_improved"].sum() == pytest.approx(46907, abs=1)
    assert by_band["band_total_demand"].sum() == pytest.approx(519493, abs=1)

    lower_third = by_band.iloc[:4]  # deciles 1-4
    upper_third = by_band.iloc[-3:]  # deciles 8-10
    lower_rate = lower_third["demand_improved"].sum() / lower_third["band_total_demand"].sum()
    upper_rate = upper_third["demand_improved"].sum() / upper_third["band_total_demand"].sum()

    assert lower_rate == pytest.approx(0.12156, abs=1e-4)
    assert upper_rate == pytest.approx(0.04205, abs=1e-4)
    # The qualitative finding the handover note surfaced: the most
    # deprived tertile benefits at roughly double (here, even higher) the
    # rate of the least deprived tertile.
    assert lower_rate > 2 * upper_rate


def test_population_impact_by_equity_group_no_equity_data_raises(
    devon_problem, devon_existing_sites
):
    baseline = devon_problem.evaluate_baseline(threshold_for_coverage=30)
    candidate = devon_problem.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
    )
    comparator = SolutionComparator(baseline, candidate)
    with pytest.raises(ValueError, match="add_equity_data"):
        comparator.population_impact_by_equity_group()


def test_population_impact_by_equity_group_respects_n_bins():
    """A caller who registered 5 bins should get 5-band population-impact
    numbers, not a silently different (e.g. hardcoded decile) granularity."""
    problem = _swap_problem()
    equity_df = pd.DataFrame(
        {"location_id": ["D1", "D2", "D3"], "score": [1.0, 50.0, 99.0]}
    )
    problem.add_equity_data(
        equity_df,
        equity_col="score",
        common_col="location_id",
        label="score",
        disadvantaged_end="low",
        continuous_measure=True,
        n_bins=3,
    )
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])
    comparator = SolutionComparator(baseline, candidate)

    by_band = comparator.population_impact_by_equity_group()
    assert len(by_band) == 3


def test_population_impact_by_equity_group_totals_match_summary():
    """Per-band demand_improved/worsened/unchanged must sum to the same
    totals population_impact_summary() reports region-wide."""
    problem = _swap_problem()
    equity_df = pd.DataFrame(
        {"location_id": ["D1", "D2", "D3"], "imd_decile": [1, 5, 10]}
    )
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])
    comparator = SolutionComparator(baseline, candidate)

    summary = comparator.population_impact_summary(as_dict=True)
    by_band = comparator.population_impact_by_equity_group()

    assert by_band["demand_improved"].sum() == pytest.approx(summary["demand_improved"])
    assert by_band["demand_worsened"].sum() == pytest.approx(summary["demand_worsened"])
    assert by_band["demand_unchanged"].sum() == pytest.approx(summary["demand_unchanged"])
    assert by_band["regions_improved"].sum() == summary["regions_improved"]
    assert by_band["regions_worsened"].sum() == summary["regions_worsened"]


def test_solve_path_equity_group_population_impact_columns(five_site_problem):
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [1, 5, 10, 10],
        }
    )
    five_site_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    baseline = five_site_problem.evaluate_baseline(site_names=["Site_1"])
    solved = five_site_problem.solve(
        p=2, baseline=baseline, show_progress=False
    )
    assert "demand_improved_by_equity_group" in solved.solution_df.columns
    assert "demand_worsened_by_equity_group" in solved.solution_df.columns
    first_row_bands = solved.solution_df.iloc[0]["demand_improved_by_equity_group"]
    assert isinstance(first_row_bands, dict)
    assert set(first_row_bands.keys()) <= {1, 5, 10}


def test_solve_path_equity_group_columns_absent_without_equity_data(five_site_problem):
    baseline = five_site_problem.evaluate_baseline(site_names=["Site_1"])
    solved = five_site_problem.solve(p=2, baseline=baseline, show_progress=False)
    assert "demand_improved_by_equity_group" not in solved.solution_df.columns


def test_population_impact_phrase_includes_threshold_wording():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"])
    candidate = problem.evaluate_baseline(site_names=["A", "C"])
    phrase = SolutionComparator(baseline, candidate).population_impact_phrase(
        meaningful_change_threshold=5.0
    )
    assert "shorter by more than 5.0 minutes" in phrase


def test_population_impact_phrase_includes_equity_sentence(
    devon_problem_with_equity, devon_existing_sites
):
    baseline = devon_problem_with_equity.evaluate_baseline(threshold_for_coverage=30)
    candidate = devon_problem_with_equity.evaluate_baseline(
        site_names=devon_existing_sites + ["Barnstaple - Archwood Retail Park"],
        threshold_for_coverage=30,
    )
    phrase = SolutionComparator(baseline, candidate).population_impact_phrase()
    assert "most disadvantaged third" in phrase
    assert "least disadvantaged third" in phrase


# --- Post-review fixes: disadvantaged_end ordering with <3 bands, numpy ----
# --- integer thresholds, and cross-side threshold_for_coverage mismatch ---


def _two_band_problem(disadvantaged_end):
    """Baseline={A}, candidate={B}: D1 (band 1) improves, D2 (band 2)
    worsens -- band 2 must sort first ("most disadvantaged") whenever
    disadvantaged_end="high", even though only 2 bands means no tertile
    split is possible."""
    demand_df = pd.DataFrame({"location_id": ["D1", "D2"], "demand": [100, 100]})
    candidate_df = pd.DataFrame(
        {"site_id": ["A", "B"], "lat": [51.1, 51.2], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {"source_id": ["D1", "D2"], "A": [5.0, 20.0], "B": [20.0, 5.0]}
    )
    equity_df = pd.DataFrame({"location_id": ["D1", "D2"], "band": [1, 2]})

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="band",
        common_col="location_id",
        label="band",
        disadvantaged_end=disadvantaged_end,
    )
    return problem


def test_population_impact_by_equity_group_orders_two_bands_by_disadvantaged_end_high():
    """Regression: with fewer than 3 distinct bands (no tertile split
    possible), the method previously fell back to plain ascending raw-bin
    order regardless of disadvantaged_end, silently breaking the documented
    "most- to least-disadvantaged" ordering for disadvantaged_end="high"."""
    problem = _two_band_problem(disadvantaged_end="high")
    baseline = problem.evaluate_baseline(site_names=["A"])
    candidate = problem.evaluate_baseline(site_names=["B"])
    by_band = SolutionComparator(baseline, candidate).population_impact_by_equity_group()
    assert list(by_band.index) == [2, 1]


def test_population_impact_by_equity_group_orders_two_bands_by_disadvantaged_end_low():
    """Same 2-band fixture with disadvantaged_end="low" (the default
    assumption) should keep ascending order -- band 1 first."""
    problem = _two_band_problem(disadvantaged_end="low")
    baseline = problem.evaluate_baseline(site_names=["A"])
    candidate = problem.evaluate_baseline(site_names=["B"])
    by_band = SolutionComparator(baseline, candidate).population_impact_by_equity_group()
    assert list(by_band.index) == [1, 2]


def test_beyond_thresholds_accepts_numpy_integer_scalar(five_site_problem):
    """Regression: np.integer doesn't subclass int (unlike np.floating,
    which subclasses float), so a numpy-integer threshold -- e.g. derived
    from a column's .max() -- previously fell through to the "treat as an
    iterable" branch and crashed with "'numpy.int64' object is not
    iterable"."""
    metrics = five_site_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1], beyond_thresholds=np.int64(20)
    ).return_solution_metrics()
    assert "demand_beyond_threshold_20" in metrics


def test_population_impact_summary_warns_on_mismatched_coverage_thresholds():
    """Regression: comparing two solutions evaluated with different
    threshold_for_coverage values would silently diff 'covered' flags
    computed against two different cutoffs for the gross newly-covered/
    newly-uncovered metrics -- a misleading result with no signal to the
    caller that anything was wrong."""
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"], threshold_for_coverage=15)
    candidate = problem.evaluate_baseline(site_names=["A", "C"], threshold_for_coverage=45)

    with pytest.warns(UserWarning, match="different threshold_for_coverage"):
        SolutionComparator(baseline, candidate).population_impact_summary()


def test_population_impact_summary_no_warning_on_matching_coverage_thresholds():
    problem = _swap_problem()
    baseline = problem.evaluate_baseline(site_names=["A", "B"], threshold_for_coverage=15)
    candidate = problem.evaluate_baseline(site_names=["A", "C"], threshold_for_coverage=15)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SolutionComparator(baseline, candidate).population_impact_summary()
