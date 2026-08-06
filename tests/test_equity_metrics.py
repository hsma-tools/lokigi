"""
End-to-end tests for the equity-band metrics computed inside
EvaluatedCombination (site_solutions.py:253-362) once equity data has been
registered via add_equity_data(). These were previously entirely untested --
no fixture exercised the equity code path at all.
"""

import pandas as pd
import pytest

import lokigi


def test_equity_band_gap_and_tertile_metrics(loaded_problem_with_equity):
    """With one demand location per equity band (deciles 1, 5, 10), each
    band's weighted average is just that location's own min_cost, making the
    gap/tertile arithmetic straightforward to hand-verify."""
    result = loaded_problem_with_equity.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group == {1: 10.0, 5: 5.0, 10: 15.0}

    assert result.gap_absolute_weighted == pytest.approx(10.0)
    assert "Spread of 10.0 units" in result.gap_absolute_desc

    assert result.gap_relative_weighted == pytest.approx(3.0)
    assert "Significant Disparity" in result.gap_relative_desc

    assert result.avg_lower_third_bins == pytest.approx(10.0)
    assert result.avg_middle_third_bins == pytest.approx(5.0)
    assert result.avg_upper_third_bins == pytest.approx(15.0)
    assert result.inter_tertile_ratio == pytest.approx(10.0 / 15.0)
    assert "Highly Progressive" in result.inter_tertile_desc


def test_gap_absolute_description_names_registered_unit():
    """gap_absolute_desc's "Spread of X units" previously always used the
    literal word "units", regardless of what unit the travel matrix was
    actually registered with -- meaningless when e.g. minutes or miles was
    the real unit. Falls back to the literal word "units" only when no
    unit was ever registered (see test_equity_band_gap_and_tertile_metrics,
    whose loaded_problem_with_equity fixture registers no unit)."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 5.0, 15.0],
            "Site_B": [20.0, 15.0, 25.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="minutes")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD decile",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )
    assert "Spread of 10.0 minutes" in result.gap_absolute_desc


def test_gap_relative_weighted_zero_baseline_guard():
    """When the best-performing equity band has a weighted average of
    exactly 0, gap_relative_weighted must not raise ZeroDivisionError -- it
    should fall back to NaN with a descriptive message."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [0.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group[1] == pytest.approx(0.0)
    assert pd.isna(result.gap_relative_weighted)
    assert result.gap_relative_desc == "N/A (Zero baseline cost)"


def test_zero_weight_band_falls_back_to_unweighted_mean():
    """
    NOTE: when every demand point in an equity band has weight 0 (here,
    demand=0), `weighted_by_equity_group` for that band silently falls back
    to the plain (unweighted) mean rather than raising a ZeroDivisionError
    (site_solutions.py:273-279). This test pins that current fallback
    behaviour -- worth flagging as a known weighting-scheme inconsistency
    (a band computed this way isn't really "weighted" like its neighbours),
    even though it's the sensible choice over crashing.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [0, 0, 200, 150],
        }
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 20.0, 30.0, 12.0],
            "Site_B": [25.0, 5.0, 15.0, 9.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [1, 1, 5, 10],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    # band 1 = LSOA_1 (min_cost 10.0) + LSOA_2 (min_cost 5.0), both demand=0
    assert result.weighted_by_equity_group[1] == pytest.approx((10.0 + 5.0) / 2)
    assert result.weighted_by_equity_group[1] == pytest.approx(
        result.unweighted_by_equity_group[1]
    )


def test_tertile_ordering_respects_disadvantaged_end_high(loaded_problem):
    """Same three-band setup as test_equity_band_gap_and_tertile_metrics
    (deciles 1, 5, 10), but with disadvantaged_end="high" instead of "low"
    -- i.e. band 10 is now the MOST disadvantaged and band 1 the LEAST.
    The tertile ordering must flip accordingly: avg_lower_third_bins (most
    disadvantaged) becomes band 10's value (15.0, not band 1's 10.0), and
    avg_upper_third_bins (least disadvantaged) becomes band 1's value
    (10.0). Before the disadvantaged_end fix, this ordering was hardcoded
    to "lower raw bin value = more disadvantaged" regardless of this
    parameter, which is wrong whenever disadvantaged_end="high"."""
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )
    loaded_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="high",
    )

    result = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group == {1: 10.0, 5: 5.0, 10: 15.0}
    assert result.avg_lower_third_bins == pytest.approx(15.0)
    assert result.avg_middle_third_bins == pytest.approx(5.0)
    assert result.avg_upper_third_bins == pytest.approx(10.0)
    assert result.inter_tertile_ratio == pytest.approx(15.0 / 10.0)
    assert "Highly Inequitable" in result.inter_tertile_desc


def test_tertile_split_uses_equal_sized_ends_not_array_split_remainder():
    """Regression test for the `_split_bins_into_tertiles()` fix: with 10
    IMD deciles (not divisible by 3), the old `np.array_split`-based split
    put the leftover band in the first chunk (1-4 / 5-7 / 8-10), comparing
    a 4-band "most deprived" group against a 3-band "least deprived" one.
    The fix keeps both ends at 3 bands each and puts the leftover decile
    (4) in the ignored middle instead (1-3 / 4-7 / 8-10).

    Decile 4's travel time (30.0) is deliberately far out of line with the
    rest, so which chunk it lands in isn't cosmetic: including it in the
    "most deprived" average (old behaviour) pushes inter_tertile_ratio from
    1.0 ("Balanced") to 1.5 ("Highly Inequitable") -- the fix changes not
    just a number but which `inter_tertile_desc` band the result falls
    into."""
    demand_df = pd.DataFrame(
        {
            "location_id": [f"LSOA_{i}" for i in range(1, 11)],
            "demand": [100] * 10,
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": [f"LSOA_{i}" for i in range(1, 11)],
            "Site_A": [10.0, 10.0, 10.0, 30.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": [f"LSOA_{i}" for i in range(1, 11)], "imd_decile": range(1, 11)}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD decile",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0]
    )

    assert result.avg_lower_third_bins == pytest.approx(10.0)
    assert result.avg_middle_third_bins == pytest.approx(7.5)
    assert result.avg_upper_third_bins == pytest.approx(10.0)

    # The invariant the discrepancy report caught: the ratio is exactly the
    # mean of the two chunks _split_bins_into_tertiles() actually returned,
    # not some other grouping.
    assert result.inter_tertile_ratio == pytest.approx(
        result.avg_lower_third_bins / result.avg_upper_third_bins
    )
    assert result.inter_tertile_ratio == pytest.approx(1.0)
    assert "Balanced" in result.inter_tertile_desc


def test_check_solution_equity_weighted_matches_weighted_by_equity_group():
    """check_solution_equity(weighted=True) must return the same figures as
    weighted_by_equity_group (and therefore inter_tertile_ratio) -- the
    whole point of adding weighted= was to remove the silent divergence
    between this method's default (plain per-region mean) and the
    demand-weighted metric plot_equity_tertiles() decomposes. Uses a band
    with two locations at very different demand so the weighted and
    unweighted means are provably different, not just non-identical by
    floating-point noise."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "demand": [990, 10, 100, 100],
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "Site_A": [0.0, 100.0, 20.0, 30.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "band": [1, 1, 2, 3],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="band",
        common_col="location_id",
        label="Band",
        disadvantaged_end="low",
    )
    solution = problem.solve(p=1, search_strategy="brute-force", show_progress=False)
    row = solution.solution_df.iloc[0]

    unweighted_summary = solution.check_solution_equity(weighted=False).set_index("band")[
        "min_cost"
    ]
    weighted_summary = solution.check_solution_equity(weighted=True).set_index("band")[
        "min_cost"
    ]

    assert unweighted_summary.loc[1] == pytest.approx(50.0)
    assert weighted_summary.loc[1] == pytest.approx(row["weighted_by_equity_group"][1])
    assert weighted_summary.loc[1] != pytest.approx(unweighted_summary.loc[1])
    assert weighted_summary.loc[1] == pytest.approx(0.0, abs=2.0)


def test_fewer_than_three_equity_bins_skips_tertile_calculation():
    """Tertile metrics require >= 3 distinct equity bands (site_solutions.py:325);
    with only 2, avg_*_third_bins and inter_tertile_ratio should stay unset."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 1, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.avg_lower_third_bins is None
    assert result.inter_tertile_ratio is None
    assert result.inter_tertile_desc == "N/A (No equity data)"
