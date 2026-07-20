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
