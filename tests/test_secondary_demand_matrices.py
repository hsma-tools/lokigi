"""Tests for registering additional (non-primary) demand scenarios via
`add_secondary_demand()`.

Fixtures live in conftest.py: `secondary_demand_df` ("future_demand") is
deliberately shaped so the primary demand's p=1 winner (Site_B) is not the
future scenario's winner (Site_A), and its id/demand column names
("region_id"/"future_demand") differ from the primary demand's
("location_id"/"demand") so an implementation that accidentally reuses the
primary demand's column names produces a KeyError rather than a
plausible-looking number.

Hand-computed primary weighted_average (demand-weighted, demand=100/200/150
for LSOA_1/2/3 -- see test_secondary_travel_matrices.py):
  Site_A: 9500 / 450 = 21.11111...
  Site_B: 5750 / 450 = 12.77777... (p=1 winner)
  Site_C: 6200 / 450 = 13.77777...

Hand-computed future_demand weighted_average (future_demand=300/50/50,
total=400):
  Site_A: (10*300 + 20*50 + 30*50) / 400 = 5500 / 400 = 13.75 (p=1 winner)
  Site_B: (25*300 + 5*50  + 15*50) / 400 = 8500 / 400 = 21.25
  Site_C: (30*300 + 10*50 + 8*50 ) / 400 = 9900 / 400 = 24.75

Hand-computed future_demand proportion_within_coverage_threshold (threshold
15, strict <), per site's own min_cost row [LSOA_1, LSOA_2, LSOA_3]:
  Site_A costs [10, 20, 30] -> within [T, F, F] -> covered = 300 / 400 = 0.75
  Site_B costs [25, 5, 15]  -> within [F, T, F] -> covered = 50  / 400 = 0.125
  Site_C costs [30, 10, 8]  -> within [F, T, T] -> covered = 100 / 400 = 0.25
"""

import pytest
import lokigi
import pandas as pd
import numpy as np

from lokigi.multiobjective import Metric
from lokigi.site_solutions import SolutionComparator

PRIMARY_WEIGHTED_AVERAGE = {
    "Site_A": 9500 / 450,
    "Site_B": 5750 / 450,
    "Site_C": 6200 / 450,
}
FUTURE_WEIGHTED_AVERAGE = {
    "Site_A": 5500 / 400,
    "Site_B": 8500 / 400,
    "Site_C": 9900 / 400,
}
FUTURE_COVERAGE_AT_15 = {
    "Site_A": 300 / 400,
    "Site_B": 50 / 400,
    "Site_C": 100 / 400,
}
SECONDARY_TRAVEL_CONSTANT = {"Site_A": 90.0, "Site_B": 50.0, "Site_C": 5.0}


def _row_for_site(solution_df, site_name):
    mask = solution_df["site_names"].apply(lambda names: list(names) == [site_name])
    matches = solution_df[mask]
    assert len(matches) == 1, f"Expected exactly one row for {site_name}"
    return matches.iloc[0]


# --- Registration errors -----------------------------------------------


def test_add_secondary_demand_rejects_empty_label(loaded_problem, secondary_demand_df):
    with pytest.raises(ValueError):
        loaded_problem.add_secondary_demand(
            secondary_demand_df,
            demand_col="future_demand",
            location_id_col="region_id",
            label="",
        )


def test_add_secondary_demand_rejects_dunder_in_label(
    loaded_problem, secondary_demand_df
):
    with pytest.raises(ValueError, match="__"):
        loaded_problem.add_secondary_demand(
            secondary_demand_df,
            demand_col="future_demand",
            location_id_col="region_id",
            label="future__demand",
        )


def test_add_secondary_demand_rejects_duplicate_label(
    loaded_problem_with_secondary_demand, secondary_demand_df
):
    with pytest.raises(ValueError, match="already been registered"):
        loaded_problem_with_secondary_demand.add_secondary_demand(
            secondary_demand_df,
            demand_col="future_demand",
            location_id_col="region_id",
            label="future_demand",
        )


def test_add_secondary_demand_rejects_label_already_used_by_secondary_travel(
    loaded_problem_with_secondary_matrix, secondary_demand_df
):
    with pytest.raises(ValueError, match="secondary travel matrix"):
        loaded_problem_with_secondary_matrix.add_secondary_demand(
            secondary_demand_df,
            demand_col="future_demand",
            location_id_col="region_id",
            label="public_transport",
        )


def test_add_secondary_travel_matrix_rejects_label_already_used_by_secondary_demand(
    loaded_problem_with_secondary_demand, secondary_travel_df
):
    with pytest.raises(ValueError, match="secondary demand scenario"):
        loaded_problem_with_secondary_demand.add_secondary_travel_matrix(
            secondary_travel_df, source_col="source_id", label="future_demand"
        )


def test_show_secondary_demand_unknown_label_raises(loaded_problem):
    with pytest.raises(KeyError):
        loaded_problem.show_secondary_demand("nonexistent")


def test_show_secondary_demand_returns_registered_data(
    loaded_problem_with_secondary_demand,
):
    df = loaded_problem_with_secondary_demand.show_secondary_demand("future_demand")
    assert df["future_demand"].tolist() == [300, 50, 50]


# --- Completeness validation (raised at solve()) ------------------------


def test_secondary_demand_missing_row_raises(loaded_problem, secondary_demand_df):
    incomplete = secondary_demand_df.iloc[:2]  # drops LSOA_3
    loaded_problem.add_secondary_demand(
        incomplete,
        demand_col="future_demand",
        location_id_col="region_id",
        label="future_demand",
    )
    with pytest.raises(KeyError, match="future_demand"):
        loaded_problem.solve(p=1)


def test_secondary_demand_nan_value_raises(loaded_problem, secondary_demand_df):
    with_nan = secondary_demand_df.copy()
    with_nan.loc[0, "future_demand"] = np.nan
    loaded_problem.add_secondary_demand(
        with_nan,
        demand_col="future_demand",
        location_id_col="region_id",
        label="future_demand",
    )
    with pytest.raises(KeyError, match="future_demand"):
        loaded_problem.solve(p=1)


def test_also_weight_matrices_unregistered_travel_label_raises(
    loaded_problem, secondary_demand_df
):
    loaded_problem.add_secondary_demand(
        secondary_demand_df,
        demand_col="future_demand",
        location_id_col="region_id",
        label="future_demand",
        also_weight_matrices=["nonexistent_travel_matrix"],
    )
    with pytest.raises(KeyError, match="nonexistent_travel_matrix"):
        loaded_problem.solve(p=1)


# --- Suffixed columns in solution_df -------------------------------------


def test_primary_baseline_site_b_wins_p1(loaded_problem_with_secondary_demand):
    """Sanity-check the un-confounded baseline before relying on the trap:
    Site_B really is the primary p=1 winner."""
    result = loaded_problem_with_secondary_demand.solve(p=1)
    assert result.solution_df.iloc[0]["site_names"] == ["Site_B"]
    assert result.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        PRIMARY_WEIGHTED_AVERAGE["Site_B"]
    )


def test_secondary_demand_columns_match_hand_computed_values(
    loaded_problem_with_secondary_demand,
):
    result = loaded_problem_with_secondary_demand.solve(p=1, threshold_for_coverage=15)
    solution_df = result.solution_df

    for site, expected_future in FUTURE_WEIGHTED_AVERAGE.items():
        row = _row_for_site(solution_df, site)
        assert row["weighted_average"] == pytest.approx(PRIMARY_WEIGHTED_AVERAGE[site])
        assert row["weighted_average__future_demand"] == pytest.approx(expected_future)
        assert row[
            "proportion_within_coverage_threshold__future_demand"
        ] == pytest.approx(FUTURE_COVERAGE_AT_15[site])

    # The future scenario's own winner (Site_A) differs from the primary
    # winner (Site_B) -- proof the trap fixture actually discriminates.
    best_future = solution_df.sort_values("weighted_average__future_demand").iloc[0]
    assert best_future["site_names"] == ["Site_A"]
    assert best_future["site_names"] != solution_df.iloc[0]["site_names"]


def test_only_demand_varying_metrics_are_suffixed(loaded_problem_with_secondary_demand):
    """unweighted_average/90th_percentile/max/proportion_regions_within_
    coverage_threshold don't vary with demand, so a secondary demand
    scenario must never suffix them -- keeping the default additive cost
    to 2 columns per scenario, not a full metric block."""
    result = loaded_problem_with_secondary_demand.solve(p=1, threshold_for_coverage=15)
    columns = result.solution_df.columns

    assert "weighted_average__future_demand" in columns
    assert "proportion_within_coverage_threshold__future_demand" in columns

    assert "unweighted_average__future_demand" not in columns
    assert "90th_percentile__future_demand" not in columns
    assert "max__future_demand" not in columns
    assert "proportion_regions_within_coverage_threshold__future_demand" not in columns
    assert "gap_absolute_weighted__future_demand" not in columns
    assert "weighted_by_equity_group__future_demand" not in columns


def test_problem_df_does_not_gain_a_visible_demand_column_name_clash(
    loaded_problem_with_secondary_demand,
):
    """The internal `demand__<label>` column lives on problem_df (needed to
    align weights correctly), but must not collide with or shadow the
    primary `demand` column."""
    result = loaded_problem_with_secondary_demand.solve(p=1)
    problem_df = result.solution_df.iloc[0]["problem_df"]
    assert "demand__future_demand" in problem_df.columns
    assert "demand" in problem_df.columns
    assert problem_df["demand__future_demand"].tolist() == [300.0, 50.0, 50.0]


# --- Opt-in cross-product with secondary travel matrices ------------------


def test_also_weight_matrices_produces_cross_columns(
    loaded_problem_with_secondary_demand_and_travel,
):
    result = loaded_problem_with_secondary_demand_and_travel.solve(
        p=1, threshold_for_coverage=60
    )
    solution_df = result.solution_df

    for site, expected_constant in SECONDARY_TRAVEL_CONSTANT.items():
        row = _row_for_site(solution_df, site)
        # public_transport cost is uniform per site, so its weighted
        # average equals that constant regardless of which demand vector
        # weights it -- true for both the primary-demand and future-demand
        # weighted columns.
        assert row["weighted_average__public_transport"] == pytest.approx(
            expected_constant
        )
        assert row[
            "weighted_average__public_transport__future_demand"
        ] == pytest.approx(expected_constant)

    # Also still get the (default) primary-travel cross.
    assert "weighted_average__future_demand" in solution_df.columns


def test_default_no_cross_product_without_also_weight_matrices(
    loaded_problem_with_secondary_matrix, secondary_demand_df
):
    loaded_problem_with_secondary_matrix.add_secondary_demand(
        secondary_demand_df,
        demand_col="future_demand",
        location_id_col="region_id",
        label="future_demand",
    )
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    columns = result.solution_df.columns

    assert "weighted_average__future_demand" in columns
    assert "weighted_average__public_transport" in columns
    assert "weighted_average__public_transport__future_demand" not in columns


# --- Weights blend (compound_weights) -------------------------------------


def test_weights_blend_matches_hand_computed_compound_average(
    loaded_problem_with_secondary_demand,
):
    """weights={"demand": 0.6, "future_demand": 0.4}: each demand vector is
    min-max normalised (constant_fill=1.0, though neither is constant
    here), then blended by weight.

    demand=[100,200,150] -> norm=[0.0, 1.0, 0.5]
    future_demand=[300,50,50] -> norm=[1.0, 0.0, 0.0]
    compound = 0.6*norm_demand + 0.4*norm_future
             = [0.4, 0.6, 0.3]  (LSOA_1, LSOA_2, LSOA_3)

    weighted_average(site) = sum(cost * compound) / sum(compound), where
    sum(compound) = 1.3:
      Site_A: (10*0.4 + 20*0.6 + 30*0.3) / 1.3 = 25.0   / 1.3 = 19.230769...
      Site_B: (25*0.4 + 5*0.6  + 15*0.3) / 1.3 = 17.5   / 1.3 = 13.461538...
      Site_C: (30*0.4 + 10*0.6 + 8*0.3 ) / 1.3 = 20.4   / 1.3 = 15.692308...
    """
    result = loaded_problem_with_secondary_demand.solve(
        p=1, weights={"demand": 0.6, "future_demand": 0.4}
    )
    solution_df = result.solution_df

    expected = {
        "Site_A": 25.0 / 1.3,
        "Site_B": 17.5 / 1.3,
        "Site_C": 20.4 / 1.3,
    }
    for site, expected_value in expected.items():
        row = _row_for_site(solution_df, site)
        assert row["weighted_average"] == pytest.approx(expected_value)

    # Distinct from the pure-primary-demand weighted average -- proof the
    # blend actually took effect rather than silently falling back.
    site_a_row = _row_for_site(solution_df, "Site_A")
    assert site_a_row["weighted_average"] != pytest.approx(
        PRIMARY_WEIGHTED_AVERAGE["Site_A"]
    )


def test_weights_with_unregistered_secondary_demand_label_raises(loaded_problem):
    with pytest.raises(KeyError):
        loaded_problem.solve(p=1, weights={"demand": 0.5, "future_demand": 0.5})


# --- Pareto -----------------------------------------------------------


def test_compute_pareto_front_on_secondary_demand_column(
    loaded_problem_with_secondary_demand,
):
    result = loaded_problem_with_secondary_demand.solve(p=1)
    result.compute_pareto_front(
        metrics=[
            Metric(column="weighted_average", direction="lower_better"),
            Metric(
                column="weighted_average__future_demand", direction="lower_better"
            ),
        ]
    )
    pareto_by_site = {
        row["site_names"][0]: row["is_pareto_optimal"]
        for _, row in result.solution_df.iterrows()
    }
    # Site_B (best primary) and Site_A (best future) trade off against each
    # other -- neither dominates the other.
    assert pareto_by_site["Site_A"] is True
    assert pareto_by_site["Site_B"] is True
    # Site_C is dominated by Site_B on both axes (worse primary AND worse
    # future weighted_average).
    assert pareto_by_site["Site_C"] is False


# --- 2SFCA `demand=` kwarg -------------------------------------------------


def test_2sfca_demand_kwarg_matches_hand_computed_values(
    sfca_problem_with_secondary_demand,
):
    region_frame, site_frame = sfca_problem_with_secondary_demand.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        demand="future_demand",
        return_site_ratios=True,
    )

    r1 = 10 / 100
    r2 = 5 / 150
    assert site_frame.loc["Site_1", "catchment_demand"] == 100
    assert site_frame.loc["Site_2", "catchment_demand"] == 150
    assert site_frame.loc["Site_1", "ratio"] == pytest.approx(r1)
    assert site_frame.loc["Site_2", "ratio"] == pytest.approx(r2)

    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(r1 + r2)
    assert region_frame.loc["LSOA_2", "accessibility"] == pytest.approx(r1)
    assert region_frame.loc["LSOA_3", "accessibility"] == pytest.approx(r2)
    assert region_frame.loc["LSOA_1", "demand"] == 50
    assert region_frame.loc["LSOA_3", "demand"] == 100


def test_2sfca_demand_kwarg_differs_from_primary(sfca_problem_with_secondary_demand):
    primary = sfca_problem_with_secondary_demand.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    future = sfca_problem_with_secondary_demand.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        demand="future_demand",
    )
    assert not primary["accessibility"].equals(future["accessibility"])


def test_2sfca_unknown_demand_label_raises(sfca_problem_with_secondary_demand):
    with pytest.raises(ValueError, match="Unknown secondary demand scenario"):
        sfca_problem_with_secondary_demand.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2"],
            demand="nonexistent",
        )


def test_solution_set_2sfca_demand_kwarg_passthrough(sfca_problem_with_secondary_demand):
    result = sfca_problem_with_secondary_demand.solve(
        p=2, search_strategy="brute-force", show_progress=False
    )
    region_frame = result.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        demand="future_demand",
    )
    assert region_frame.loc["LSOA_1", "demand"] == 50


# --- site_allocation_summary `demand=` kwarg -------------------------------


def test_site_allocation_summary_demand_kwarg_matches_hand_computed_values(
    loaded_problem_with_secondary_demand,
):
    result = loaded_problem_with_secondary_demand.solve(p=1)  # Site_B wins primary

    default_summary = result.site_allocation_summary(by="demand")
    future_summary = result.site_allocation_summary(by="demand", demand="future_demand")

    assert default_summary.loc["Site_B", "allocated_demand"] == 450
    assert default_summary.loc["Site_B", "average_travel_cost"] == pytest.approx(
        PRIMARY_WEIGHTED_AVERAGE["Site_B"]
    )

    assert future_summary.loc["Site_B", "allocated_demand"] == 400
    assert future_summary.loc["Site_B", "average_travel_cost"] == pytest.approx(
        FUTURE_WEIGHTED_AVERAGE["Site_B"]
    )


def test_site_allocation_summary_unknown_demand_label_raises(
    loaded_problem_with_secondary_demand,
):
    result = loaded_problem_with_secondary_demand.solve(p=1)
    with pytest.raises(ValueError, match="Unknown secondary demand scenario"):
        result.site_allocation_summary(by="demand", demand="nonexistent")


def test_compare_site_allocation_demand_kwarg_passthrough(
    loaded_problem_with_secondary_demand,
):
    result_p1 = loaded_problem_with_secondary_demand.solve(p=1)
    result_p2 = loaded_problem_with_secondary_demand.solve(
        p=2, search_strategy="brute-force"
    )

    comparator = SolutionComparator(result_p1, result_p2, labels=("1-site", "2-site"))
    comparison = comparator.compare_site_allocation(
        by="demand", demand="future_demand"
    )
    assert comparison is not None
    assert len(comparison) > 0


# --- Equity + secondary demand (no crash) ----------------------------------


def test_equity_and_secondary_demand_solve_does_not_crash(
    loaded_problem_with_equity_and_secondary_demand,
):
    """_compute_travel_metrics still computes the by-equity-group breakdown
    internally for a secondary demand scenario (even though it is never
    emitted), so the scenario's weight series must share evaluated_
    combination_df's index or the equity groupby's `.loc[group.index]`
    raises. This is a regression test for that alignment."""
    result = loaded_problem_with_equity_and_secondary_demand.solve(p=1)
    assert "weighted_average__future_demand" in result.solution_df.columns
    assert "weighted_by_equity_group__future_demand" not in result.solution_df.columns


# --- Plotting ---------------------------------------------------------


def test_plot_region_geometry_layer_demand_kwarg_runs(
    loaded_problem_with_secondary_demand,
):
    import geopandas
    from shapely.geometry import Point

    regions = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "geometry": [
                Point(0, 0).buffer(0.4, cap_style=3),
                Point(1, 0).buffer(0.4, cap_style=3),
                Point(0, 1).buffer(0.4, cap_style=3),
            ],
        },
        crs="EPSG:27700",
    )
    loaded_problem_with_secondary_demand.add_region_geometry_layer(
        regions, common_col="location_id"
    )
    ax = loaded_problem_with_secondary_demand.plot_region_geometry_layer(
        plot_demand=True, demand="future_demand", add_basemap=False
    )
    assert ax is not None


def test_plot_region_geometry_layer_demand_requires_plot_demand_true(
    loaded_problem_with_secondary_demand,
):
    import geopandas
    from shapely.geometry import Point

    regions = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "geometry": [
                Point(0, 0).buffer(0.4, cap_style=3),
                Point(1, 0).buffer(0.4, cap_style=3),
                Point(0, 1).buffer(0.4, cap_style=3),
            ],
        },
        crs="EPSG:27700",
    )
    loaded_problem_with_secondary_demand.add_region_geometry_layer(
        regions, common_col="location_id"
    )
    with pytest.raises(ValueError, match="plot_demand=True"):
        loaded_problem_with_secondary_demand.plot_region_geometry_layer(
            plot_demand=False, demand="future_demand"
        )


def test_plot_region_geometry_layer_unknown_demand_label_raises(
    loaded_problem_with_secondary_demand,
):
    import geopandas
    from shapely.geometry import Point

    regions = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "geometry": [
                Point(0, 0).buffer(0.4, cap_style=3),
                Point(1, 0).buffer(0.4, cap_style=3),
                Point(0, 1).buffer(0.4, cap_style=3),
            ],
        },
        crs="EPSG:27700",
    )
    loaded_problem_with_secondary_demand.add_region_geometry_layer(
        regions, common_col="location_id"
    )
    with pytest.raises(ValueError, match="Unknown secondary demand scenario"):
        loaded_problem_with_secondary_demand.plot_region_geometry_layer(
            plot_demand=True, demand="nonexistent", add_basemap=False
        )


def test_plot_site_allocation_summary_demand_kwarg_runs(
    loaded_problem_with_secondary_demand,
):
    result = loaded_problem_with_secondary_demand.solve(p=1)
    fig = result.plot_site_allocation_summary(demand="future_demand", interactive=False)
    assert fig is not None


# --- Per-equity-band breakdowns per demand scenario ------------------------
#
# `loaded_problem_with_equity`'s three LSOAs sit one-per-equity-band, so
# every band's weighted mean collapses to that single region's cost and is
# identical under any demand scenario -- fine for the no-crash regression
# above, useless for checking the scenario's weights actually reach the
# band aggregation. These tests use a four-region problem where band 1
# holds two regions with a different demand split in each scenario, so a
# wrong (or primary-demand-leaking) weight series produces a visibly
# different number rather than the right one by construction.
#
# Hand-computed, min_cost = [10, 30, 20, 40] for LSOA_1..4 (Site_X always
# wins -- Site_Y is 100 everywhere), bands 1/1/5/10:
#
#   primary demand 100/100/100/100:
#     band 1 = (10*100 + 30*100) / 200  = 20.0
#     band 5 = 20.0, band 10 = 40.0
#     tertiles of [1, 5, 10] -> lower [1], upper [10]
#     inter_tertile_ratio    = 20 / 40 = 0.5
#     gap_absolute_weighted  = 40 - 20 = 20.0
#
#   future_demand 900/300/100/100:
#     band 1 = (10*900 + 30*300) / 1200 = 15.0
#     band 5 = 20.0, band 10 = 40.0
#     inter_tertile_ratio    = 15 / 40 = 0.375
#     gap_absolute_weighted  = 40 - 15 = 25.0


@pytest.fixture
def banded_problem_with_secondary_demand():
    problem = lokigi.site.SiteProblem(debug_mode=False)
    ids = ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"]
    problem.add_demand(
        pd.DataFrame({"location_id": ids, "demand": [100, 100, 100, 100]}),
        demand_col="demand",
        location_id_col="location_id",
    )
    problem.add_sites(
        pd.DataFrame(
            {
                "site_id": ["Site_X", "Site_Y"],
                "lat": [51.5, 51.6],
                "long": [-0.1, -0.2],
            }
        ),
        candidate_id_col="site_id",
    )
    problem.add_travel_matrix(
        pd.DataFrame(
            {
                "source_id": ids,
                "Site_X": [10.0, 30.0, 20.0, 40.0],
                "Site_Y": [100.0, 100.0, 100.0, 100.0],
            }
        ),
        source_col="source_id",
    )
    problem.add_equity_data(
        pd.DataFrame({"location_id": ids, "imd_decile": [1, 1, 5, 10]}),
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    problem.add_secondary_demand(
        pd.DataFrame({"region_id": ids, "future_demand": [900, 300, 100, 100]}),
        demand_col="future_demand",
        location_id_col="region_id",
        label="future_demand",
    )
    return problem


def test_float_equity_aggregations_are_suffixed_per_demand_scenario(
    banded_problem_with_secondary_demand,
):
    """The float-valued equity aggregations vary with demand, so each
    scenario gets its own -- and the scenario's value must be computed from
    the scenario's weights, not the primary demand's."""
    row = banded_problem_with_secondary_demand.solve(p=1).solution_df.iloc[0]

    assert row["inter_tertile_ratio"] == pytest.approx(0.5)
    assert row["inter_tertile_ratio__future_demand"] == pytest.approx(0.375)

    assert row["gap_absolute_weighted"] == pytest.approx(20.0)
    assert row["gap_absolute_weighted__future_demand"] == pytest.approx(25.0)

    assert row["gap_relative_weighted"] == pytest.approx(40.0 / 20.0)
    assert row["gap_relative_weighted__future_demand"] == pytest.approx(40.0 / 15.0)

    assert row["avg_lower_third_bins__future_demand"] == pytest.approx(15.0)
    assert row["avg_middle_third_bins__future_demand"] == pytest.approx(20.0)
    assert row["avg_upper_third_bins__future_demand"] == pytest.approx(40.0)


def test_dict_equity_breakdowns_need_full_secondary_metrics(
    banded_problem_with_secondary_demand,
):
    """Dict-valued per-band breakdowns are one column per scenario but one
    dict entry per band, so they sit behind full_secondary_metrics -- the
    same gate the secondary travel matrix block uses."""
    default_cols = banded_problem_with_secondary_demand.solve(p=1).solution_df.columns
    assert "weighted_by_equity_group__future_demand" not in default_cols
    assert "coverage_by_equity_group__future_demand" not in default_cols
    assert "inter_tertile_description__future_demand" not in default_cols
    # ...while the float aggregation from the same scenario is present.
    assert "inter_tertile_ratio__future_demand" in default_cols

    row = (
        banded_problem_with_secondary_demand.solve(
            p=1, threshold_for_coverage=25, full_secondary_metrics=True
        )
        .solution_df.iloc[0]
    )

    assert row["weighted_by_equity_group"] == {1: 20.0, 5: 20.0, 10: 40.0}
    assert row["weighted_by_equity_group__future_demand"] == {
        1: 15.0,
        5: 20.0,
        10: 40.0,
    }

    # within_threshold at 25 -> [T, F, T, F] for costs [10, 30, 20, 40].
    # Band 1 holds LSOA_1 (covered) and LSOA_2 (not), so its covered share
    # is the covered region's demand share within the band: 100/200 = 0.5
    # primary, 900/1200 = 0.75 under the scenario.
    assert row["coverage_by_equity_group"] == {1: 0.5, 5: 1.0, 10: 0.0}
    assert row["coverage_by_equity_group__future_demand"] == {
        1: 0.75,
        5: 1.0,
        10: 0.0,
    }

    assert isinstance(row["inter_tertile_description__future_demand"], str)


def test_unweighted_equity_breakdowns_never_suffixed_per_demand_scenario(
    banded_problem_with_secondary_demand,
):
    """The per-band breakdowns computed from travel cost alone carry no
    demand term, so a per-scenario copy would just repeat the primary
    matrix's value under a new name. Excluded even under
    full_secondary_metrics, for the same reason regions_unreachable is."""
    columns = banded_problem_with_secondary_demand.solve(
        p=1, threshold_for_coverage=25, full_secondary_metrics=True
    ).solution_df.columns

    for base in (
        "unweighted_by_equity_group",
        "coverage_regions_by_equity_group",
        "max_cost_by_equity_group",
        "regions_unreachable_by_equity_group",
    ):
        assert base in columns, f"{base} should still exist for the primary matrix"
        assert f"{base}__future_demand" not in columns


def test_beyond_threshold_equity_breakdown_per_demand_scenario(
    banded_problem_with_secondary_demand,
):
    """beyond_thresholds' per-band headcount is demand-weighted, so it gets
    a per-scenario variant under full_secondary_metrics. At 15, beyond =
    [F, T, T, T]; band 1's only beyond-region is LSOA_2, whose demand is
    100 primary but 300 under the scenario."""
    row = (
        banded_problem_with_secondary_demand.solve(
            p=1, beyond_thresholds=[15], full_secondary_metrics=True
        )
        .solution_df.iloc[0]
    )

    assert row["demand_beyond_threshold_15_by_equity_group"] == {
        1: 100.0,
        5: 100.0,
        10: 100.0,
    }
    assert row["demand_beyond_threshold_15_by_equity_group__future_demand"] == {
        1: 300.0,
        5: 100.0,
        10: 100.0,
    }

    # Region counts don't vary with demand -- primary only.
    assert "regions_beyond_threshold_15_by_equity_group" in row.index
    assert "regions_beyond_threshold_15_by_equity_group__future_demand" not in row.index


def test_no_equity_columns_for_scenario_without_equity_data(
    loaded_problem_with_secondary_demand,
):
    """A problem with no add_equity_data() call has no equity answer to
    give, so a registered scenario must not add six None columns. Unlike
    the secondary travel matrix block, which emits them unconditionally."""
    columns = loaded_problem_with_secondary_demand.solve(
        p=1, full_secondary_metrics=True
    ).solution_df.columns

    for base in (
        "gap_absolute_weighted",
        "gap_relative_weighted",
        "avg_lower_third_bins",
        "avg_middle_third_bins",
        "avg_upper_third_bins",
        "inter_tertile_ratio",
    ):
        assert f"{base}__future_demand" not in columns


def test_scenario_equity_columns_classify_as_equity(
    banded_problem_with_secondary_demand,
):
    """describe_solution_columns() groups by the base name before the first
    `__`, so the new suffixed columns must land in "Equity" rather than
    "Other" without _SOLUTION_COLUMN_GROUPS needing per-scenario entries."""
    from lokigi.site_solutions import _classify_solution_column

    assert _classify_solution_column("inter_tertile_ratio__future_demand") == "Equity"
    assert (
        _classify_solution_column("weighted_by_equity_group__pt__future_demand")
        == "Equity"
    )

    result = banded_problem_with_secondary_demand.solve(
        p=1, threshold_for_coverage=25, full_secondary_metrics=True
    )
    grouped = result.describe_solution_columns(return_dict=True)

    # Asserted positively rather than as "nothing landed in Other": there
    # is no "Other" group at all here, so a `not in Other` check would pass
    # even if the columns had gone missing entirely.
    equity_cols = set(grouped["Equity"])
    for base in (
        "inter_tertile_ratio",
        "gap_absolute_weighted",
        "weighted_by_equity_group",
        "coverage_by_equity_group",
        "inter_tertile_description",
    ):
        assert f"{base}__future_demand" in equity_cols

    scenario_cols = {
        col
        for cols in grouped.values()
        for col in cols
        if str(col).endswith("__future_demand")
    }
    assert scenario_cols, "no scenario columns were classified at all"
    assert "Other" not in grouped
