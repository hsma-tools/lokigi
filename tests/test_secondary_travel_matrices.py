"""Tests for registering additional (non-primary) travel matrices via
`add_secondary_travel_matrix()`.

Fixtures live in conftest.py: `secondary_travel_df` is deliberately shaped
so the primary matrix's p=1 winner (Site_B) is not the secondary matrix's
winner (Site_C), and its columns are ordered the reverse of `travel_df`
(Site_C, Site_B, Site_A instead of Site_A, Site_B, Site_C) -- so an
implementation that accidentally reuses the primary matrix's positional
column indices produces visibly wrong numbers rather than plausible ones.

Hand-computed primary weighted_average (demand-weighted, demand=100/200/150
for LSOA_1/2/3):
  Site_A: (100*10 + 200*20 + 150*30) / 450 = 9500 / 450 = 21.11111...
  Site_B: (100*25 + 200*5  + 150*15) / 450 = 5750 / 450 = 12.77777...
  Site_C: (100*30 + 200*10 + 150*8 ) / 450 = 6200 / 450 = 13.77777...

Secondary ("public_transport") costs are uniform per site across all demand
locations, so every travel statistic (weighted_average, unweighted_average,
max, 90th_percentile) for a p=1 solution equals that site's constant value:
  Site_A = 90.0, Site_B = 50.0, Site_C = 5.0
"""

import pytest
import pandas as pd
import numpy as np

from lokigi.multiobjective import Metric

PRIMARY_WEIGHTED_AVERAGE = {
    "Site_A": 9500 / 450,
    "Site_B": 5750 / 450,
    "Site_C": 6200 / 450,
}
SECONDARY_CONSTANT = {"Site_A": 90.0, "Site_B": 50.0, "Site_C": 5.0}


def _row_for_site(solution_df, site_name):
    mask = solution_df["site_names"].apply(lambda names: list(names) == [site_name])
    matches = solution_df[mask]
    assert len(matches) == 1, f"Expected exactly one row for {site_name}"
    return matches.iloc[0]


# --- Registration errors -----------------------------------------------


def test_add_secondary_travel_matrix_rejects_empty_label(
    loaded_problem, secondary_travel_df
):
    with pytest.raises(ValueError):
        loaded_problem.add_secondary_travel_matrix(
            secondary_travel_df, source_col="source_id", label=""
        )


def test_add_secondary_travel_matrix_rejects_dunder_in_label(
    loaded_problem, secondary_travel_df
):
    with pytest.raises(ValueError, match="__"):
        loaded_problem.add_secondary_travel_matrix(
            secondary_travel_df, source_col="source_id", label="public__transport"
        )


def test_add_secondary_travel_matrix_rejects_duplicate_label(
    loaded_problem_with_secondary_matrix, secondary_travel_df
):
    with pytest.raises(ValueError, match="already been registered"):
        loaded_problem_with_secondary_matrix.add_secondary_travel_matrix(
            secondary_travel_df, source_col="source_id", label="public_transport"
        )


def test_show_secondary_travel_matrix_unknown_label_raises(loaded_problem):
    with pytest.raises(KeyError):
        loaded_problem.show_secondary_travel_matrix("nonexistent")


def test_show_secondary_travel_matrix_returns_registered_data(
    loaded_problem_with_secondary_matrix,
):
    df = loaded_problem_with_secondary_matrix.show_secondary_travel_matrix(
        "public_transport"
    )
    assert list(df["Site_C"]) == [5.0, 5.0, 5.0]


# --- Completeness validation (raised at solve()) ------------------------


def test_secondary_matrix_missing_demand_row_raises(loaded_problem, secondary_travel_df):
    incomplete = secondary_travel_df.iloc[:2]  # drops LSOA_3
    loaded_problem.add_secondary_travel_matrix(
        incomplete, source_col="source_id", label="public_transport"
    )
    with pytest.raises(KeyError, match="public_transport"):
        loaded_problem.solve(p=1)


def test_secondary_matrix_missing_site_column_raises(
    loaded_problem, secondary_travel_df
):
    incomplete = secondary_travel_df.drop(columns=["Site_A"])
    loaded_problem.add_secondary_travel_matrix(
        incomplete, source_col="source_id", label="public_transport"
    )
    with pytest.raises(KeyError, match="public_transport"):
        loaded_problem.solve(p=1)


def test_secondary_matrix_nan_cell_raises(loaded_problem, secondary_travel_df):
    with_nan = secondary_travel_df.copy()
    with_nan.loc[0, "Site_B"] = np.nan
    loaded_problem.add_secondary_travel_matrix(
        with_nan, source_col="source_id", label="public_transport"
    )
    with pytest.raises(KeyError, match="public_transport"):
        loaded_problem.solve(p=1)


# --- Suffixed columns in solution_df and problem_df ---------------------


def test_primary_baseline_site_b_wins_p1(loaded_problem_with_secondary_matrix):
    """Sanity-check the un-confounded baseline before relying on the trap:
    Site_B really is the primary p=1 winner."""
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    assert result.solution_df.iloc[0]["site_names"] == ["Site_B"]
    assert result.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        PRIMARY_WEIGHTED_AVERAGE["Site_B"]
    )


def test_secondary_columns_match_hand_computed_values(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    solution_df = result.solution_df

    for site, expected_primary in PRIMARY_WEIGHTED_AVERAGE.items():
        row = _row_for_site(solution_df, site)
        assert row["weighted_average"] == pytest.approx(expected_primary)
        assert row["weighted_average__public_transport"] == pytest.approx(
            SECONDARY_CONSTANT[site]
        )
        assert row["unweighted_average__public_transport"] == pytest.approx(
            SECONDARY_CONSTANT[site]
        )
        assert row["max__public_transport"] == pytest.approx(SECONDARY_CONSTANT[site])
        assert row["90th_percentile__public_transport"] == pytest.approx(
            SECONDARY_CONSTANT[site]
        )

    # The secondary matrix's own winner (Site_C) differs from the primary
    # winner (Site_B) -- proof the trap fixture actually discriminates.
    best_secondary = solution_df.sort_values("weighted_average__public_transport").iloc[
        0
    ]
    assert best_secondary["site_names"] == ["Site_C"]
    assert best_secondary["site_names"] != solution_df.iloc[0]["site_names"]


def test_problem_df_carries_secondary_per_region_columns(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    site_b_row = _row_for_site(result.solution_df, "Site_B")
    problem_df = site_b_row["problem_df"]

    assert "min_cost__public_transport" in problem_df.columns
    assert "selected_site__public_transport" in problem_df.columns
    assert "within_threshold__public_transport" in problem_df.columns
    assert (problem_df["min_cost__public_transport"] == 50.0).all()
    assert (problem_df["selected_site__public_transport"] == "Site_B").all()


# --- Per-matrix coverage threshold --------------------------------------


def test_secondary_matrix_falls_back_to_solve_threshold(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1, threshold_for_coverage=15)
    site_b_row = _row_for_site(result.solution_df, "Site_B")
    # min_cost__public_transport == 50 for Site_B, primary threshold is 15
    assert site_b_row["proportion_within_coverage_threshold__public_transport"] == 0.0

    site_c_row = _row_for_site(result.solution_df, "Site_C")
    # min_cost__public_transport == 5 for Site_C, below the threshold of 15
    assert site_c_row["proportion_within_coverage_threshold__public_transport"] == 1.0


def test_secondary_matrix_own_threshold_overrides_solve_threshold(
    loaded_problem, secondary_travel_df
):
    loaded_problem.add_secondary_travel_matrix(
        secondary_travel_df,
        source_col="source_id",
        label="public_transport",
        threshold_for_coverage=60,
    )
    result = loaded_problem.solve(p=1, threshold_for_coverage=15)
    site_b_row = _row_for_site(result.solution_df, "Site_B")
    # min_cost__public_transport == 50, below the matrix's own threshold of
    # 60 even though it's above the primary solve() threshold of 15.
    assert site_b_row["proportion_within_coverage_threshold__public_transport"] == 1.0
    # The regions variant honours the per-matrix threshold identically.
    assert (
        site_b_row["proportion_regions_within_coverage_threshold__public_transport"]
        == 1.0
    )


# --- Pareto and ranking ---------------------------------------------------


def test_compute_pareto_front_on_secondary_column(loaded_problem_with_secondary_matrix):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    result.compute_pareto_front(
        metrics=[
            Metric(column="weighted_average", direction="lower_better"),
            Metric(
                column="weighted_average__public_transport", direction="lower_better"
            ),
        ]
    )
    pareto_by_site = {
        row["site_names"][0]: row["is_pareto_optimal"]
        for _, row in result.solution_df.iterrows()
    }
    # Site_A is worse than both Site_B and Site_C on both axes -> dominated.
    assert pareto_by_site["Site_A"] is False
    # Site_B (best primary) and Site_C (best secondary) trade off against
    # each other -- neither dominates the other.
    assert pareto_by_site["Site_B"] is True
    assert pareto_by_site["Site_C"] is True


def test_rank_on_secondary_column_reorders_and_changes_winner(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1)

    primary_best = result.return_best_combination_site_names()
    secondary_best = result.return_best_combination_site_names(
        rank_on="max__public_transport"
    )

    assert primary_best == ["Site_B"]
    assert secondary_best == ["Site_C"]
    assert primary_best != secondary_best


# --- Unit conversion is independent per matrix ---------------------------


def test_secondary_matrix_unit_conversion_independent_of_primary(
    loaded_problem, secondary_travel_df
):
    loaded_problem.add_secondary_travel_matrix(
        secondary_travel_df,
        source_col="source_id",
        label="public_transport",
        from_unit="minutes",
        to_unit="hours",
    )
    # Primary matrix (travel_df, added by loaded_problem) has no unit
    # conversion applied -- unaffected by the secondary matrix's conversion.
    assert loaded_problem._travel_matrix_unit is None
    assert loaded_problem.travel_matrix["Site_A"].tolist() == [10.0, 20.0, 30.0]

    # Secondary matrix converted minutes -> hours (factor 1/60), and its
    # unit label is stored independently of the primary matrix's.
    converted = loaded_problem.secondary_travel_matrices["public_transport"]["data"]
    assert converted["Site_C"].tolist() == pytest.approx([5.0 / 60, 5.0 / 60, 5.0 / 60])
    assert (
        loaded_problem.secondary_travel_matrices["public_transport"]["unit"] == "hours"
    )


# --- full_secondary_metrics opt-in ---------------------------------------


def test_full_secondary_metrics_default_omits_dict_valued_columns(
    loaded_problem_with_equity_and_secondary_matrix,
):
    result = loaded_problem_with_equity_and_secondary_matrix.solve(p=1)
    columns = result.solution_df.columns

    # Core scalar metrics + float equity aggregations are always present.
    assert "weighted_average__public_transport" in columns
    assert "gap_absolute_weighted__public_transport" in columns
    # Both coverage proportions, so a secondary matrix reports the same pair
    # as the primary one.
    assert "proportion_within_coverage_threshold__public_transport" in columns
    assert "proportion_regions_within_coverage_threshold__public_transport" in columns

    # Dict-valued breakdowns and description strings are not, by default.
    assert "weighted_by_equity_group__public_transport" not in columns
    assert "unweighted_by_equity_group__public_transport" not in columns
    assert "coverage_by_equity_group__public_transport" not in columns
    assert "coverage_regions_by_equity_group__public_transport" not in columns
    assert "max_cost_by_equity_group__public_transport" not in columns
    assert "gap_absolute_description__public_transport" not in columns
    assert "gap_relative_description__public_transport" not in columns
    assert "inter_tertile_description__public_transport" not in columns


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_full_secondary_metrics_true_includes_dict_valued_columns(
    loaded_problem_with_equity_and_secondary_matrix, search_strategy
):
    result = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=1,
        search_strategy=search_strategy,
        full_secondary_metrics=True,
        grasp_num_solutions=1,
    )
    columns = result.solution_df.columns

    assert "weighted_by_equity_group__public_transport" in columns
    assert "unweighted_by_equity_group__public_transport" in columns
    assert "coverage_by_equity_group__public_transport" in columns
    assert "coverage_regions_by_equity_group__public_transport" in columns
    assert "max_cost_by_equity_group__public_transport" in columns
    assert "gap_absolute_description__public_transport" in columns
    assert "gap_relative_description__public_transport" in columns
    assert "inter_tertile_description__public_transport" in columns

    row = result.solution_df.iloc[0]
    # Every equity group (imd_decile 1/5/10) is present as a dict key, with
    # the value the demand-weighted public_transport cost for that group.
    assert set(row["weighted_by_equity_group__public_transport"].keys()) == {
        1,
        5,
        10,
    }
    assert isinstance(row["gap_absolute_description__public_transport"], str)


def test_full_secondary_metrics_values_match_uniform_site_cost(
    loaded_problem_with_equity_and_secondary_matrix,
):
    """Site_B's public_transport cost is uniformly 50.0 for every demand
    location, so every equity group's weighted/unweighted average for that
    row is exactly 50.0 -- a hand-checkable value, not just "some dict"."""
    result = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=1, full_secondary_metrics=True
    )
    site_b_row = _row_for_site(result.solution_df, "Site_B")
    assert site_b_row["weighted_by_equity_group__public_transport"] == {
        1: 50.0,
        5: 50.0,
        10: 50.0,
    }
    assert site_b_row["unweighted_by_equity_group__public_transport"] == {
        1: 50.0,
        5: 50.0,
        10: 50.0,
    }


def test_return_solution_metrics_full_secondary_metrics_direct_call(
    loaded_problem_with_equity_and_secondary_matrix,
):
    """The flag also works when calling return_solution_metrics() directly
    on an EvaluatedCombination, not just via solve()."""
    loaded_problem_with_equity_and_secondary_matrix.solve(p=1)  # builds travel_and_demand_df
    combination = (
        loaded_problem_with_equity_and_secondary_matrix.evaluate_single_solution_single_objective(
            site_names=["Site_B"]
        )
    )
    default_metrics = combination.return_solution_metrics()
    full_metrics = combination.return_solution_metrics(full_secondary_metrics=True)

    assert "weighted_by_equity_group__public_transport" not in default_metrics
    assert "weighted_by_equity_group__public_transport" in full_metrics
    assert full_metrics["weighted_by_equity_group__public_transport"] == {
        1: 50.0,
        5: 50.0,
        10: 50.0,
    }


# --- Plotting ---------------------------------------------------------


def test_plot_travel_time_distribution_matrix_kwarg(loaded_problem_with_secondary_matrix):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    fig = result.plot_travel_time_distribution(matrix="public_transport", top_n=1)
    assert fig is not None


def test_plot_travel_time_distribution_unknown_matrix_raises(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    with pytest.raises(ValueError, match="Unknown secondary travel matrix"):
        result.plot_travel_time_distribution(matrix="nonexistent")


def test_check_solution_equity_matrix_kwarg(
    loaded_problem_with_equity_and_secondary_matrix,
):
    result = loaded_problem_with_equity_and_secondary_matrix.solve(p=1)
    # Default ordering ranks Site_B first (primary weighted_average winner).
    summary = result.check_solution_equity(
        solution_rank=1, matrix="public_transport", return_plot=False
    )
    # Site_B's public_transport cost is uniformly 50 across all demand
    # locations, so every equity group's mean is exactly 50.
    assert (summary["min_cost"] == 50.0).all()


def test_check_solution_equity_unknown_matrix_raises(
    loaded_problem_with_equity_and_secondary_matrix,
):
    result = loaded_problem_with_equity_and_secondary_matrix.solve(p=1)
    with pytest.raises(ValueError, match="Unknown secondary travel matrix"):
        result.check_solution_equity(matrix="nonexistent", return_plot=False)
