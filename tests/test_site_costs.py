"""
Tests for site-cost support: flexible cost column naming in add_sites(),
always-on total_cost reporting in solutions_df, and cost only influencing
which solution is selected when explicitly passed as a weight.
"""

import pandas as pd
import pytest


# --- add_sites(cost_col=...) storage / flexible naming ---


def test_add_sites_stores_the_configured_cost_col_name(loaded_problem_with_cost):
    """Flexible naming: the attribute should hold the user's actual column
    name ("build_cost"), not a hardcoded "cost"."""
    assert loaded_problem_with_cost._candidate_sites_cost_col == "build_cost"


def test_add_sites_without_cost_col_leaves_attribute_none(loaded_problem):
    assert loaded_problem._candidate_sites_cost_col is None


def test_setup_sites_df_from_travel_matrix_fallback_sets_cost_col_none(
    basic_problem, demand_df, travel_df
):
    """When add_sites() is never called, solve() auto-derives candidate
    sites from the travel matrix columns -- that fallback path must also
    leave _candidate_sites_cost_col as None so downstream code can always
    safely read the attribute.

    The fallback requires the travel matrix's source column to share a name
    with the demand id column, so the travel_df fixture (source_col=
    "source_id") is renamed here to match demand_df's "location_id".
    """
    basic_problem.add_demand(
        demand_df, demand_col="demand", location_id_col="location_id"
    )
    renamed_travel_df = travel_df.rename(columns={"source_id": "location_id"})
    basic_problem.add_travel_matrix(renamed_travel_df, source_col="location_id")

    with pytest.warns(UserWarning, match="No candidate site dataframe was given"):
        basic_problem.solve(p=2, objectives="p_median", show_progress=False)

    assert basic_problem._candidate_sites_cost_col is None


def test_add_sites_with_missing_cost_col_raises_value_error(
    basic_problem, candidate_df
):
    with pytest.raises(ValueError, match="missing these columns"):
        basic_problem.add_sites(
            candidate_df, candidate_id_col="site_id", cost_col="nonexistent"
        )


def test_add_sites_with_non_numeric_cost_col_raises_type_error(basic_problem):
    non_numeric_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.5, 51.6],
            "long": [-0.1, -0.2],
            "build_cost": ["expensive", "cheap"],
        }
    )
    with pytest.raises(TypeError, match="must contain numbers"):
        basic_problem.add_sites(
            non_numeric_df, candidate_id_col="site_id", cost_col="build_cost"
        )


# --- total_cost correctness ---


def test_total_cost_correct_for_a_single_evaluated_combination(
    loaded_problem_with_cost,
):
    result = loaded_problem_with_cost.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    ).return_solution_metrics()

    # Site_A (10.0) + Site_B (500.0)
    assert result["total_cost"] == pytest.approx(510.0)


def test_total_cost_correct_across_all_combinations_via_solve(
    loaded_problem_with_cost,
):
    expected_by_combo = {
        frozenset({"Site_A", "Site_B"}): 510.0,
        frozenset({"Site_A", "Site_C"}): 60.0,
        frozenset({"Site_B", "Site_C"}): 550.0,
    }

    result = loaded_problem_with_cost.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    assert len(result.solution_df) == len(expected_by_combo)
    for _, row in result.solution_df.iterrows():
        combo = frozenset(row["site_names"])
        assert combo in expected_by_combo
        assert row["total_cost"] == pytest.approx(expected_by_combo[combo])


# --- optionality: everything still works with no cost_col ---


def test_total_cost_is_nan_when_no_cost_col_configured(loaded_problem):
    result = loaded_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    assert len(result.solution_df) == 3
    assert result.solution_df["total_cost"].isna().all()


def test_weights_cost_without_cost_col_raises_key_error(loaded_problem):
    with pytest.raises(KeyError):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.5, "cost": 0.5},
        )


def test_weights_cost_only_does_not_crash(loaded_problem_with_cost):
    """Regression test: a weights dict containing only "cost" (no "demand"
    or "equity") used to fall through EvaluatedCombination's row-level
    blending loop with an all-zero compound_weights array, raising
    ZeroDivisionError from np.average. "cost" must be excluded from that
    loop and handled purely at the combination-comparison level."""
    result = loaded_problem_with_cost.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"cost": 1.0},
    )
    assert all(
        result.solution_df["weighted_average"].apply(lambda v: v == v)  # not NaN
    )


# --- cost does not influence the solution unless weighted ---


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_cost_does_not_change_winner_without_a_cost_weight(
    cost_flips_winner_problem, search_strategy
):
    result = cost_flips_winner_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy=search_strategy,
        show_progress=False,
        random_seed=42,
    )
    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_Fast"]
    assert best["weighted_average"] == pytest.approx(10.0)


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_cost_does_not_change_winner_with_demand_only_weight(
    cost_flips_winner_problem, search_strategy
):
    """Explicitly passing weights={"demand": 1.0} (the legacy default) must
    behave identically to omitting weights entirely -- cost still has no
    say in the outcome."""
    result = cost_flips_winner_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy=search_strategy,
        show_progress=False,
        random_seed=42,
        weights={"demand": 1.0},
    )
    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_Fast"]


# --- cost does influence the solution when weighted ---


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_cost_flips_the_winner_when_weighted(cost_flips_winner_problem, search_strategy):
    result = cost_flips_winner_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy=search_strategy,
        show_progress=False,
        random_seed=42,
        weights={"demand": 0.1, "cost": 0.9},
    )
    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_Cheap"]


@pytest.mark.parametrize("cost_key", ["Cost", "COST", "CoSt"])
def test_cost_flips_the_winner_regardless_of_weight_key_casing(
    cost_flips_winner_problem, cost_key
):
    """Regression test: the "cost" weight key used to be validated
    case-insensitively but applied via an exact-case dict lookup, so a
    differently-cased key (e.g. "Cost") passed validation but silently
    never influenced the solution. Brute-force only, since this is about
    weight-key handling, not search-strategy behaviour."""
    result = cost_flips_winner_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        random_seed=42,
        weights={"demand": 0.1, cost_key: 0.9},
    )
    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_Cheap"]
