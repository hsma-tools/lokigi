"""
Tests for solve()'s weight validation, normalisation, and custom
multi-key weight blending.

As with test_search_strategies.py, some tests here pin *correct* behaviour
for bugs found while exploring the code and are expected to FAIL until
fixed -- see each docstring for the specific defect.
"""

import pytest


# --- basic weight validation (currently correct -- lock these in) ---


def test_negative_weight_raises(loaded_problem):
    with pytest.raises(ValueError, match="cannot be negative"):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": -1.0},
        )


def test_zero_sum_weights_raises(loaded_problem):
    with pytest.raises(ValueError, match="greater than zero"):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.0},
        )


def test_non_dict_weights_raises_type_error(loaded_problem):
    with pytest.raises(TypeError, match="Expected 'weights' to be a dict"):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights=[1.0],
        )


# --- unknown weight labels ---


def test_unknown_weight_label_raises_a_clear_error(loaded_problem):
    """An unrecognized weight label should raise a clear, catchable error --
    whether caught early by solve()'s weight-key validation (site.py) or, if
    it slips past that (e.g. a direct EvaluatedCombination construction),
    by the direction-resolution check in site_solutions.py."""
    with pytest.raises((ValueError, KeyError)):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"min_cost": 0.5, "demand": 0.5},
        )


# --- custom multi-key weight blending changes the result ---


def test_custom_equity_weighting_changes_the_ranking(loaded_problem_with_equity):
    """End-to-end coverage of EvaluatedCombination's compound-weight
    blending (site_solutions.py:162-233): mixing in an equity weight should
    produce a different weighted_average (and potentially a different best
    combination) than demand-only weighting."""
    demand_only = loaded_problem_with_equity.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 1.0},
    )
    blended = loaded_problem_with_equity.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.5, "equity": 0.5},
    )

    demand_only_best = demand_only.solution_df.iloc[0]["weighted_average"]
    blended_best = blended.solution_df.iloc[0]["weighted_average"]

    assert blended_best != pytest.approx(demand_only_best)


# --- weight-rejection guard for p_center ---


def test_p_center_rejects_custom_weights(loaded_problem):
    """p_center ranks solely by worst-case travel time (max), so a weights
    dict has no effect on the outcome -- solve() should reject it explicitly
    rather than silently accepting and ignoring it."""
    with pytest.raises(ValueError):
        loaded_problem.solve(
            p=2,
            objectives="p_center",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 1.0},
        )


def test_p_center_works_fine_with_default_weights(loaded_problem):
    """The implicit demand-only default (no `weights` argument at all)
    should keep working -- only an explicitly-passed weights dict is
    rejected."""
    result = loaded_problem.solve(
        p=2, objectives="p_center", search_strategy="brute-force", show_progress=False
    )
    assert len(result.solution_df) == 3


# --- max_value_cutoff (hybrid objectives only) ---


def test_max_value_cutoff_filters_combinations_exceeding_it(loaded_problem):
    unfiltered = loaded_problem.solve(
        p=2,
        objectives="hybrid_p_median",
        search_strategy="brute-force",
        show_progress=False,
    )
    assert 25.0 in list(unfiltered.solution_df["max"])

    filtered = loaded_problem.solve(
        p=2,
        objectives="hybrid_p_median",
        search_strategy="brute-force",
        show_progress=False,
        max_value_cutoff=20,
    )

    assert 25.0 not in list(filtered.solution_df["max"])
    assert all(value <= 20 for value in filtered.solution_df["max"])


def test_max_value_cutoff_rejected_for_non_hybrid_objective(loaded_problem):
    with pytest.raises(ValueError):
        loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            max_value_cutoff=20,
        )
