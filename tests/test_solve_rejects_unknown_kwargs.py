"""
Tests pinning that solve() rejects unrecognised keyword arguments instead
of silently swallowing them.

solve()'s signature used to end with a bare `**kwargs` that was never
read or forwarded anywhere -- dead code that existed only to catch
anything not explicitly named. Its docstring even claimed "Additional
arguments passed to the internal solver", which was false: nothing was
ever passed anywhere. The practical effect was that any misspelled
keyword (e.g. `serach_strategy` instead of `search_strategy`, or
`treshold_for_coverage` instead of `threshold_for_coverage`) was silently
absorbed and ignored, and solve() ran with whatever default that
parameter has -- most dangerously, a misspelled `search_strategy` or
`max_value_cutoff` would silently produce a DIFFERENT solution than the
caller asked for, with no error or warning pointing at the typo.

Removing `**kwargs` lets Python's own machinery raise a clear TypeError
(with a "Did you mean...?" suggestion on modern Python) for any unknown
keyword.
"""

import pytest

import lokigi


def test_solve_raises_on_a_misspelled_search_strategy(loaded_problem):
    """The exact bug that motivated this fix: a typo'd search_strategy
    used to be silently ignored, falling back to the default
    'brute-force' instead of erroring."""
    with pytest.raises(TypeError, match="serach_strategy"):
        loaded_problem.solve(
            p=1,
            objectives="p_median",
            serach_strategy="grasp",
            show_progress=False,
        )


def test_solve_raises_on_a_misspelled_threshold_for_coverage(loaded_problem):
    with pytest.raises(TypeError, match="treshold_for_coverage"):
        loaded_problem.solve(
            p=1,
            objectives="mclp",
            treshold_for_coverage=15,
            show_progress=False,
        )


def test_solve_raises_on_a_completely_unknown_keyword(loaded_problem):
    with pytest.raises(TypeError, match="not_a_real_parameter"):
        loaded_problem.solve(
            p=1,
            objectives="p_median",
            not_a_real_parameter=True,
            show_progress=False,
        )


def test_solve_still_accepts_every_documented_keyword(loaded_problem_with_cost):
    """Sanity check that removing **kwargs didn't accidentally break any
    real, documented parameter -- every one of them should still be
    accepted together without raising."""
    result = loaded_problem_with_cost.solve(
        p=1,
        objectives="p_median",
        weights={"demand": 0.5, "cost": 0.5},
        capacitated=False,
        search_strategy="brute-force",
        brute_force_ignore_limit=False,
        show_progress=False,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        max_value_cutoff=None,
        threshold_for_coverage=None,
        grasp_num_solutions=5,
        grasp_alpha=0.2,
        grasp_max_attempts="default",
        grasp_min_sites_different=1,
        grasp_local_search_chance=0.8,
        grasp_max_swap_count_local_search=10,
        random_seed=42,
    )

    assert len(result.solution_df) >= 1
