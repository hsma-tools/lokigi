"""
Tests pinning that "demand" and "equity" weight keys are handled
case-insensitively end-to-end, matching the case-insensitive validation
solve() already performs.

These pin the fix for a silent-crash bug: solve()'s missing-key check
(site.py) validated weight keys case-insensitively ("Demand" was accepted
as a valid reference to the demand column), but the actual per-row
weighting logic in EvaluatedCombination (site_solutions.py) compared
labels with exact-case `==` in one place (col_name resolution) while using
`.lower()` a few lines later in the same function (direction resolution).
A key like "Demand" therefore passed validation, then silently matched no
real column: the row-level weight array stayed all zeros, and
`np.average(..., weights=compound_weights)` crashed with
`ZeroDivisionError: Weights sum to zero` -- a confusing error pointing
nowhere near the actual typo-adjacent cause.

`evaluate_single_solution_single_objective` never goes through solve()'s
validation at all, so the fix lives in EvaluatedCombination itself (the
root cause), with solve()'s own weight normalisation also canonicalising
case for defence in depth.

Additional-data labels are deliberately NOT case-normalised: they are
user-defined exact strings registered via add_additional_data(label=...),
not built-in keywords, so "CO2" continuing to mean something different
from "co2" is correct, matching-string behaviour, not a bug.
"""

import pytest

import lokigi


# --- solve(): case must not affect the result ---


def test_solve_with_uppercase_demand_weight_key_matches_lowercase(loaded_problem):
    lower = loaded_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 1.0},
    )
    upper = loaded_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"Demand": 1.0},
    )

    assert upper.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        lower.solution_df.iloc[0]["weighted_average"]
    )


def test_solve_with_uppercase_demand_key_matches_the_default_weights(loaded_problem):
    """A single {"Demand": 1.0} weight should take the same legacy
    fast-path as weights=None / the implicit default -- not just avoid
    crashing, but produce an identical result."""
    default = loaded_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
    )
    uppercase_demand = loaded_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"Demand": 1.0},
    )

    assert uppercase_demand.solution_df.iloc[0][
        "weighted_average"
    ] == pytest.approx(default.solution_df.iloc[0]["weighted_average"])


def test_solve_with_uppercase_equity_weight_key_matches_lowercase(
    loaded_problem_with_equity,
):
    lower = loaded_problem_with_equity.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"equity": 1.0},
    )
    upper = loaded_problem_with_equity.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"Equity": 1.0},
    )

    assert upper.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        lower.solution_df.iloc[0]["weighted_average"]
    )


def test_solve_with_mixed_case_blended_weights_matches_lowercase(
    loaded_problem_with_equity,
):
    lower = loaded_problem_with_equity.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.5, "equity": 0.5},
    )
    mixed_case = loaded_problem_with_equity.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"Demand": 0.5, "EQUITY": 0.5},
    )

    assert mixed_case.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        lower.solution_df.iloc[0]["weighted_average"]
    )


# --- direct evaluate_single_solution_single_objective(): bypasses solve() entirely ---


def test_direct_evaluation_with_uppercase_demand_weight_key(loaded_problem):
    """evaluate_single_solution_single_objective doesn't pass through
    solve()'s validation/normalisation, so the fix must live in
    EvaluatedCombination itself, not just solve()."""
    lower = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"], weights={"demand": 1.0}
    ).return_solution_metrics()
    upper = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"], weights={"Demand": 1.0}
    ).return_solution_metrics()

    assert upper["weighted_average"] == pytest.approx(lower["weighted_average"])


def test_direct_evaluation_with_uppercase_equity_weight_key(
    loaded_problem_with_equity,
):
    lower = loaded_problem_with_equity.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"], weights={"equity": 1.0}
    ).return_solution_metrics()
    upper = loaded_problem_with_equity.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"], weights={"Equity": 1.0}
    ).return_solution_metrics()

    assert upper["weighted_average"] == pytest.approx(lower["weighted_average"])


# --- additional-data labels remain case-sensitive on purpose ---


def test_additional_data_label_case_is_still_significant(
    loaded_problem_with_additional_data,
):
    """"co2" was registered via add_additional_data(label="co2"); unlike
    the built-in "demand"/"equity" keywords, a differently-cased weight
    key here is a genuine mismatch, not a formatting quirk -- it must
    still raise, not be silently accepted."""
    with pytest.raises(ValueError, match="does not correspond to demand"):
        loaded_problem_with_additional_data.evaluate_single_solution_single_objective(
            objective="p_median", site_names=["Site_A"], weights={"CO2": 1.0}
        )


def test_additional_data_label_lowercase_still_works(
    loaded_problem_with_additional_data,
):
    """Sanity check that the fixture's registered label still resolves
    correctly, so the case-sensitivity assertion above is meaningful
    rather than masking an unrelated failure."""
    result = loaded_problem_with_additional_data.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"], weights={"co2": 1.0}
    ).return_solution_metrics()

    assert result["weighted_average"] == pytest.approx(
        result["weighted_average"]
    )  # finite, comparable number


# --- adjacent bug found while fixing the above: an unrecognised weight
# key (not a case mismatch -- a genuine typo, with no additional data
# registered at all) must raise the intended "does not correspond to..."
# error rather than being silently skipped and crashing later with
# ZeroDivisionError. The direction-resolution/error-raising logic used to
# live INSIDE the `if col_name in columns:` check, so a label whose
# col_name was never a real column skipped straight past it. ---


def test_unrecognised_weight_key_raises_clearly_instead_of_crashing(loaded_problem):
    with pytest.raises(ValueError, match="does not correspond to demand"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_names=["Site_A"], weights={"populaton": 1.0}
        )
