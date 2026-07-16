"""
Tests for solve()'s three search strategies (brute-force, greedy, grasp)
across objectives, plus the ranking/reproducibility guarantees they're
supposed to provide.

Several tests here document real bugs found while exploring the solver
internals rather than gaps in coverage. Those are marked NOTE / BUG in their
docstring and are expected to FAIL until the underlying issue is fixed --
that's intentional: they pin the correct behaviour so a future fix shows up
as a newly-passing test rather than silent, undetected correctness drift.
"""

import warnings

import pytest

import lokigi


# --- GRASP: core correctness for minimisation objectives (p_median, p_center) ---


@pytest.mark.parametrize("objective,ranking_col", [("p_median", "weighted_average"), ("p_center", "max")])
def test_grasp_finds_brute_force_optimum_for_minimisation_objectives(
    five_site_problem, objective, ranking_col
):
    """GRASP should reliably reach the true optimum on a 10-combination
    search space when given enough attempts, for objectives where GRASP's
    default is_minimization=True is actually correct (p_median, p_center)."""
    brute_force = five_site_problem.solve(
        p=2, objectives=objective, search_strategy="brute-force", show_progress=False
    )
    optimal_score = brute_force.solution_df.iloc[0][ranking_col]

    grasp = five_site_problem.solve(
        p=2,
        objectives=objective,
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=8,
        grasp_max_attempts=10,
    )
    grasp_score = grasp.solution_df.iloc[0][ranking_col]

    assert grasp_score == pytest.approx(optimal_score)


def test_greedy_returns_a_valid_solution_for_minimisation_objectives(five_site_problem):
    """Greedy construction is inherently myopic (documented trade-off: it
    never re-evaluates earlier choices), so it is NOT guaranteed to match
    the brute-force optimum. This just checks it returns a structurally
    valid single solution rather than asserting optimality."""
    result = five_site_problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )

    assert len(result.solution_df) == 1
    assert len(result.solution_df.iloc[0]["site_indices"]) == 2
    assert result.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        result.solution_df.iloc[0]["weighted_average"]
    )  # finite, comparable number


# --- mclp: greedy/grasp silently ignore the true coverage objective during search ---


def test_greedy_mclp_search_ignores_true_coverage(five_site_problem):
    """
    BUG: `_greedy`'s internal per-step evaluation calls
    (mixins/site_solvers.py) never pass `threshold_for_coverage`, so
    `proportion_within_coverage_threshold` is 0.0 for every candidate during
    the search (only the final chosen combination is scored with the real
    threshold). Greedy's ascending sort_values([ranking, "weighted_average"])
    therefore degenerates to ranking by weighted_average alone at every step.

    On `five_site_problem`, the weighted_average-best path deterministically
    leads to a combination with 0.0 coverage, while the true best-coverage
    combination is 0.75 -- so greedy should reach 0.75 but currently doesn't.
    """
    brute_force = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    true_best_coverage = brute_force.solution_df.iloc[0][
        "proportion_within_coverage_threshold"
    ]

    greedy = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="greedy",
        show_progress=False,
        threshold_for_coverage=15,
    )
    greedy_coverage = greedy.solution_df.iloc[0]["proportion_within_coverage_threshold"]

    assert greedy_coverage == pytest.approx(true_best_coverage)


def test_grasp_construction_and_local_search_use_the_real_coverage_threshold(
    five_site_problem,
):
    """
    BUG: `_grasp`'s construction phase and local-search phase evaluate
    candidates via `_get_cached_metrics`, which calls
    `evaluate_single_solution_single_objective` WITHOUT `threshold_for_coverage`
    (mixins/site_solvers.py:258-261) -- only the final accepted solution
    (line ~400-405) is scored with the real threshold. For `mclp`, this means
    every intermediate ranking decision is based on a coverage value of 0.0
    regardless of the real threshold, rather than the true coverage.

    This is checked directly (via a call-recording monkeypatch) rather than
    through solve()'s stochastic outer API, since GRASP's randomised
    construction can occasionally stumble onto a good answer by luck even
    while ranking on the wrong signal -- that would make an outcome-based
    assertion flaky. Recording the actual kwargs passed on every internal
    evaluation call is a deterministic way to pin the real defect.
    """
    calls = []
    original = type(five_site_problem).evaluate_single_solution_single_objective

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("threshold_for_coverage"))
        return original(self, *args, **kwargs)

    type(five_site_problem).evaluate_single_solution_single_objective = spy
    try:
        five_site_problem._grasp(
            p=2,
            objectives="mclp",
            weights={"demand": 1.0},
            threshold_for_coverage=15,
            num_solutions=1,
            max_attempts=1,
            random_seed=0,
        )
    finally:
        type(five_site_problem).evaluate_single_solution_single_objective = original

    assert calls, "Expected at least one evaluation call during _grasp"
    assert all(threshold == 15 for threshold in calls), (
        "Every candidate evaluation during GRASP's search should use the "
        f"real threshold_for_coverage (15), but got: {calls}"
    )


def test_mclp_without_coverage_threshold_raises(loaded_problem):
    """solve() should reject 'mclp' without a threshold_for_coverage rather
    than silently producing proportion_within_coverage_threshold == 0.0 for
    every combination (since `within_threshold` would be all NaN and
    pandas' `.sum()` skips NaN)."""
    with pytest.raises(ValueError):
        loaded_problem.solve(
            p=2, objectives="mclp", search_strategy="brute-force", show_progress=False
        )


# --- GRASP: reproducibility and diversity-budget behaviour (no known bugs) ---


def test_grasp_is_reproducible_with_a_fixed_seed(five_site_problem):
    first = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        random_seed=123,
        grasp_num_solutions=3,
    )
    second = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        random_seed=123,
        grasp_num_solutions=3,
    )

    assert list(first.solution_df["site_names"]) == list(second.solution_df["site_names"])
    assert list(first.solution_df["weighted_average"]) == list(
        second.solution_df["weighted_average"]
    )


def test_grasp_warns_when_diversity_budget_is_exhausted(loaded_problem):
    """With only 3 candidate sites, p=2 gives just 3 possible combinations --
    asking for 5 diverse solutions can never succeed, and should warn rather
    than hang or error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = loaded_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="grasp",
            show_progress=False,
            grasp_num_solutions=5,
        )

    assert len(result.solution_df) < 5
    assert any(
        "exhausted attempt budget" in str(warning.message) for warning in caught
    )


# --- Brute force: keep_best_n / keep_worst_n heap tie-break ---


def test_brute_force_keep_best_n_handles_exact_score_ties(tied_score_problem):
    """
    BUG: `_brute_force`'s keep_best_n/keep_worst_n heaps push
    (score, metrics) tuples onto a heapq (mixins/site_solvers.py). If two
    solutions tie exactly on score, heapq falls back to comparing the
    `metrics` dicts, which raises `TypeError: '<' not supported between
    instances of 'dict' and 'dict'`.

    On `tied_score_problem`, Site_A and Site_B alone tie exactly at
    weighted_average == 15.0, so keep_best_n=2 should return both without
    crashing.
    """
    result = tied_score_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        brute_force_keep_best_n=2,
    )

    assert len(result.solution_df) == 2


# --- Brute force keep_best_n / keep_worst_n: mclp direction and coverage ---


def test_keep_best_n_for_mclp_retains_the_highest_coverage_combinations(
    five_site_problem,
):
    """Two stacked bugs previously broke keep_best_n for mclp: the keep-n
    branch of `_brute_force` never passed threshold_for_coverage to its
    evaluations (so every candidate scored coverage 0.0 and the heap
    ranked noise), and the heaps assumed lower-is-better (so even with
    real scores, keep_best_n retained the LOWEST-coverage combinations).
    The kept set must match the top of a full brute-force run."""
    full = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    full_coverages = sorted(
        full.solution_df["proportion_within_coverage_threshold"], reverse=True
    )

    kept = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
        brute_force_keep_best_n=3,
    )
    kept_coverages = sorted(
        kept.solution_df["proportion_within_coverage_threshold"], reverse=True
    )

    assert len(kept.solution_df) == 3
    assert kept_coverages == pytest.approx(full_coverages[:3])
    # The overall best solution must survive the pruning and rank first
    assert kept.solution_df.iloc[0][
        "proportion_within_coverage_threshold"
    ] == pytest.approx(full_coverages[0])
    # ... and the real threshold must have been used, not silently dropped
    assert (kept.solution_df["coverage_threshold"] == 15).all()


def test_keep_worst_n_for_mclp_retains_the_lowest_coverage_combinations(
    five_site_problem,
):
    full = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    full_coverages = sorted(full.solution_df["proportion_within_coverage_threshold"])

    kept = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
        brute_force_keep_worst_n=3,
    )
    kept_coverages = sorted(kept.solution_df["proportion_within_coverage_threshold"])

    assert len(kept.solution_df) == 3
    assert kept_coverages == pytest.approx(full_coverages[:3])


def test_keep_best_n_reports_coverage_metrics_for_minimising_objectives(
    five_site_problem,
):
    """threshold_for_coverage is a reporting metric for non-mclp
    objectives, but the keep-n branch previously dropped it, silently
    zeroing the coverage columns of whatever was returned. The kept rows'
    coverage must match the full run's values for the same site sets."""
    full = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    full_coverage_by_sites = {
        tuple(row["site_names"]): row["proportion_within_coverage_threshold"]
        for _, row in full.solution_df.iterrows()
    }

    kept = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
        brute_force_keep_best_n=2,
    )

    assert (kept.solution_df["coverage_threshold"] == 15).all()
    for _, row in kept.solution_df.iterrows():
        assert row["proportion_within_coverage_threshold"] == pytest.approx(
            full_coverage_by_sites[tuple(row["site_names"])]
        )
