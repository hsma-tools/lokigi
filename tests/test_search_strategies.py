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

import itertools
import math
import random
import warnings

import pandas as pd
import pytest

import lokigi
from lokigi.utils import _resolve_ranking_metric


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
            scorer=_resolve_ranking_metric(objective="mclp")[0],
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


@pytest.mark.parametrize(
    "n_required,expected_total_combinations",
    [
        (3, math.comb(7, 2)),  # 10 sites, 3 required, p=5 -> free slots C(7,2)=21
        (0, math.comb(10, 5)),  # no required sites -> unrestricted C(10,5)=252
    ],
    ids=["with_required_sites", "no_required_sites"],
)
def test_grasp_total_combinations_uses_required_site_adjusted_math_comb(
    monkeypatch, n_required, expected_total_combinations
):
    """GRASP's default `max_attempts` is capped at `total_combinations =
    math.comb(total_n_sites - n_required, p - n_required)`
    (mixins/site_solvers.py:591-596) -- the free search space once required
    sites are pinned, not the raw C(n, p). Forgetting the required-site
    adjustment would instead compute C(10, 5)=252 for the first case.

    `total_combinations`/`max_attempts` are local variables never exposed
    on the result, so this counts calls to `random.Random.randint` instead:
    it's the only `.randint(` call in the file, made exactly once per
    while-loop iteration (attempt), immediately before `attempts += 1` --
    an exact, external attempt counter.

    grasp_num_solutions=300 (budget 300*20=6000) exceeds both 21 and 252
    either way, and since at most 252 distinct combinations can ever exist
    across both parametrized cases, GRASP can never accept 300 diverse
    solutions in either -- so it's guaranteed to run until attempts ==
    max_attempts == total_combinations, making the attempt count a
    deterministic proxy for the value under test regardless of how easy
    or hard the diversity constraint happens to be to satisfy.
    """
    n_sites = 10
    p = 5
    demand_df = pd.DataFrame({"location_id": ["LSOA_1"], "demand": [100]})
    candidate_df = pd.DataFrame(
        {
            "site_id": [f"Site_{i}" for i in range(n_sites)],
            "lat": [51.0 + 0.01 * i for i in range(n_sites)],
            "long": [-0.01 * (i + 1) for i in range(n_sites)],
            "required": ["yes"] * n_required + ["no"] * (n_sites - n_required),
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1"],
            **{f"Site_{i}": [float(i + 1)] for i in range(n_sites)},
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(
        candidate_df, candidate_id_col="site_id", required_sites_col="required"
    )
    problem.add_travel_matrix(travel_df, source_col="source_id")

    call_count = 0
    original_randint = random.Random.randint

    def counting_randint(self, a, b):
        nonlocal call_count
        call_count += 1
        return original_randint(self, a, b)

    monkeypatch.setattr(random.Random, "randint", counting_randint)

    problem.solve(
        p=p,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=300,
        grasp_max_attempts="default",
    )

    assert call_count == expected_total_combinations


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


# --- Brute force keep_best_n / keep_worst_n with cost weighting active ---
#
# `_brute_force`'s keep_best_n/keep_worst_n heap prunes to N using the raw,
# pre-cost ranking value one combination at a time. Cost weighting only
# gets applied afterwards, over whatever tiny subset the heap let survive
# -- so a combination that only looks good once cost is blended in, but has
# a mediocre raw ranking, was silently discarded before cost was ever
# considered. Whenever a positive "cost" weight is active, keep_best_n/
# keep_worst_n must instead materialise every combination and blend cost in
# over the full batch before pruning.


def test_brute_force_keep_best_n_with_cost_weighting_retains_the_true_cost_weighted_best(
    cost_weighted_pruning_problem,
):
    """Site_Cheap has the worst raw weighted_average (15.0, dead last of 4)
    but becomes the true cost-weighted best once a heavy cost weight is
    applied. keep_best_n=1 must still find it, not whatever the raw-ranking
    heap would have kept."""
    full = cost_weighted_pruning_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.05, "cost": 0.95},
    )
    assert full.solution_df.iloc[0]["site_names"] == ["Site_Cheap"]

    with pytest.warns(UserWarning, match="brute_force_keep_best_n"):
        pruned = cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_best_n=1,
        )

    assert pruned.solution_df.iloc[0]["site_names"] == ["Site_Cheap"]


def test_brute_force_keep_worst_n_with_cost_weighting_retains_the_true_cost_weighted_worst(
    cost_weighted_pruning_problem,
):
    """Site_Mid is the true cost-weighted worst: unlike Site_Fast1/Fast2, it
    is bad on BOTH dimensions at once (a mediocre 12.0 raw travel time, on
    top of the same 1000.0 build_cost) which, once blended
    (composite_score = 0.05*primary_badness + 0.95*cost_badness), gives it
    a worse composite_score (~0.984) than Fast1 (~0.949) or Fast2
    (~0.954). keep_worst_n=1 must find Site_Mid, not whatever the
    raw-ranking heap (which only sees Site_Mid's better-than-Cheap raw
    travel time) would have kept."""
    with pytest.warns(UserWarning, match="brute_force_keep_worst_n"):
        pruned = cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_worst_n=1,
        )

    assert pruned.solution_df.iloc[0]["site_names"] == ["Site_Mid"]


def test_brute_force_keep_best_n_and_keep_worst_n_combined_with_cost_weighting(
    cost_weighted_pruning_problem,
):
    """Both flags set at once should still return best + worst, correctly
    cost-weighted, via the same code path (mixins/site_solvers.py's
    `best_list + worst_list` / concatenated-DataFrame return contract):
    Site_Cheap (best) and Site_Mid (worst, see the keep_worst_n test
    above)."""
    with pytest.warns(UserWarning):
        result = cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_best_n=1,
            brute_force_keep_worst_n=1,
        )

    kept_site_names = {row["site_names"][0] for _, row in result.solution_df.iterrows()}
    assert len(result.solution_df) == 2
    assert kept_site_names == {"Site_Cheap", "Site_Mid"}


def test_brute_force_cost_weighted_keep_best_n_matches_full_run_top_n(
    cost_weighted_five_site_problem,
):
    """On a larger (10-combination) adversarial search space, a pruned
    cost-weighted keep_best_n=N run must retain exactly the same top-N
    combinations as a full, unpruned run -- identified by which sites were
    picked, not by comparing composite_score values directly: solve()
    redundantly re-applies cost weighting once more, over just the
    returned subset, for final display ordering (see
    `_solve_pmedian_pcenter_mclp_problem`), which renormalizes the score
    scale without changing which combinations were selected."""
    weights = {"demand": 0.3, "cost": 0.7}

    full = cost_weighted_five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights=weights,
    )
    full_top_4_site_sets = {
        frozenset(row["site_names"])
        for _, row in full.solution_df.head(4).iterrows()
    }

    with pytest.warns(UserWarning, match="brute_force_keep_best_n"):
        kept = cost_weighted_five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights=weights,
            brute_force_keep_best_n=4,
        )
    kept_site_sets = {
        frozenset(row["site_names"]) for _, row in kept.solution_df.iterrows()
    }

    assert len(kept.solution_df) == 4
    assert kept_site_sets == full_top_4_site_sets


def test_brute_force_cost_weighted_keep_best_n_respects_max_value_cutoff(
    cost_weighted_five_site_problem,
):
    """max_value_cutoff filtering (infeasible combinations dropped during
    collection) must still apply when cost weighting forces keep_best_n
    into the full-materialization path."""
    weights = {"demand": 0.3, "cost": 0.7}

    full = cost_weighted_five_site_problem.solve(
        p=2,
        objectives="hybrid_p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights=weights,
        max_value_cutoff=30,
    )
    assert (full.solution_df["max"] <= 30).all()

    with pytest.warns(UserWarning, match="brute_force_keep_best_n"):
        kept = cost_weighted_five_site_problem.solve(
            p=2,
            objectives="hybrid_p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights=weights,
            max_value_cutoff=30,
            brute_force_keep_best_n=2,
        )

    assert (kept.solution_df["max"] <= 30).all()
    assert len(kept.solution_df) == 2


def test_brute_force_cost_weighted_keep_best_n_handles_exact_score_ties(
    tied_score_problem_with_cost,
):
    """Site_A and Site_B tie exactly on both weighted_average and
    build_cost, so they also tie exactly on composite_score once cost is
    blended in -- both scoring strictly worse than Site_C, so
    keep_worst_n=2 must return exactly {Site_A, Site_B}. The pandas-based
    (nsmallest/nlargest) cost-weighted path must handle this tie without
    crashing, unlike the heap-based path's need for an explicit
    tie-breaker."""
    with pytest.warns(UserWarning, match="brute_force_keep_worst_n"):
        result = tied_score_problem_with_cost.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_worst_n=2,
        )

    assert len(result.solution_df) == 2
    assert set(row["site_names"][0] for _, row in result.solution_df.iterrows()) == {
        "Site_A",
        "Site_B",
    }


# --- Brute force keep_best_n / keep_worst_n: cost-fallback warning presence ---
#
# The warning is meant to fire exactly when BOTH keep_best_n/keep_worst_n
# AND a positive cost weight are active together (the combination that
# forces `_brute_force` to give up keep_n's memory-saving benefit). It
# must stay silent for every other combination: keep_n alone, cost
# weighting alone, or a "cost" key present but weighted at 0.


def _cost_fallback_warnings(caught):
    return [
        w
        for w in caught
        if "was requested together with a cost weight" in str(w.message)
    ]


def test_warns_when_keep_best_n_and_cost_weighting_are_both_active(
    cost_weighted_pruning_problem,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_best_n=1,
        )

    assert len(_cost_fallback_warnings(caught)) == 1


def test_warns_when_keep_worst_n_and_cost_weighting_are_both_active(
    cost_weighted_pruning_problem,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_worst_n=1,
        )

    assert len(_cost_fallback_warnings(caught)) == 1


def test_warns_exactly_once_when_keep_best_n_and_keep_worst_n_are_both_active_with_cost_weighting(
    cost_weighted_pruning_problem,
):
    """Both pruning flags set at once must still only warn once -- the
    warning is about the fallback being triggered for this call, not one
    per flag."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
            brute_force_keep_best_n=1,
            brute_force_keep_worst_n=1,
        )

    assert len(_cost_fallback_warnings(caught)) == 1


def test_no_cost_fallback_warning_when_keep_best_n_is_used_without_cost_weighting(
    five_site_problem,
):
    """keep_best_n alone (no weights at all) must keep using the
    memory-efficient streaming heap, silently -- the fallback and its
    warning are specific to cost weighting being active."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            brute_force_keep_best_n=3,
        )

    assert _cost_fallback_warnings(caught) == []


def test_no_cost_fallback_warning_when_keep_worst_n_is_used_without_cost_weighting(
    five_site_problem,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            brute_force_keep_worst_n=3,
        )

    assert _cost_fallback_warnings(caught) == []


def test_no_cost_fallback_warning_when_keep_n_is_used_with_a_demand_only_weight(
    cost_weighted_pruning_problem,
):
    """A weights dict that omits "cost" entirely (even though the problem
    has a cost_col configured) must not trigger the fallback -- only an
    actual positive "cost" weight should."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 1.0},
            brute_force_keep_best_n=1,
        )

    assert _cost_fallback_warnings(caught) == []


def test_no_cost_fallback_warning_when_cost_weight_is_zero(
    cost_weighted_pruning_problem,
):
    """weights={"cost": 0} is present but not "positive" -- the same
    `weights.get("cost", 0) > 0` check used everywhere else in this
    codebase for "is cost weighting actually active" must treat it as
    inactive, leaving keep_best_n on the cheap streaming path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 1.0, "cost": 0},
            brute_force_keep_best_n=1,
        )

    assert _cost_fallback_warnings(caught) == []


def test_no_cost_fallback_warning_when_cost_weighting_is_used_without_keep_n(
    cost_weighted_pruning_problem,
):
    """Cost weighting alone (no keep_best_n/keep_worst_n at all) already
    materialises every combination -- there is nothing to fall back from,
    so no warning should fire."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_weighted_pruning_problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"demand": 0.05, "cost": 0.95},
        )

    assert _cost_fallback_warnings(caught) == []


# --- Brute force n_jobs: parallel results must match serial exactly ---


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"brute_force_keep_best_n": 3},
        {"brute_force_keep_worst_n": 3},
        {"brute_force_keep_best_n": 2, "brute_force_keep_worst_n": 2},
    ],
    ids=["plain", "keep_best_n", "keep_worst_n", "keep_both"],
)
def test_brute_force_n_jobs_matches_serial(five_site_problem, extra_kwargs):
    """n_jobs is purely a performance knob -- parallel brute force must
    return exactly the same solutions as the serial (n_jobs=1) run, for
    the plain, keep_best_n, keep_worst_n, and keep-both cases."""
    serial = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=1,
        **extra_kwargs,
    )
    parallel = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=2,
        **extra_kwargs,
    )

    serial_indices = sorted(map(tuple, serial.solution_df["site_indices"]))
    parallel_indices = sorted(map(tuple, parallel.solution_df["site_indices"]))
    assert serial_indices == parallel_indices
    assert sorted(serial.solution_df["weighted_average"]) == pytest.approx(
        sorted(parallel.solution_df["weighted_average"])
    )


@pytest.mark.parametrize("n_jobs", [2, 3, 4, -1])
def test_brute_force_n_jobs_keep_best_n_matches_serial_under_exact_ties(
    many_tied_sites_problem, n_jobs
):
    """30 combinations, drawn from a handful of exactly-repeated
    weighted_average values (the best value, 3.0, is shared by 8 of them
    -- more than keep_best_n=5), get split across several chunks.

    Note on what this actually guarantees: because 3.0 is the *global*
    best value here, no strictly-better combination ever exists to evict
    an already-kept 3.0, so which 5 of the 8 survive is decided purely by
    encounter order -- and since chunks are contiguous slices processed in
    ascending global-index order, that order matches a single-process run
    exactly. This is the realistic case (e.g. several candidate sites at
    the identical minimum distance) and this test pins it, across several
    n_jobs/chunk-count combinations. It is NOT a general guarantee: a tied
    cluster competing mid-stream against interleaved strictly-better
    combinations (not present in this fixture) can in principle see a
    different, equally-valid tied combination survive under n_jobs>1 than
    a serial run would keep -- see _push_capped's docstring and
    test_brute_force_n_jobs_may_pick_a_different_tied_combination_mid_stream
    below for a fixture where that actually happens. n_jobs=1 has no such
    caveat: it always evaluates as a single chunk/heap, byte-for-byte
    reproducing prior (pre-parallelisation) behaviour, which is pinned
    separately by
    test_backtest_tied_score_problem_p_median_brute_force_keep_best_n.
    """
    serial = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=1,
        brute_force_keep_best_n=5,
    )
    parallel = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=n_jobs,
        brute_force_keep_best_n=5,
    )

    serial_indices = sorted(map(tuple, serial.solution_df["site_indices"]))
    parallel_indices = sorted(map(tuple, parallel.solution_df["site_indices"]))
    assert len(serial_indices) == 5
    assert serial_indices == parallel_indices


@pytest.mark.parametrize("n_jobs", [2, 3, 4, -1])
def test_brute_force_n_jobs_keep_worst_n_matches_serial_under_exact_ties(
    many_tied_sites_problem, n_jobs
):
    """Symmetric to the keep_best_n exact-ties test above, but for
    keep_worst_n: the *worst* (highest travel time) value in this fixture,
    12.0, is shared by 3 combinations -- more than keep_worst_n=2. The
    bottom_n_heap branch in _brute_force's merge loop
    (mixins/site_solvers.py) is separate code from the top_n_heap branch
    and was untested under exact ties before this test; the same "global
    extreme value" argument applies (nothing is worse than 12.0, so
    nothing ever evicts an already-kept 12.0, so survivorship depends only
    on encounter order, which matches a serial run since chunks are
    processed in ascending global-index order)."""
    serial = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=1,
        brute_force_keep_worst_n=2,
    )
    parallel = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=n_jobs,
        brute_force_keep_worst_n=2,
    )

    serial_indices = sorted(map(tuple, serial.solution_df["site_indices"]))
    parallel_indices = sorted(map(tuple, parallel.solution_df["site_indices"]))
    assert len(serial_indices) == 2
    assert serial_indices == parallel_indices


def test_brute_force_n_jobs_may_pick_a_different_tied_combination_mid_stream(
    many_tied_sites_problem,
):
    """Documents (and pins, so it can't silently regress further) the
    actual, narrower guarantee n_jobs>1 provides under exact ties, using a
    fixture/parameter combination verified to genuinely diverge from a
    serial run: brute_force_keep_best_n=3 with n_jobs=3 on
    many_tied_sites_problem keeps site indices {0, 2, 9} serially but
    {0, 2, 10} in parallel -- both index 9 and index 10 score exactly 3.0,
    and 3.0 is NOT the sole occupant of the boundary here (there are 8
    combinations tied at 3.0 competing for 3 slots), so which one survives
    depends on how chunk boundaries happen to split that tied group.

    This is expected and documented, not a bug: n_jobs>1 always returns
    the correct COUNT and the correct SCORES (this test asserts both), it
    just doesn't promise identical tie-break IDENTITY to a serial run in
    this narrower case. Contrast with the keep_best_n=5 case above, where
    3.0 is the sole value at the boundary and identity is preserved."""
    serial = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=1,
        brute_force_keep_best_n=3,
    )
    parallel = many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=3,
        brute_force_keep_best_n=3,
    )

    assert len(parallel.solution_df) == len(serial.solution_df) == 3
    assert sorted(parallel.solution_df["weighted_average"]) == pytest.approx(
        sorted(serial.solution_df["weighted_average"])
    )
    # The specific combination kept is NOT guaranteed to match -- see
    # docstring. This fixture/parameter combination is a verified case
    # where it genuinely doesn't.
    serial_indices = sorted(map(tuple, serial.solution_df["site_indices"]))
    parallel_indices = sorted(map(tuple, parallel.solution_df["site_indices"]))
    assert serial_indices != parallel_indices


def test_brute_force_n_jobs_one_evaluates_as_a_single_chunk(
    many_tied_sites_problem, monkeypatch
):
    """Pins the specific invariant that makes n_jobs=1 byte-for-byte
    identical to pre-parallelisation behaviour (mixins/site_solvers.py,
    _brute_force): the whole combination list must be dispatched to
    _evaluate_chunk as ONE chunk, so it goes through a single heap exactly
    as the old inline loop did. Without this test, that guarantee is only
    ever checked indirectly (via other tests happening to pass); a future
    change to the chunk-sizing heuristic that accidentally applies to
    n_jobs=1 too would only be caught by luck of fixture size."""
    import lokigi.mixins.site_solvers as site_solvers

    calls = []
    original = site_solvers._evaluate_chunk

    def spy(site_problem, indexed_chunk, *args, **kwargs):
        calls.append(indexed_chunk)
        return original(site_problem, indexed_chunk, *args, **kwargs)

    monkeypatch.setattr(site_solvers, "_evaluate_chunk", spy)

    many_tied_sites_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=1,
        brute_force_keep_best_n=5,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 30  # every combination, in the one chunk


# --- Brute force: every generated combination is evaluated, exactly once ---


@pytest.mark.parametrize("n_jobs", [1, 2], ids=["single_chunk", "multi_chunk"])
def test_brute_force_evaluates_every_combination_exactly_once(
    five_site_problem, n_jobs
):
    """The plain ("keep everything") brute-force path must return exactly
    the set of combinations `_generate_all_combinations` produces -- no
    skips, no duplicates, no substitutions. A count-only check (`len(...)
    == 10`) wouldn't catch a bug that evaluates one combination twice while
    dropping another, so this compares the actual sets of site indices.

    n_jobs=2 is the meaningful case: with 10 combinations and n_jobs=1
    everything runs as a single chunk (mixins/site_solvers.py:252-253), so
    the chunk-splitting/merging path (`n_chunks = min(n_combinations,
    n_workers*4)`, merged back via `outputs.extend(result)` per chunk at
    line 297) never actually runs. At n_jobs=2, `five_site_problem`'s 10
    combinations split into 5 chunks of 2 each (n_chunks=8 is only a
    target used to derive chunk_size=ceil(10/8)=2; ceil-rounding that size
    up means fewer, larger chunks actually cover the 10 combinations),
    exercising that merge."""
    result = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        n_jobs=n_jobs,
    )

    expected = {frozenset(c) for c in itertools.combinations(range(5), 2)}
    actual = {frozenset(row["site_indices"]) for _, row in result.solution_df.iterrows()}

    assert actual == expected


# --- Brute force: BRUTE_FORCE_WARN_THRESHOLD / BRUTE_FORCE_LIMIT reaction ---


def test_brute_force_warns_when_warn_threshold_is_crossed(
    five_site_problem, monkeypatch
):
    """Crossing BRUTE_FORCE_WARN_THRESHOLD (but not BRUTE_FORCE_LIMIT) must
    emit a UserWarning mentioning the combination count, and still return
    every combination -- it's advisory, not a cap."""
    import lokigi.mixins.site_solvers as site_solvers

    monkeypatch.setattr(site_solvers, "BRUTE_FORCE_WARN_THRESHOLD", 3)
    monkeypatch.setattr(site_solvers, "BRUTE_FORCE_LIMIT", 50)

    with pytest.warns(UserWarning, match="10"):
        result = five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )

    assert len(result.solution_df) == 10


def test_brute_force_raises_when_limit_is_crossed(five_site_problem, monkeypatch):
    """Crossing BRUTE_FORCE_LIMIT without `brute_force_ignore_limit` must
    raise MemoryError rather than silently evaluating everything."""
    import lokigi.mixins.site_solvers as site_solvers

    monkeypatch.setattr(site_solvers, "BRUTE_FORCE_LIMIT", 5)

    with pytest.raises(MemoryError, match="10"):
        five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )


def test_brute_force_ignore_limit_suppresses_the_error_but_still_warns(
    five_site_problem, monkeypatch
):
    """`brute_force_ignore_limit=True` must suppress the MemoryError, emit
    a different ("opted to ignore") warning instead, and still evaluate
    every combination rather than silently truncating."""
    import lokigi.mixins.site_solvers as site_solvers

    monkeypatch.setattr(site_solvers, "BRUTE_FORCE_LIMIT", 5)

    with pytest.warns(UserWarning, match="ignore the advised limit"):
        result = five_site_problem.solve(
            p=2,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            brute_force_ignore_limit=True,
        )

    assert len(result.solution_df) == 10
