"""
Backtests / regression snapshots.

Unlike the hand-computed "known solution" tests elsewhere in this suite,
these tests don't independently verify correctness -- they pin the exact
output of `solve()` for a fixed set of sample problems (both the small
synthetic fixtures in conftest.py and the CSV/geojson-backed
`brighton_problem`) and known solver configurations, so that any future
change to solver internals which alters results shows up as a failing test
here, even if it doesn't happen to violate one of the narrower correctness
checks.

Expected values are NOT hardcoded here -- they live in
tests/backtest_snapshots.json, keyed by test name, via the `assert_backtest`
fixture (defined in conftest.py). If a change to `solve()`, its objectives,
or search strategies is intentional and the new numbers are believed to be
correct, regenerate the snapshots instead of hand-editing them:

    pytest tests/test_backtests.py --update-backtests

...then review the resulting diff in tests/backtest_snapshots.json before
committing it.

Each fixture/config pair captures:
  - the ordered list of returned site combinations (order encodes ranking,
    so a ranking regression is caught even if individual scores don't move)
  - a handful of key metric columns per row, rounded to 6dp to absorb
    floating point noise across platforms/numpy versions.

Written by Claude Opus 4.8
"""

import pandas as pd

from lokigi.utils import _is_maximise_metric


def _fingerprint(result, cols=("weighted_average", "max", "unweighted_average")):
    """Order-preserving, tuple-based snapshot of a SiteSolutionSet.solution_df:
    one tuple per solution row, in the order solve() ranked them, of
    (sorted site names, *requested metric columns rounded to 6dp)."""
    rows = []
    for _, row in result.solution_df.iterrows():
        entry = [tuple(sorted(row["site_names"]))]
        for col in cols:
            value = row[col]
            entry.append(round(float(value), 6) if pd.notna(value) else None)
        rows.append(tuple(entry))
    return rows


# --- loaded_problem (3 sites / 3 demand points, synthetic) ---


def test_backtest_loaded_problem_p_median_brute_force(loaded_problem, assert_backtest):
    result = loaded_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )
    assert_backtest(_fingerprint(result))


def test_backtest_loaded_problem_p_median_greedy(loaded_problem, assert_backtest):
    result = loaded_problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )
    assert_backtest(_fingerprint(result))


def test_backtest_loaded_problem_p_median_grasp(loaded_problem, assert_backtest):
    result = loaded_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
    )
    assert_backtest(_fingerprint(result))


# --- five_site_problem (5 sites / 4 demand points, adversarial travel times) ---


def test_backtest_five_site_problem_p_median_brute_force(
    five_site_problem, assert_backtest
):
    result = five_site_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )
    assert_backtest(_fingerprint(result))


def test_backtest_five_site_problem_p_center_brute_force(
    five_site_problem, assert_backtest
):
    result = five_site_problem.solve(
        p=2, objectives="p_center", search_strategy="brute-force", show_progress=False
    )
    assert_backtest(_fingerprint(result, cols=("max",)))


def test_backtest_five_site_problem_mclp_brute_force(five_site_problem, assert_backtest):
    result = five_site_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    # Both coverage definitions are pinned: the demand-weighted one that mclp
    # actually ranks on, and the region count, so a future change to either
    # weighting shows up here rather than silently.
    assert_backtest(
        _fingerprint(
            result,
            cols=(
                "proportion_within_coverage_threshold",
                "proportion_regions_within_coverage_threshold",
            ),
        )
    )


def test_backtest_five_site_problem_p_median_greedy(five_site_problem, assert_backtest):
    result = five_site_problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )
    assert_backtest(_fingerprint(result))


def test_backtest_five_site_problem_p_median_grasp(five_site_problem, assert_backtest):
    result = five_site_problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
    )
    assert_backtest(_fingerprint(result))


# --- tied_score_problem: exact-tie ordering through the keep_best_n heap ---


def test_backtest_tied_score_problem_p_median_brute_force_keep_best_n(
    tied_score_problem, assert_backtest
):
    result = tied_score_problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        brute_force_keep_best_n=2,
    )
    assert_backtest(_fingerprint(result))


# --- hybrid_p_median_problem: max_value_cutoff safety net ---


def test_backtest_hybrid_p_median_problem_brute_force(
    hybrid_p_median_problem, assert_backtest
):
    result = hybrid_p_median_problem.solve(
        p=2,
        objectives="hybrid_p_median",
        search_strategy="brute-force",
        show_progress=False,
        max_value_cutoff=10,
    )
    assert_backtest(_fingerprint(result))


# --- loaded_problem_with_cost: weighted p_median with a cost component ---


def test_backtest_loaded_problem_with_cost_weighted_p_median(
    loaded_problem_with_cost, assert_backtest
):
    result = loaded_problem_with_cost.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.5, "cost": 0.5},
    )
    assert_backtest(_fingerprint(result, cols=("weighted_average", "total_cost")))


# --- loaded_problem_with_equity: weighted p_median with a demand+equity blend ---


def test_backtest_loaded_problem_with_equity_weighted_p_median(
    loaded_problem_with_equity, assert_backtest
):
    result = loaded_problem_with_equity.solve(
        p=2,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.5, "equity": 0.5},
    )
    assert_backtest(_fingerprint(result))


# --- brighton_problem: real CSV demand/travel matrix + geojson candidate sites ---


def test_backtest_brighton_problem_p_median_brute_force_top5(
    brighton_problem, assert_backtest
):
    result = brighton_problem.solve(
        p=3,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        brute_force_keep_best_n=5,
    )
    assert_backtest(_fingerprint(result))


def test_backtest_brighton_problem_p_median_greedy(brighton_problem, assert_backtest):
    result = brighton_problem.solve(
        p=3, objectives="p_median", search_strategy="greedy", show_progress=False
    )
    assert_backtest(_fingerprint(result))


def test_backtest_brighton_problem_p_median_grasp(brighton_problem, assert_backtest):
    result = brighton_problem.solve(
        p=3,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
    )
    assert_backtest(_fingerprint(result))


def test_backtest_brighton_problem_p_center_brute_force(
    brighton_problem, assert_backtest
):
    result = brighton_problem.solve(
        p=2, objectives="p_center", search_strategy="brute-force", show_progress=False
    )
    assert_backtest(_fingerprint(result, cols=("max",)))


def test_backtest_brighton_problem_mclp_brute_force(brighton_problem, assert_backtest):
    result = brighton_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=20,
    )
    # Both coverage definitions are pinned: the demand-weighted one that mclp
    # actually ranks on, and the region count, so a future change to either
    # weighting shows up here rather than silently.
    assert_backtest(
        _fingerprint(
            result,
            cols=(
                "proportion_within_coverage_threshold",
                "proportion_regions_within_coverage_threshold",
            ),
        )
    )


# --- sort-direction contract ----------------------------------------------


def _direction_fingerprint(result):
    """Snapshot of which `solution_df` columns are treated as maximisation
    objectives, as (column, is_maximised) pairs in column order.

    Unlike the metric fingerprints above, this pins a *contract* rather than
    numbers: every column the solver emits, and whether higher is better for
    it. Adding a metric column, renaming one, or changing how direction is
    inferred all surface here -- including the failure mode this guards
    against, where a coverage column not spelled exactly
    "proportion_within_coverage_threshold" silently falls through to being
    minimised, so rankings and Pareto fronts come out backwards (see
    tests/test_metric_direction.py for the call-site level tests).
    """
    return [
        (column, bool(_is_maximise_metric(column)))
        for column in result.solution_df.columns
    ]


def _rank_on_fingerprint(result, rank_on_cols):
    """Snapshot of the order `rank_on` actually puts solutions in, driven
    through the public accessor rather than the sort helper directly.

    For each ranking column: the distinct metric values in ranked order,
    each paired with the site combinations sitting at that value. A
    direction regression reverses the value sequence, so it cannot survive
    this snapshot.

    Solutions are grouped by value, and names within a group are sorted,
    because ties are common here (`max` takes only 5 distinct values across
    the 15 Brighton combinations) and pandas' default
    `sort_values(kind="quicksort")` is not stable -- a flat ordered list of
    site names would be free to reshuffle within a tie between runs or
    platforms and make this test flaky for reasons unrelated to direction.
    """
    entries = []
    for col in rank_on_cols:
        ordered = result.return_best_combination_details(
            rank_on=col, top_n=len(result.solution_df)
        )
        levels = []
        for value, group in ordered.groupby(col, sort=False, dropna=False):
            rounded = round(float(value), 6) if pd.notna(value) else None
            names = sorted(tuple(sorted(names)) for names in group["site_names"])
            levels.append((rounded, names))
        entries.append((col, levels))
    return entries


def test_backtest_rank_on_ordering_brighton_mclp(brighton_problem, assert_backtest):
    """Pins `rank_on` ordering on real data for a coverage metric, its
    regions counterpart, and two travel costs as controls -- the costs must
    stay ascending while the coverage metrics run highest-first.

    threshold=8 rather than the 20 used by the mclp metric backtest above:
    at 20 the coverage columns collapse to 4 distinct values across 15
    solutions, which pins ordering far more weakly.
    """
    result = brighton_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=8,
    )
    assert_backtest(
        _rank_on_fingerprint(
            result,
            rank_on_cols=(
                "proportion_within_coverage_threshold",
                "proportion_regions_within_coverage_threshold",
                "max",
                "weighted_average",
            ),
        )
    )


def test_backtest_rank_on_ordering_with_secondary_matrix(
    loaded_problem_with_equity_and_secondary_matrix, assert_backtest
):
    """The same contract for `__<label>` secondary-matrix columns, which are
    the ones the exact-match direction check used to miss entirely."""
    result = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
    )
    assert_backtest(
        _rank_on_fingerprint(
            result,
            rank_on_cols=(
                "proportion_within_coverage_threshold__public_transport",
                "proportion_regions_within_coverage_threshold__public_transport",
                "max__public_transport",
            ),
        )
    )


def test_backtest_direction_contract_with_secondary_matrix(
    loaded_problem_with_equity_and_secondary_matrix, assert_backtest
):
    """Solved with equity data, a secondary travel matrix and
    full_secondary_metrics=True, so the snapshot covers the widest schema the
    library emits -- primary metrics, `__<label>` secondary metrics, and both
    coverage spellings for each.
    """
    result = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=15,
        full_secondary_metrics=True,
    )
    assert_backtest(_direction_fingerprint(result))
