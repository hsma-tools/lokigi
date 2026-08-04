"""
Regression tests for `sort_by`'s sort direction.

Every `sort_by` call site used to sort with a bare
`solution_df.sort_values(sort_by)` -- always ascending. That is correct for
the travel-cost metrics, which are all "lower is better", but backwards for
the coverage proportions, where higher is better. The result was that

    solutions.return_best_combination_site_names(
        sort_by="proportion_within_coverage_threshold"
    )

returned the WORST-covering combination, from a method whose name promises
the best one. Every `sort_by`-accepting method shared the bug, including the
plotting entry points, which selected the worst solutions to plot.

Direction is now resolved per column by `_sort_solutions_by_metric`, which
`_is_maximise_metric` backs (see tests/test_metric_direction.py for the
direction-inference tests themselves).

The fixture below is deliberately built so that ranking on coverage and
ranking on a travel cost pick *different* solutions, and so that ascending
and descending order of the coverage column pick different solutions too --
otherwise these tests would pass with the bug still present.
"""

import pandas as pd
import pytest

from lokigi.site_solutions import SiteSolutionSet
from lokigi.utils import _select_solution, _sort_solutions_by_metric

COVERAGE = "proportion_within_coverage_threshold"
COVERAGE_REGIONS = "proportion_regions_within_coverage_threshold"
COVERAGE_SECONDARY = "proportion_within_coverage_threshold__public_transport"


@pytest.fixture
def solutions():
    """Three solutions where the best on coverage is the worst on `max`,
    so a wrong direction on either column is unambiguous."""
    df = pd.DataFrame(
        {
            "solution_rank": [1, 2, 3],
            "site_indices": [[0, 1], [0, 2], [1, 2]],
            "site_names": [["BEST_COVER"], ["MIDDLE"], ["WORST_COVER"]],
            COVERAGE: [0.9, 0.5, 0.1],
            COVERAGE_REGIONS: [0.9, 0.5, 0.1],
            COVERAGE_SECONDARY: [0.9, 0.5, 0.1],
            "max": [30.0, 20.0, 10.0],
            "weighted_average": [15.0, 12.0, 9.0],
        }
    )
    return SiteSolutionSet(
        solution_df=df, site_problem=None, objectives="mclp", n_sites=2
    )


# --- the helper itself ----------------------------------------------------


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_sort_helper_puts_highest_coverage_first(solutions, column):
    ordered = _sort_solutions_by_metric(solutions.solution_df, column)
    assert ordered["site_names"].iloc[0] == ["BEST_COVER"]


@pytest.mark.parametrize("column", ["max", "weighted_average"])
def test_sort_helper_puts_lowest_cost_first(solutions, column):
    ordered = _sort_solutions_by_metric(solutions.solution_df, column)
    assert ordered["site_names"].iloc[0] == ["WORST_COVER"]


def test_sort_helper_resolves_each_column_independently(solutions):
    """A coverage metric tie-broken by a travel cost: the coverage column
    must sort descending while the tie-breaker sorts ascending."""
    df = solutions.solution_df.copy()
    df[COVERAGE] = [0.9, 0.9, 0.1]  # force a tie on the primary metric

    ordered = _sort_solutions_by_metric(df, [COVERAGE, "max"])

    # Both tied at 0.9; MIDDLE wins the tie-break on the lower `max` (20 < 30).
    assert ordered["site_names"].iloc[0] == ["MIDDLE"]
    assert ordered["site_names"].iloc[1] == ["BEST_COVER"]
    assert ordered["site_names"].iloc[2] == ["WORST_COVER"]


# --- public accessors -----------------------------------------------------


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_return_best_combination_site_names_ranks_coverage_highest_first(
    solutions, column
):
    assert solutions.return_best_combination_site_names(sort_by=column) == [
        "BEST_COVER"
    ]


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_return_best_combination_site_indices_ranks_coverage_highest_first(
    solutions, column
):
    assert solutions.return_best_combination_site_indices(sort_by=column) == [0, 1]


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_return_best_combination_details_ranks_coverage_highest_first(
    solutions, column
):
    details = solutions.return_best_combination_details(sort_by=column, top_n=2)

    assert details["site_names"].tolist() == [["BEST_COVER"], ["MIDDLE"]]


def test_return_best_combination_details_still_ranks_cost_lowest_first(solutions):
    """The mirror case -- travel costs must keep their existing ascending
    order, so the direction fix can't have flipped everything blindly."""
    details = solutions.return_best_combination_details(sort_by="max", top_n=2)

    assert details["site_names"].tolist() == [["WORST_COVER"], ["MIDDLE"]]


def test_sort_by_none_preserves_solve_order(solutions):
    """Without `sort_by`, the solver's own ranking is used untouched."""
    assert solutions.return_best_combination_site_names() == ["BEST_COVER"]
    assert solutions.return_best_combination_details(top_n=3)[
        "site_names"
    ].tolist() == [["BEST_COVER"], ["MIDDLE"], ["WORST_COVER"]]


# --- _select_solution (shared by the plotting entry points) ---------------


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_select_solution_ranks_coverage_highest_first(solutions, column):
    selected = _select_solution(solutions.solution_df, sort_by=column, solution_rank=1)
    assert selected["site_names"].iloc[0] == ["BEST_COVER"]


@pytest.mark.parametrize("column", [COVERAGE, COVERAGE_REGIONS, COVERAGE_SECONDARY])
def test_select_solution_second_rank_walks_down_from_the_best(solutions, column):
    selected = _select_solution(solutions.solution_df, sort_by=column, solution_rank=2)
    assert selected["site_names"].iloc[0] == ["MIDDLE"]


def test_select_solution_still_ranks_cost_lowest_first(solutions):
    selected = _select_solution(solutions.solution_df, sort_by="max", solution_rank=1)
    assert selected["site_names"].iloc[0] == ["WORST_COVER"]


def test_select_solution_explicit_site_names_ignores_sort_by(solutions):
    """`site_names` has priority over `sort_by`, so direction is irrelevant."""
    selected = _select_solution(
        solutions.solution_df, sort_by=COVERAGE, site_names=["WORST_COVER"]
    )
    assert selected["site_names"].iloc[0] == ["WORST_COVER"]


# --- tie stability --------------------------------------------------------


def test_sort_helper_keeps_tied_solutions_in_their_existing_order():
    """Solutions tied on the ranking metric must keep the order they already
    had, rather than being reshuffled arbitrarily.

    pandas' default single-column sort is quicksort, which is NOT stable, so
    without `kind="mergesort"` the solution reported as "best" among a group
    of equally-good ones could differ between runs, machines or pandas
    versions. Ties are routine: on the sample Brighton problem `max` takes
    only 5 distinct values across 15 candidate combinations.

    Sized deliberately large with many tie groups -- a handful of all-equal
    rows is a degenerate case that quicksort happens to leave untouched, so
    a small fixture would pass with an unstable sort and prove nothing.
    """
    n = 600
    df = pd.DataFrame(
        {
            "site_names": [[f"S{i}"] for i in range(n)],
            # 6 distinct values, 100 solutions tied at each.
            COVERAGE: [round((i % 6) / 10, 1) for i in range(n)],
        }
    )

    ordered = _sort_solutions_by_metric(df, COVERAGE)

    for value, group in ordered.groupby(COVERAGE, sort=False):
        positions = [int(name[0].removeprefix("S")) for name in group["site_names"]]
        assert positions == sorted(positions), (
            f"solutions tied at {value} were reordered: {positions[:10]}..."
        )


# NOT TESTED HERE: that repeated calls on the same input return the same
# order. It reads like the natural companion to the test above, and a version
# of it was written and then dropped, because it cannot fail.
#
# Quicksort is unstable but it is not random: for identical input, on one
# machine, with one pandas/numpy build, it reorders ties the same way every
# time. So such a test passes whether or not `kind="mergesort"` is present,
# and would sit in the suite implying coverage it does not provide.
#
# The genuine exposure is *across* pandas/numpy versions, platforms and
# architectures, where the partitioning can differ -- which no test running
# in a single environment can reproduce. The test above covers what is
# actually checkable: that ties come out in input order, which is the
# property that makes results comparable across environments in the first
# place.


# --- plot_travel_time_distribution ----------------------------------------
#
# The only sort_by call site that sorts on two columns (sort_by plus
# secondary_ranking), and the only one reached through a plotting method
# rather than an accessor, so it is exercised end-to-end on a solved problem.


def _plotted_site_sets(figure):
    """The site combinations a distribution figure actually faceted on,
    read back out of its per-facet annotation titles."""
    return {
        frozenset(
            annotation.text.split("Sites:")[1].split("|")[0].strip().split(", ")
        )
        for annotation in (figure.layout.annotations or [])
        if annotation.text.startswith("Sites:")
    }


@pytest.fixture
def solved_mclp(loaded_problem):
    """Coverage at threshold 12 over `loaded_problem`'s demand of 100/200/150:
    {A, C} = 1.0, {B, C} = 350/450, {A, B} = 300/450. Ranked correctly the
    best two are {A, C} and {B, C}; ranked ascending they would be {A, B}
    and {B, C}, so a two-solution selection tells the orderings apart.
    """
    return loaded_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=12,
    )


def test_travel_time_distribution_plots_the_best_covering_solutions(solved_mclp):
    figure = solved_mclp.plot_travel_time_distribution(
        top_n=2, sort_by="proportion_within_coverage_threshold"
    )

    assert _plotted_site_sets(figure) == {
        frozenset({"Site_A", "Site_C"}),
        frozenset({"Site_B", "Site_C"}),
    }


def test_travel_time_distribution_still_plots_lowest_cost_first(solved_mclp):
    """Mirror case: ranking on a travel cost keeps its ascending order."""
    figure = solved_mclp.plot_travel_time_distribution(top_n=1, sort_by="max")

    assert _plotted_site_sets(figure) == {frozenset({"Site_A", "Site_C"})}


def test_travel_time_distribution_accepts_bottom_n(solved_mclp):
    """`bottom_n` used to raise TypeError -- the result was appended with
    `filtered_dfs.append(temp_bottom=...)`, and list.append() takes no
    keyword arguments, so passing bottom_n at all was an instant crash.
    """
    figure = solved_mclp.plot_travel_time_distribution(
        top_n=1, bottom_n=1, sort_by="proportion_within_coverage_threshold"
    )

    # Best-covering {A, C} plus worst-covering {A, B}; {B, C} sits between
    # them and is excluded.
    assert _plotted_site_sets(figure) == {
        frozenset({"Site_A", "Site_C"}),
        frozenset({"Site_A", "Site_B"}),
    }
