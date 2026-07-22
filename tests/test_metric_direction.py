"""
Regression tests for sort-direction inference on coverage metrics
(`lokigi.utils._is_maximise_metric` and its call sites).

Coverage proportions are the only solution metrics where HIGHER is better --
everything else reported is a travel cost. Three call sites have to know
that, and each of them used to decide it with an exact string comparison
against "proportion_within_coverage_threshold":

    ascending = False if objective == "proportion_within_coverage_threshold"

That comparison silently failed for any coverage column that wasn't spelled
exactly that way, so both

  - `proportion_within_coverage_threshold__<label>` (a registered secondary
    travel matrix's coverage, present since v0.6.0), and
  - `proportion_regions_within_coverage_threshold` (added in v0.7.0)

were treated as metrics to MINIMISE -- ranking and Pareto fronts came out
backwards, picking the worst-covering solutions as the best.

The unit tests in test_utils_solving_helpers.py cover `_is_maximise_metric`
itself. These tests cover the call sites, so that reverting any one of them
to an `==` check fails here rather than passing silently.
"""

import pandas as pd
import pytest
import sweetpareto.vis as spv

from lokigi.site_solutions import SiteSolutionSet, SolutionComparator

COVERAGE_COLUMNS = [
    "proportion_within_coverage_threshold",
    "proportion_regions_within_coverage_threshold",
    "proportion_within_coverage_threshold__public_transport",
    "proportion_regions_within_coverage_threshold__public_transport",
]


def _make_set(site_indices, site_names, coverage, coverage_col, max_cost):
    """A minimal SiteSolutionSet carrying one coverage column under a chosen
    name, so the same scenario can be replayed for each spelling."""
    df = pd.DataFrame(
        {
            "site_indices": site_indices,
            "site_names": site_names,
            coverage_col: coverage,
            "max": max_cost,
        }
    )
    return SiteSolutionSet(
        solution_df=df, site_problem=None, objectives="mclp", n_sites=2
    )


# --- SolutionComparator ranking -------------------------------------------


@pytest.mark.parametrize("coverage_col", COVERAGE_COLUMNS)
def test_comparator_ranks_coverage_highest_first(coverage_col):
    """`find_balanced_solution` ranks each set by `objective` before taking
    the top `top_n`, so with top_n=1 the sort direction alone decides which
    pair is returned.

    Both sets contain the well-covering ['A', 'B'] (0.9 / 0.8) and a
    poorly-covering alternative (0.1 / 0.2). Ranked correctly, ['A', 'B']
    tops both sets and is returned as a perfect Jaccard match. Ranked
    ascending -- the old behaviour for every spelling but the first -- the
    two worst solutions are compared instead, and they share no sites.
    """
    set_a = _make_set(
        site_indices=[[0, 1], [0, 2]],
        site_names=[["A", "B"], ["A", "C"]],
        coverage=[0.9, 0.1],
        coverage_col=coverage_col,
        max_cost=[20.0, 22.0],
    )
    set_b = _make_set(
        site_indices=[[0, 1], [7, 8]],
        site_names=[["A", "B"], ["X", "Y"]],
        coverage=[0.8, 0.2],
        coverage_col=coverage_col,
        max_cost=[21.0, 23.0],
    )
    comparator = SolutionComparator(set_a, set_b)

    solution_a, solution_b = comparator.find_balanced_solution(
        top_n=1,
        method="jaccard",
        objective=coverage_col,
        secondary_objective="max",
    )

    assert solution_a["site_names"] == ["A", "B"]
    assert solution_b["site_names"] == ["A", "B"]


def test_comparator_still_ranks_cost_metrics_lowest_first():
    """The mirror of the test above: `_is_maximise_metric` must not
    over-match and flip travel-cost columns, which are still minimised."""
    set_a = _make_set(
        site_indices=[[0, 1], [0, 2]],
        site_names=[["A", "B"], ["A", "C"]],
        coverage=[0.9, 0.1],
        coverage_col="proportion_within_coverage_threshold",
        max_cost=[30.0, 10.0],
    )
    set_b = _make_set(
        site_indices=[[0, 2], [7, 8]],
        site_names=[["A", "C"], ["X", "Y"]],
        coverage=[0.8, 0.2],
        coverage_col="proportion_within_coverage_threshold",
        max_cost=[12.0, 40.0],
    )
    comparator = SolutionComparator(set_a, set_b)

    solution_a, solution_b = comparator.find_balanced_solution(
        top_n=1,
        method="jaccard",
        objective="max",
        secondary_objective="proportion_within_coverage_threshold",
    )

    # Lowest `max` tops each set: ['A', 'C'] at 10.0 and ['A', 'C'] at 12.0.
    assert solution_a["site_names"] == ["A", "C"]
    assert solution_b["site_names"] == ["A", "C"]


# --- Pareto plotting ------------------------------------------------------


@pytest.fixture
def pareto_spy(monkeypatch):
    """Capture the maxx/maxy flags the pareto plotting methods infer, without
    rendering anything. Intercepts the external sweetpareto entry point only
    -- the direction inference under test is all on lokigi's side of it."""
    calls = []

    class _FakePlot:
        def on(self, ax):
            return self

        def plot(self):
            return self

    def _fake_pareto_plot(df, x, y, maxx, maxy, **kwargs):
        calls.append({"x": x, "y": y, "maxx": maxx, "maxy": maxy})
        return _FakePlot()

    monkeypatch.setattr(spv, "pareto_plot", _fake_pareto_plot)
    return calls


def _solution_set_with_coverage_columns():
    df = pd.DataFrame(
        {
            "solution_rank": [1, 2],
            "site_indices": [[0, 1], [0, 2]],
            "site_names": [["A", "B"], ["A", "C"]],
            "coverage_threshold": [20, 20],
            "weighted_average": [10.0, 12.0],
            "unweighted_average": [11.0, 13.0],
            "90th_percentile": [18.0, 19.0],
            "max": [20.0, 22.0],
            **{col: [0.9, 0.7] for col in COVERAGE_COLUMNS},
        }
    )
    return SiteSolutionSet(
        solution_df=df, site_problem=None, objectives="mclp", n_sites=2
    )


@pytest.mark.parametrize("coverage_col", COVERAGE_COLUMNS)
def test_simple_pareto_front_infers_maximise_for_coverage(pareto_spy, coverage_col):
    solutions = _solution_set_with_coverage_columns()

    solutions.plot_simple_pareto_front_pairs(x_axis=coverage_col, y_axis="max")

    assert pareto_spy[-1]["maxx"] is True
    assert pareto_spy[-1]["maxy"] is False


@pytest.mark.parametrize("coverage_col", COVERAGE_COLUMNS)
def test_simple_pareto_front_infers_maximise_on_the_y_axis_too(
    pareto_spy, coverage_col
):
    solutions = _solution_set_with_coverage_columns()

    solutions.plot_simple_pareto_front_pairs(x_axis="weighted_average", y_axis=coverage_col)

    assert pareto_spy[-1]["maxx"] is False
    assert pareto_spy[-1]["maxy"] is True


def test_explicit_maxx_maxy_still_override_the_inference(pareto_spy):
    """Direction is only inferred when the caller leaves it as None."""
    solutions = _solution_set_with_coverage_columns()

    solutions.plot_simple_pareto_front_pairs(
        x_axis="proportion_within_coverage_threshold",
        y_axis="max",
        maxx=False,
        maxy=True,
    )

    assert pareto_spy[-1]["maxx"] is False
    assert pareto_spy[-1]["maxy"] is True


def test_all_metric_pareto_front_pairs_infers_maximise_for_coverage(pareto_spy):
    """`plot_all_metric_pareto_front_pairs` builds its own metric pairs and
    infers a direction per pair, so every pair involving the coverage metric
    must flag it as a maximisation objective."""
    solutions = _solution_set_with_coverage_columns()

    solutions.plot_all_metric_pareto_front_pairs()

    assert pareto_spy, "no pareto pairs were plotted"
    saw_coverage = False
    for call in pareto_spy:
        assert call["maxx"] is (
            call["x"] == "proportion_within_coverage_threshold"
        ), f"wrong maxx for x={call['x']!r}"
        assert call["maxy"] is (
            call["y"] == "proportion_within_coverage_threshold"
        ), f"wrong maxy for y={call['y']!r}"
        saw_coverage |= any(
            "within_coverage_threshold" in axis for axis in (call["x"], call["y"])
        )
    assert saw_coverage, "coverage metric was never paired"
