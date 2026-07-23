import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from lokigi.mixins.site_solution_pareto import ParetoMixin
from lokigi.multiobjective import ParetoMetric


class DummySolutionSet(ParetoMixin):
    """Minimal host object for ParetoMixin -- only needs a solution_df."""

    def __init__(self, solution_df):
        self.solution_df = solution_df


# --- fixtures ---


@pytest.fixture
def solution_df():
    """
    Four candidate solutions across three metrics.

    Solution 3 is dominated by solutions 1, 2 and 4 (worse or equal on every
    metric, strictly worse on at least one). Solutions 1, 2 and 4 are each
    the best on at least one metric, so all three are Pareto-optimal.
    """
    return pd.DataFrame(
        {
            "solution_rank": [1, 2, 3, 4],
            "weighted_average": [10.0, 12.0, 15.0, 11.0],
            "max": [20.0, 15.0, 25.0, 18.0],
            "proportion_within_coverage_threshold": [0.90, 0.85, 0.80, 0.88],
            "site_names": [
                ["Site A", "Site B"],
                ["Site A", "Site C"],
                ["Site B", "Site C"],
                ["Site D", "Site E"],
            ],
            "scenario_name": [
                "Solution Alpha",
                "Solution Beta",
                "Solution Gamma",
                "Solution Delta",
            ],
            "coverage_threshold": [15, 15, 15, 15],
        }
    )


@pytest.fixture
def single_optimum_df():
    """A single-metric solution set where exactly one solution is Pareto-optimal."""
    return pd.DataFrame(
        {
            "solution_rank": [1, 2, 3],
            "weighted_average": [10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def pareto_metrics():
    return [
        ParetoMetric(
            column="weighted_average",
            direction="lower_better",
            label="Average travel time",
            unit="minutes",
        ),
        ParetoMetric(
            column="max",
            direction="lower_better",
            label="Maximum travel time",
            unit="minutes",
        ),
        ParetoMetric(
            column="proportion_within_coverage_threshold",
            direction="higher_better",
            label="Coverage",
            as_percentage=True,
        ),
    ]


@pytest.fixture
def solution_set(solution_df):
    return DummySolutionSet(solution_df)


@pytest.fixture
def solved_solution_set(solution_set, pareto_metrics):
    """A solution set with compute_pareto_front() already run."""
    solution_set.compute_pareto_front(pareto_metrics)
    return solution_set


@pytest.fixture
def long_label_solution_set():
    """Two metrics with long labels and only one Pareto-optimal solution --
    with few metrics the figure is narrow by default, and long labels rotated
    at a shallow angle used to overhang far enough to collapse the axes."""
    df = pd.DataFrame(
        {
            "solution_rank": [1, 2],
            "weighted_average": [20.0, 22.0],
            "weighted_average_2": [21.0, 23.0],
            "site_names": [["Site A"], ["Site B"]],
        }
    )
    metrics = [
        ParetoMetric(
            column="weighted_average",
            direction="lower_better",
            label="whole-population weighted average travel time",
            unit="minutes",
        ),
        ParetoMetric(
            column="weighted_average_2",
            direction="lower_better",
            label="older-population weighted average travel time",
            unit="minutes",
        ),
    ]
    solution_set = DummySolutionSet(df)
    solution_set.compute_pareto_front(metrics)
    return solution_set


@pytest.fixture
def full_metrics_df():
    """Solution set using the exact hard-coded metric names the plotting
    helpers (plot_simple_pareto_front_pairs / plot_all_metric_pareto_front_pairs)
    expect to find as columns."""
    return pd.DataFrame(
        {
            "solution_rank": [1, 2, 3, 4],
            "weighted_average": [10.0, 12.0, 15.0, 11.0],
            "unweighted_average": [9.0, 11.0, 14.0, 10.5],
            "90th_percentile": [18.0, 14.0, 22.0, 16.0],
            "max": [20.0, 15.0, 25.0, 18.0],
            "proportion_within_coverage_threshold": [0.90, 0.85, 0.80, 0.88],
            "coverage_threshold": [15, 15, 15, 15],
        }
    )


# --- compute_pareto_front ---


def test_compute_pareto_front_flags_optimal_and_dominated(solved_solution_set):
    df = solved_solution_set.solution_df.set_index("solution_rank")
    assert df.loc[1, "is_pareto_optimal"]
    assert df.loc[2, "is_pareto_optimal"]
    assert df.loc[4, "is_pareto_optimal"]
    assert not df.loc[3, "is_pareto_optimal"]


def test_compute_pareto_front_records_all_dominators(solved_solution_set):
    df = solved_solution_set.solution_df.set_index("solution_rank")
    assert sorted(df.loc[3, "dominated_by"]) == [1, 2, 4]


def test_compute_pareto_front_optimal_solutions_have_no_dominators(
    solved_solution_set,
):
    df = solved_solution_set.solution_df.set_index("solution_rank")
    assert df.loc[1, "dominated_by"] == []
    assert df.loc[2, "dominated_by"] == []
    assert df.loc[4, "dominated_by"] == []


def test_compute_pareto_front_stores_metrics_for_later_use(
    solution_set, pareto_metrics
):
    solution_set.compute_pareto_front(pareto_metrics)
    assert solution_set.pareto_metrics == pareto_metrics


# --- pareto_summary ---


def test_pareto_summary_raises_before_compute(solution_set):
    with pytest.raises(ValueError) as exc_info:
        solution_set.pareto_summary()
    assert "Pareto front has not been calculated" in str(exc_info.value)


def test_pareto_summary_returns_only_optimal_solutions(solved_solution_set):
    summary = solved_solution_set.pareto_summary()
    assert sorted(summary["solution_rank"]) == [1, 2, 4]


def test_pareto_summary_sorts_by_first_metric_ascending(solved_solution_set):
    """weighted_average is lower_better, and is the first metric passed in,
    so the default sort should rank 1 (10.0) before 4 (11.0) before 2 (12.0)."""
    summary = solved_solution_set.pareto_summary()
    assert list(summary["solution_rank"]) == [1, 4, 2]


def test_pareto_summary_default_columns(solved_solution_set):
    summary = solved_solution_set.pareto_summary()
    assert list(summary.columns) == [
        "solution_rank",
        "weighted_average",
        "max",
        "proportion_within_coverage_threshold",
    ]


def test_pareto_summary_return_full_df_keeps_all_columns(solved_solution_set):
    summary = solved_solution_set.pareto_summary(return_full_df=True)
    assert "is_pareto_optimal" in summary.columns
    assert "dominated_by" in summary.columns
    assert len(summary) == 3


# --- describe_tradeoffs ---


def test_describe_tradeoffs_single_optimum_message(single_optimum_df, pareto_metrics):
    single_metric = [pareto_metrics[0]]
    solution_set = DummySolutionSet(single_optimum_df)
    solution_set.compute_pareto_front(single_metric)

    statements = solution_set.describe_tradeoffs()

    assert statements == [
        "Only one Pareto-optimal solution found -- no trade-offs to describe."
    ]


def test_describe_tradeoffs_compares_every_option_to_the_anchor(solved_solution_set):
    """Anchor defaults to the solution best on the first metric (lowest
    weighted_average -> solution 1); statements should cover the two other
    Pareto-optimal solutions and skip the dominated one entirely."""
    statements = solved_solution_set.describe_tradeoffs()

    assert len(statements) == 2
    assert (
        statements[0]
        == "Solution 2 vs solution 1: costs average travel time by 2.00; "
        "improves maximum travel time by 5.00; costs coverage by 0.05"
    )
    assert (
        statements[1]
        == "Solution 4 vs solution 1: costs average travel time by 1.00; "
        "improves maximum travel time by 2.00; costs coverage by 0.02"
    )


def test_describe_tradeoffs_honours_explicit_anchor(solved_solution_set):
    statements = solved_solution_set.describe_tradeoffs(anchor=2)

    assert all("vs solution 2" in statement for statement in statements)
    assert any(statement.startswith("Solution 1 ") for statement in statements)
    assert any(statement.startswith("Solution 4 ") for statement in statements)


# --- describe_tradeoffs_for_stakeholders ---


def test_describe_tradeoffs_for_stakeholders_single_optimum_message(
    single_optimum_df, pareto_metrics
):
    single_metric = [pareto_metrics[0]]
    solution_set = DummySolutionSet(single_optimum_df)
    solution_set.compute_pareto_front(single_metric)

    statements = solution_set.describe_tradeoffs_for_stakeholders()

    assert len(statements) == 1
    assert "One option clearly stands out" in statements[0]


def test_describe_tradeoffs_for_stakeholders_headline(solved_solution_set):
    statements = solved_solution_set.describe_tradeoffs_for_stakeholders()
    assert "We found 3 genuinely different options" in statements[0]


def test_describe_tradeoffs_for_stakeholders_markdown_blocks(solved_solution_set):
    statements = solved_solution_set.describe_tradeoffs_for_stakeholders()

    assert statements[1] == (
        "**Option 2** (compared with Option 1)\n"
        "- Gains: Maximum travel time improves by 5.0 minutes\n"
        "- Costs: Average travel time is 2.0 minutes worse; "
        "Coverage is 5.0 percentage points worse"
    )
    assert statements[2] == (
        "**Option 4** (compared with Option 1)\n"
        "- Gains: Maximum travel time improves by 2.0 minutes\n"
        "- Costs: Average travel time is 1.0 minutes worse; "
        "Coverage is 2.0 percentage points worse"
    )


def test_describe_tradeoffs_for_stakeholders_plain_text(solved_solution_set):
    statements = solved_solution_set.describe_tradeoffs_for_stakeholders(markdown=False)

    assert statements[1] == (
        "Option 2, compared with Option 1: Maximum travel time improves by 5.0 "
        "minutes, but Average travel time is 2.0 minutes worse and Coverage is "
        "5.0 percentage points worse."
    )


def test_describe_tradeoffs_for_stakeholders_uses_name_col(solved_solution_set):
    statements = solved_solution_set.describe_tradeoffs_for_stakeholders(
        name_col="scenario_name"
    )

    assert "**Solution Beta** (compared with Solution Alpha)" in statements[1]
    assert "**Solution Delta** (compared with Solution Alpha)" in statements[2]


# --- plotting smoke tests ---
#
# These methods are graphics helpers rather than pure logic, so the tests
# here only check that they run without error against realistic input and
# hand back the kind of object callers expect -- not the visual output.


def test_plot_pareto_summary_returns_figure(solved_solution_set):
    fig = solved_solution_set.plot_pareto_summary()
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


def test_plot_pareto_facets_returns_figure(solved_solution_set):
    fig = solved_solution_set.plot_pareto_facets()
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


def test_plot_pareto_summary_axes_not_collapsed_by_long_labels(
    long_label_solution_set,
):
    """Regression test: long metric labels in a narrow (few-metric) figure
    used to overhang far enough that tight_layout() squeezed the axes down
    to near-zero width -- see the module docstring comment on
    metric_label_wrap_width for the mechanism."""
    fig = long_label_solution_set.plot_pareto_summary()
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert bbox.width > 0.2 * fig.get_size_inches()[0]
    finally:
        plt.close(fig)


def test_plot_pareto_facets_axes_not_collapsed_by_long_labels(
    long_label_solution_set,
):
    fig = long_label_solution_set.plot_pareto_facets()
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert bbox.width > 0.5 * fig.get_size_inches()[0]
    finally:
        plt.close(fig)


def test_plot_pareto_facets_raises_if_no_pareto_optimal_solutions(solution_set):
    """compute_pareto_front hasn't been run, so is_pareto_optimal is missing
    entirely -- but if every row were flagged non-optimal, this should raise
    a clear error rather than plotting an empty grid."""
    solution_set.solution_df["is_pareto_optimal"] = False
    solution_set.pareto_metrics = []

    with pytest.raises(ValueError, match="No Pareto-optimal solutions found"):
        solution_set.plot_pareto_facets()


def test_plot_simple_pareto_front_pairs_runs(full_metrics_df):
    solution_set = DummySolutionSet(full_metrics_df)
    result = solution_set.plot_simple_pareto_front_pairs(
        x_axis="weighted_average", y_axis="max"
    )
    assert result is not None


def test_plot_all_metric_pareto_front_pairs_returns_figure(full_metrics_df):
    solution_set = DummySolutionSet(full_metrics_df)
    fig = solution_set.plot_all_metric_pareto_front_pairs(cols=2)
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)
