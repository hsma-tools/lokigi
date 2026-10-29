import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import lokigi.site
from lokigi.mixins.site_solution_pareto import ParetoMixin
from lokigi.multiobjective import Metric


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
        Metric(
            column="weighted_average",
            direction="lower_better",
            label="Average travel time",
            unit="minutes",
        ),
        Metric(
            column="max",
            direction="lower_better",
            label="Maximum travel time",
            unit="minutes",
        ),
        Metric(
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
        Metric(
            column="weighted_average",
            direction="lower_better",
            label="whole-population weighted average travel time",
            unit="minutes",
        ),
        Metric(
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


# --- diff_against (plot_pareto_summary / plot_pareto_facets) ---------------
#
# _resolve_site_diff lives on the real SiteSolutionSet (site_solutions.py),
# not on the minimal DummySolutionSet stub used above -- these tests need an
# actual solve() result so diff_against has something to call.


@pytest.fixture
def pareto_solution_set():
    """4-site/3-demand-location problem, brute-forced at p=2 (6
    combinations). Rank 1 = {Site_B, Site_D}, rank 2 = {Site_A, Site_C} --
    disjoint from rank 1, so every site differs between them, and both are
    Pareto-optimal across (weighted_average, max) by construction:
        L1: A=10, B=25, C=30, D=12
        L2: A=20, B=5,  C=10, D=22
        L3: A=30, B=15, C=8,  D=9
    """
    demand_df = pd.DataFrame(
        {"location_id": ["L1", "L2", "L3"], "demand": [100, 100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "lat": [51.1, 51.2, 51.3, 51.4],
            "long": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["L1", "L2", "L3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
            "Site_D": [12.0, 22.0, 9.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.solve(p=2, search_strategy="brute-force", show_progress=False)
    result.compute_pareto_front(
        [
            Metric(
                column="weighted_average",
                direction="lower_better",
                label="Average travel time",
                unit="minutes",
            ),
            Metric(
                column="max",
                direction="lower_better",
                label="Max travel time",
                unit="minutes",
            ),
        ]
    )
    return result


def test_plot_pareto_summary_diff_against_none_is_unchanged(pareto_solution_set):
    fig = pareto_solution_set.plot_pareto_summary()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Option 1", "Option 2"]


def test_plot_pareto_summary_diff_against_rank_1_labels_the_reference_empty(
    pareto_solution_set,
):
    fig = pareto_solution_set.plot_pareto_summary(diff_against="rank_1")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0] == "Option 1"


def test_plot_pareto_summary_diff_against_rank_1_shows_added_and_removed(
    pareto_solution_set,
):
    fig = pareto_solution_set.plot_pareto_summary(diff_against="rank_1")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    other = labels[1]
    assert "+Site_A, Site_C" in other
    assert "-Site_B, Site_D" in other
    # The two clauses must be semicolon-separated, not comma-separated,
    # or "+Site_A, Site_C, -Site_B, Site_D" misreads as four separately
    # signed items instead of two added and two removed.
    assert "; -" in other


def test_plot_pareto_summary_invalid_diff_against_raises(pareto_solution_set):
    with pytest.raises(ValueError, match="diff_against must be one of"):
        pareto_solution_set.plot_pareto_summary(diff_against="sideways")


def test_plot_pareto_facets_diff_against_none_omits_diff_line(pareto_solution_set):
    fig = pareto_solution_set.plot_pareto_facets()
    for ax in fig.axes:
        assert "Diff (vs" not in ax.get_title(loc="left")


def test_plot_pareto_facets_diff_against_rank_1_adds_diff_line(pareto_solution_set):
    fig = pareto_solution_set.plot_pareto_facets(diff_against="rank_1")
    titles = [ax.get_title(loc="left") for ax in fig.axes]

    # Rank 1's own facet has nothing to diff against itself.
    assert not any("Diff (vs" in t for t in titles if "Option\\ 1" in t)
    # Rank 2's facet shows the diff against rank 1.
    rank_2_title = next(t for t in titles if "Option\\ 2" in t)
    assert "Diff (vs rank 1): +Site_A, Site_C; -Site_B, Site_D" in rank_2_title


def test_plot_pareto_facets_invalid_diff_against_raises(pareto_solution_set):
    with pytest.raises(ValueError, match="diff_against must be one of"):
        pareto_solution_set.plot_pareto_facets(diff_against="sideways")


# --- plot_pareto_radar / plot_pareto_radar_facets ---------------------------


@pytest.fixture
def radar_solution_set(pareto_solution_set):
    """pareto_solution_set with a third metric, since a radar chart needs at
    least 3 spokes to draw a shape."""
    pareto_solution_set.compute_pareto_front(
        pareto_solution_set.pareto_metrics
        + [
            Metric(
                column="unweighted_average",
                direction="lower_better",
                label="Unweighted average travel time",
                unit="minutes",
            )
        ]
    )
    return pareto_solution_set


def _front_size(solution_set):
    return int(solution_set.solution_df["is_pareto_optimal"].sum())


def test_plot_pareto_radar_returns_polar_figure(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar()
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].name == "polar"
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["Option 1", "Option 2", "Option 4"]
        # One filled polygon per Pareto-optimal option
        assert len(fig.axes[0].patches) == _front_size(solved_solution_set)
    finally:
        plt.close(fig)


def test_plot_pareto_radar_fill_false_draws_no_polygons(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(fill=False)
    try:
        assert len(fig.axes[0].patches) == 0
    finally:
        plt.close(fig)


def test_plot_pareto_radar_show_dominated_adds_background_lines(solved_solution_set):
    fig_without = solved_solution_set.plot_pareto_radar()
    fig_with = solved_solution_set.plot_pareto_radar(show_dominated=True)
    try:
        n_dominated = len(solved_solution_set.solution_df) - _front_size(
            solved_solution_set
        )
        assert (
            len(fig_with.axes[0].lines)
            == len(fig_without.axes[0].lines) + n_dominated
        )
    finally:
        plt.close(fig_without)
        plt.close(fig_with)


def test_plot_pareto_radar_polygons_are_closed(solved_solution_set):
    """Each shape must end where it started, or the last spoke's segment
    back to the first is silently missing and the outline has a gap."""
    fig = solved_solution_set.plot_pareto_radar()
    try:
        n_metrics = len(solved_solution_set.pareto_metrics)
        for line in fig.axes[0].lines:
            theta, r = line.get_data()
            assert len(theta) == n_metrics + 1
            assert theta[0] == theta[-1]
            assert r[0] == r[-1]
    finally:
        plt.close(fig)


def test_plot_pareto_radar_further_out_is_better(solved_solution_set):
    """Solution 1 has the lowest (best) weighted_average, a lower_better
    metric, so it must sit on the outer ring of that spoke -- and solution
    2, the worst Pareto-optimal option on it, nearer the centre."""
    fig = solved_solution_set.plot_pareto_radar()
    try:
        ax = fig.axes[0]
        lines = {line.get_label(): line for line in ax.lines}
        spoke = 0  # weighted_average is the first metric
        assert lines["Option 1"].get_ydata()[spoke] == pytest.approx(1.0)
        assert lines["Option 2"].get_ydata()[spoke] < 1.0
    finally:
        plt.close(fig)


def test_plot_pareto_radar_draws_onto_supplied_polar_ax(solved_solution_set):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        returned = solved_solution_set.plot_pareto_radar(ax=ax)
        assert returned is fig
        assert len(ax.lines) == _front_size(solved_solution_set)
    finally:
        plt.close(fig)


def test_plot_pareto_radar_rejects_cartesian_ax(solved_solution_set):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="needs polar axes"):
            solved_solution_set.plot_pareto_radar(ax=ax)
    finally:
        plt.close(fig)


def _trace(fig, name):
    return next(trace for trace in fig.data if trace.name == name)


def _legend_order(fig):
    """Option names as the legend lists them -- trace order is draw order,
    which is by shape size, not option order."""
    front = [trace for trace in fig.data if trace.showlegend is not False]
    return [trace.name for trace in sorted(front, key=lambda t: t.legendrank)]


def test_plot_pareto_radar_interactive_returns_plotly_figure(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    assert type(fig).__module__.startswith("plotly")
    assert fig.to_dict()
    assert _legend_order(fig) == ["Option 1", "Option 2", "Option 4"]
    # Hover carries the raw value, not just the rescaled radius
    option_1 = _trace(fig, "Option 1")
    assert option_1.customdata[0][0] == "Average travel time"
    assert option_1.customdata[0][1] == "10"
    # Closed like the static version
    assert option_1.r[0] == option_1.r[-1]
    assert option_1.theta[0] == option_1.theta[-1]


def test_plot_pareto_radar_interactive_hover_lists_sites(solved_solution_set):
    """Every vertex of an option carries its sites, and so does the shape's
    own fill hover, so hovering inside the shape shows them too."""
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    option_1 = _trace(fig, "Option 1")
    sites = "<br><br><b>Sites</b><br>\u2022 Site A<br>\u2022 Site B"
    assert all(point[3] == sites for point in option_1.customdata)
    assert "%{customdata[3]}" in option_1.hovertemplate
    assert option_1.hoveron == "points+fills"
    assert option_1.text == f"<b>Option 1</b>{sites}"


def test_plot_pareto_radar_interactive_no_fill_no_fill_hover(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(interactive=True, fill=False)
    assert all(trace.hoveron == "points" for trace in fig.data)


def test_plot_pareto_radar_interactive_draws_smallest_shapes_last(
    solved_solution_set,
):
    """Plotly gives a hover inside overlapping fills to the last-drawn one,
    so drawing largest-first makes it name the smallest shape containing
    the cursor rather than whichever option comes last."""
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    lower = fig.layout.polar.radialaxis.range[0]

    def area(trace):
        radii = np.array(trace.r[:-1]) - lower
        return float(np.sum(radii * np.roll(radii, -1)))

    areas = [area(trace) for trace in fig.data]
    assert len(set(areas)) == len(areas)  # otherwise the order is untested
    assert areas == sorted(areas, reverse=True)


def test_plot_pareto_radar_interactive_hover_zone_has_no_dead_edges(
    solved_solution_set,
):
    """polar.hole makes plotly's hover zone a ring with a seam along 0
    degrees, killing hover on east-pointing spokes; and a range ending at
    exactly 0/1 leaves best/worst vertices half outside it. The empty
    centre comes from the range instead, padded at the top."""
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    polar = fig.layout.polar
    assert not polar.hole
    low, high = polar.radialaxis.range
    assert low < 0 and high > 1


def test_plot_pareto_radar_interactive_hover_names_tied_options(
    solution_df, pareto_metrics
):
    """Options tied on a metric share one vertex, and plotly only names one
    of them on hover -- so each vertex lists the others."""
    # Option 4 now ties option 1 on coverage, and is still Pareto-optimal
    # (better than option 1 on max travel time)
    is_option_4 = solution_df["solution_rank"] == 4
    solution_df.loc[is_option_4, "proportion_within_coverage_threshold"] = 0.90
    solution_set = DummySolutionSet(solution_df)
    solution_set.compute_pareto_front(pareto_metrics)
    fig = solution_set.plot_pareto_radar(interactive=True)
    coverage_spoke = 2
    assert _trace(fig, "Option 1").customdata[coverage_spoke][2].endswith(
        "<br>Same value: Option 4"
    )
    assert _trace(fig, "Option 4").customdata[coverage_spoke][2].endswith(
        "<br>Same value: Option 1"
    )
    assert "Same value" not in _trace(fig, "Option 2").customdata[coverage_spoke][2]
    # Untied spokes say nothing
    assert "Same value" not in _trace(fig, "Option 1").customdata[0][2]


def _visible(fig):
    """Option name -> whether it starts visible (not "legendonly")."""
    return {
        trace.name: trace.visible != "legendonly"
        for trace in fig.data
        if trace.showlegend is not False
    }


def test_plot_pareto_radar_initially_shown_defaults_to_all(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    assert _visible(fig) == {"Option 1": True, "Option 2": True, "Option 4": True}
    assert "greyed-out" not in fig.layout.annotations[0].text


def test_plot_pareto_radar_initially_shown_first(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(
        interactive=True, initially_shown="first", show_dominated=True
    )
    assert _visible(fig) == {"Option 1": True, "Option 2": False, "Option 4": False}
    # Hidden options keep their legend entry, so they can be clicked back on
    assert all(
        trace.showlegend is not False
        for trace in fig.data
        if trace.visible == "legendonly"
    )
    # Dominated background outlines have no legend entry to restore them
    # from, so they are unaffected
    assert [t.visible for t in fig.data if t.showlegend is False] == [None]
    caption = fig.layout.annotations[0].text.replace("<br>", " ")
    assert "click a greyed-out option in the legend" in caption


@pytest.mark.parametrize(
    "ids, expected",
    [
        ([2], {"Option 1": False, "Option 2": True, "Option 4": False}),
        ((1, 4), {"Option 1": True, "Option 2": False, "Option 4": True}),
        ({4}, {"Option 1": False, "Option 2": False, "Option 4": True}),
    ],
)
def test_plot_pareto_radar_initially_shown_list_of_ids(
    solved_solution_set, ids, expected
):
    fig = solved_solution_set.plot_pareto_radar(interactive=True, initially_shown=ids)
    assert _visible(fig) == expected


def test_plot_pareto_radar_initially_shown_ids_ignore_name_col(solved_solution_set):
    """Ids are always id_col values, even when the legend shows names."""
    fig = solved_solution_set.plot_pareto_radar(
        interactive=True, name_col="scenario_name", initially_shown=[2]
    )
    assert _visible(fig) == {
        "Solution Alpha": False,
        "Solution Beta": True,
        "Solution Delta": False,
    }


def test_plot_pareto_radar_initially_shown_rejects_dominated_id(solved_solution_set):
    # Solution 3 is dominated
    with pytest.raises(ValueError, match=r"3 exist but is not Pareto-optimal.*1, 2, 4"):
        solved_solution_set.plot_pareto_radar(interactive=True, initially_shown=[1, 3])


def test_plot_pareto_radar_initially_shown_rejects_unknown_id(solved_solution_set):
    with pytest.raises(ValueError, match="aren't Pareto-optimal options: 99"):
        solved_solution_set.plot_pareto_radar(interactive=True, initially_shown=[99])


@pytest.mark.parametrize("bad", ["sideways", [], 5])
def test_plot_pareto_radar_initially_shown_rejects_bad_values(solved_solution_set, bad):
    with pytest.raises(ValueError, match="initially_shown"):
        solved_solution_set.plot_pareto_radar(interactive=True, initially_shown=bad)


def test_plot_pareto_radar_initially_shown_ignored_when_static(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(initially_shown="first")
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].lines) == _front_size(solved_solution_set)
    finally:
        plt.close(fig)
    # An invalid value is still a mistake, so still raises
    with pytest.raises(ValueError, match="initially_shown"):
        solved_solution_set.plot_pareto_radar(initially_shown="sideways")


def test_plot_pareto_radar_interactive_without_site_names(solution_df, pareto_metrics):
    solution_set = DummySolutionSet(solution_df.drop(columns="site_names"))
    solution_set.compute_pareto_front(pareto_metrics)
    fig = solution_set.plot_pareto_radar(interactive=True)
    option_1 = _trace(fig, "Option 1")
    assert all(point[3] == "" for point in option_1.customdata)
    assert option_1.text == "<b>Option 1</b>"


def test_plot_pareto_radar_interactive_show_dominated_skips_hover(
    solved_solution_set,
):
    fig = solved_solution_set.plot_pareto_radar(interactive=True, show_dominated=True)
    dominated = [trace for trace in fig.data if trace.showlegend is False]
    assert len(dominated) == 1
    assert dominated[0].hoverinfo == "skip"
    assert len(fig.data) == _front_size(solved_solution_set) + 1


def test_plot_pareto_radar_best_worst_scoped_to_all_options(solved_solution_set):
    """The spoke ranges come from every solution, dominated ones included,
    so the labels must say so -- not read as best/worst of the shapes
    shown. solution 3 is dominated, so here the two differ."""
    fig = solved_solution_set.plot_pareto_radar()
    try:
        spoke_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert all("of all 4 options" in label for label in spoke_labels)
        # Worst max travel time (25) only occurs in the dominated solution
        assert "worst 25" in spoke_labels[1]
    finally:
        plt.close(fig)

    interactive_fig = solved_solution_set.plot_pareto_radar(interactive=True)
    option_1 = _trace(interactive_fig, "Option 1")
    assert "of all 4 options" in option_1.theta[0]
    assert option_1.customdata[0][2] == "1st of all 4 options"
    assert "across all 4 options" in interactive_fig.layout.annotations[0].text


def test_plot_pareto_radar_interactive_caption_points_to_hover(solved_solution_set):
    """The interactive version draws no per-point target labels, so its
    caption must point to the hover rather than to labels that aren't there."""
    target_metric = Metric(
        column="max", direction="closest_to_target", target=18.0, label="Max"
    )
    solved_solution_set.compute_pareto_front(
        solved_solution_set.pareto_metrics[:1]
        + [target_metric]
        + solved_solution_set.pareto_metrics[2:]
    )
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    caption = fig.layout.annotations[0].text
    assert "hover over a point" in caption
    assert "small label" not in caption


def test_plot_pareto_radar_interactive_sets_explicit_height(solved_solution_set):
    """Plotly's 450px default leaves only a sliver for the chart once the
    title and caption margins come out of it."""
    fig = solved_solution_set.plot_pareto_radar(interactive=True)
    margin = fig.layout.margin
    assert fig.layout.height - margin.t - margin.b >= 500


def test_plot_pareto_radar_caption_can_be_suppressed(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar(caption="")
    try:
        assert not fig.texts
    finally:
        plt.close(fig)
    interactive_fig = solved_solution_set.plot_pareto_radar(
        interactive=True, caption=""
    )
    assert not interactive_fig.layout.annotations


@pytest.mark.parametrize("method", ["plot_pareto_radar", "plot_pareto_radar_facets"])
def test_plot_pareto_radar_needs_three_metrics(long_label_solution_set, method):
    with pytest.raises(ValueError, match="at least 3 Pareto metrics.*plot_pareto_"):
        getattr(long_label_solution_set, method)()


@pytest.mark.parametrize("method", ["plot_pareto_radar", "plot_pareto_radar_facets"])
def test_plot_pareto_radar_raises_before_compute(solution_set, method):
    with pytest.raises(ValueError, match="Pareto front has not been calculated"):
        getattr(solution_set, method)()


def test_plot_pareto_radar_diff_against_rank_1_labels(radar_solution_set):
    fig = radar_solution_set.plot_pareto_radar(diff_against="rank_1")
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels[0] == "Option 1"
        assert "+Site_A, Site_C; -Site_B, Site_D" in labels[1]
    finally:
        plt.close(fig)


def test_plot_pareto_radar_facets_one_polar_axes_per_option(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar_facets()
    try:
        # 3 options at the default ncols=3: no spare cells left over
        assert len(fig.axes) == _front_size(solved_solution_set)
        assert all(ax.name == "polar" for ax in fig.axes)
        for ax in fig.axes:
            title = ax.get_title(loc="left")
            assert "Strengths:" in title
            assert "Sacrifices:" in title
    finally:
        plt.close(fig)


def test_plot_pareto_radar_facets_deletes_unused_cells(solved_solution_set):
    fig = solved_solution_set.plot_pareto_radar_facets(ncols=2)
    try:
        assert len(fig.axes) == _front_size(solved_solution_set)
    finally:
        plt.close(fig)


def test_plot_pareto_radar_facets_diff_against_rank_1_adds_diff_line(
    radar_solution_set,
):
    fig = radar_solution_set.plot_pareto_radar_facets(diff_against="rank_1")
    try:
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        assert not any("Diff (vs" in t for t in titles if "Option\\ 1" in t)
        rank_2_title = next(t for t in titles if "Option\\ 2" in t)
        assert "Diff (vs rank 1): +Site_A, Site_C; -Site_B, Site_D" in rank_2_title
    finally:
        plt.close(fig)


def test_plot_pareto_radar_facets_invalid_rank_scope_raises(solved_solution_set):
    with pytest.raises(ValueError, match="rank_scope must be one of"):
        solved_solution_set.plot_pareto_radar_facets(rank_scope="sideways")


def test_plot_pareto_radar_facets_wraps_strengths_to_wrap_at(solved_solution_set):
    """Radar facets are narrow, so unlike plot_pareto_facets() the
    Strengths/Sacrifices lines are wrapped too -- otherwise long metric
    labels run on into the neighbouring subplot."""
    fig = solved_solution_set.plot_pareto_radar_facets(wrap_at=30)
    try:
        for ax in fig.axes:
            meta = ax.get_title(loc="left").split("---\n")[1].split("\n")
            assert len(meta) > 2  # actually wrapped, not just short
            assert all(len(line) <= 30 for line in meta)
    finally:
        plt.close(fig)
