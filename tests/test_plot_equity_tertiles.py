"""Targeted tests for `plot_equity_tertiles()`, which visually decomposes
`inter_tertile_ratio` into the two bar groups (most-/least-disadvantaged
thirds) it actually divides. Complements the generic run/render smoke
coverage in `test_plotting_smoke.py` with the invariants that make this
chart trustworthy rather than merely renderable: the bars and dashed
tertile-average lines must agree exactly with the metric they claim to
decompose (see the "silent divergence" risk between this method and
`check_solution_equity()` called out in its docstring).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import lokigi


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _build_band_solution(costs, demands=None, disadvantaged_end="low"):
    """A single-site problem with one demand location per equity band, so
    `weighted_by_equity_group`/`unweighted_by_equity_group` are both
    trivially hand-computable from `costs` (each band has exactly one
    location, so its "mean" -- weighted or not -- is just that location's
    own travel cost)."""
    n = len(costs)
    demand_df = pd.DataFrame(
        {
            "location_id": [f"LSOA_{i}" for i in range(1, n + 1)],
            "demand": demands or [100] * n,
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": [f"LSOA_{i}" for i in range(1, n + 1)],
            "Site_A": costs,
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": [f"LSOA_{i}" for i in range(1, n + 1)],
            "band": list(range(1, n + 1)),
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="minutes")
    problem.add_equity_data(
        equity_df,
        equity_col="band",
        common_col="location_id",
        label="Band",
        disadvantaged_end=disadvantaged_end,
    )
    return problem.solve(p=1, search_strategy="brute-force", show_progress=False)


# Deliberately not evenly divisible by 3, and decile 4's cost (30.0) is far
# out of line with its neighbours so it visibly matters which chunk it ends
# up in (see the tertile-split regression test in test_equity_metrics.py).
DECILE_COSTS = [10.0, 10.0, 10.0, 30.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]


def test_bars_match_official_avg_and_ratio_weighted_default():
    """The invariant that makes the chart honest: the drawn bars for the
    most-/least-disadvantaged thirds must average to exactly
    avg_lower_third_bins/avg_upper_third_bins, and their quotient must
    equal inter_tertile_ratio -- otherwise the chart could show one thing
    while the metric it claims to visualise says another."""
    solution = _build_band_solution(DECILE_COSTS)
    row = solution.solution_df.iloc[0]

    fig, ax = plt.subplots()
    returned = solution.plot_equity_tertiles(ax=ax)
    assert returned is ax.get_figure()

    bar_heights = [p.get_height() for p in ax.patches]
    assert bar_heights == pytest.approx(DECILE_COSTS)

    most_avg = np.mean(bar_heights[0:3])
    least_avg = np.mean(bar_heights[7:10])
    assert most_avg == pytest.approx(row["avg_lower_third_bins"])
    assert least_avg == pytest.approx(row["avg_upper_third_bins"])
    assert most_avg / least_avg == pytest.approx(row["inter_tertile_ratio"])


def test_bar_order_matches_order_bins_most_to_least_disadvantaged():
    """Bar order must match `_order_bins_most_to_least_disadvantaged()`
    exactly for both directions -- NOT a full element-wise reversal between
    "low" and "high": per that helper's own documented behaviour, only the
    three tertile CHUNKS swap places, while each chunk's internal band
    order stays ascending (e.g. "low" gives [1,2,3 | 4,5,6,7 | 8,9,10] and
    "high" gives [8,9,10 | 4,5,6,7 | 1,2,3], not [10,9,8,...])."""
    from lokigi.utils import _order_bins_most_to_least_disadvantaged

    bins = sorted(range(1, len(DECILE_COSTS) + 1))
    for disadvantaged_end in ("low", "high"):
        solution = _build_band_solution(DECILE_COSTS, disadvantaged_end=disadvantaged_end)
        fig, ax = plt.subplots()
        solution.plot_equity_tertiles(ax=ax)

        labels = [t.get_text() for t in ax.get_xticklabels()]
        expected = [
            str(b) for b in _order_bins_most_to_least_disadvantaged(bins, disadvantaged_end)
        ]
        assert labels == expected

    low_solution = _build_band_solution(DECILE_COSTS, disadvantaged_end="low")
    high_solution = _build_band_solution(DECILE_COSTS, disadvantaged_end="high")
    fig_low, ax_low = plt.subplots()
    low_solution.plot_equity_tertiles(ax=ax_low)
    fig_high, ax_high = plt.subplots()
    high_solution.plot_equity_tertiles(ax=ax_high)

    low_labels = [t.get_text() for t in ax_low.get_xticklabels()]
    high_labels = [t.get_text() for t in ax_high.get_xticklabels()]
    assert high_labels != list(reversed(low_labels))  # documents the non-obvious asymmetry above


def test_hlines_span_exact_chunk_boundaries():
    """An off-by-one in the hline span would silently misattribute an
    average to the wrong bars (e.g. leaking decile 4 into the "most
    disadvantaged" line) without changing whether a line is drawn at all --
    so this checks the exact x-extent, not just that two lines exist."""
    solution = _build_band_solution(DECILE_COSTS)

    fig, ax = plt.subplots()
    solution.plot_equity_tertiles(ax=ax)

    segments = [seg for c in ax.collections for seg in c.get_segments()]
    x_spans = sorted((round(seg[0][0], 6), round(seg[1][0], 6)) for seg in segments)

    assert x_spans == [(-0.5, 2.5), (6.5, 9.5)]


@pytest.mark.parametrize("n_bands", [5, 8, 11])
def test_smoke_non_decile_band_counts(n_bands):
    """Quintiles (5) and other counts not divisible by 3 shouldn't crash,
    and should always draw exactly one bar per band."""
    costs = [float(i) for i in range(1, n_bands + 1)]
    solution = _build_band_solution(costs)

    fig, ax = plt.subplots()
    solution.plot_equity_tertiles(ax=ax)

    assert len(ax.patches) == n_bands


def test_fewer_than_three_bands_raises():
    solution = _build_band_solution([10.0, 20.0])
    with pytest.raises(ValueError, match="at least 3 distinct equity bands"):
        solution.plot_equity_tertiles()


def test_no_equity_data_raises():
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    solution = problem.solve(p=1, search_strategy="brute-force", show_progress=False)

    with pytest.raises(ValueError, match="No equity data loaded"):
        solution.plot_equity_tertiles()


def test_weighted_true_vs_false_differ_on_non_uniform_demand():
    """The parameter's whole reason for existing: on data where demand is
    not uniform within a band, the demand-weighted and plain means must
    actually differ -- otherwise weighted= would be a no-op."""
    n = 3
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "demand": [990, 10, 100, 100],
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "Site_A": [0.0, 100.0, 20.0, 30.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "band": [1, 1, 2, 3],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="band",
        common_col="location_id",
        label="Band",
        disadvantaged_end="low",
    )
    solution = problem.solve(p=1, search_strategy="brute-force", show_progress=False)

    fig_w, ax_w = plt.subplots()
    solution.plot_equity_tertiles(ax=ax_w, weighted=True)
    fig_u, ax_u = plt.subplots()
    solution.plot_equity_tertiles(ax=ax_u, weighted=False)

    # Band 1's bar (first bar, both weighted and unweighted) must differ:
    # weighted skews toward LSOA_1a (990 demand, cost 0.0), unweighted is
    # the plain mean of 0.0 and 100.0.
    weighted_band_1 = ax_w.patches[0].get_height()
    unweighted_band_1 = ax_u.patches[0].get_height()
    assert weighted_band_1 == pytest.approx(0.0, abs=2.0)
    assert unweighted_band_1 == pytest.approx(50.0)
    assert weighted_band_1 != pytest.approx(unweighted_band_1)


def test_matrix_requires_full_secondary_metrics(loaded_problem_with_equity_and_secondary_matrix):
    solution_without_flag = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=1, search_strategy="brute-force", show_progress=False
    )
    with pytest.raises(ValueError, match="full_secondary_metrics"):
        solution_without_flag.plot_equity_tertiles(matrix="public_transport")

    solution_with_flag = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=1,
        search_strategy="brute-force",
        show_progress=False,
        full_secondary_metrics=True,
    )
    fig, ax = plt.subplots()
    returned = solution_with_flag.plot_equity_tertiles(matrix="public_transport", ax=ax)
    assert returned is ax.get_figure()
    assert len(ax.patches) == 3


def test_ax_embedding_does_not_close_caller_figure():
    """When ax= is supplied, the caller owns the figure's lifecycle --
    plot_equity_tertiles must not call plt.close() on it (that would break
    embedding this as one panel of a larger layout)."""
    solution = _build_band_solution(DECILE_COSTS)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    solution.plot_equity_tertiles(ax=ax1)

    assert plt.fignum_exists(fig.number)
