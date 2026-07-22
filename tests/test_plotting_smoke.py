"""
Smoke tests for `SiteSolutionSet`'s plotting methods.

These deliberately assert very little about *what* is drawn -- only that
each public plotting method runs to completion on a normal solved problem
and returns a figure-like object. They exist because the plotting surface
had no test coverage at all, which let two crash-on-call bugs sit in
released code:

  - `plot_travel_time_distribution(bottom_n=...)` raised TypeError from
    `list.append(temp_bottom=...)` for every call that passed `bottom_n`
  - `plot_solution_comparison` raised AttributeError from
    `self._get_ordinal_suffix(...)` -- a module-level helper called as if
    it were a method -- for every solution other than rank 1

Neither needed a sophisticated test to catch; simply calling the method
would have done it. That is all this file does.

Rendering is checked, not just construction: the Agg backend still
executes the full matplotlib draw path, so a bad artist or format string
surfaces here rather than in a user's notebook.
"""

import matplotlib

matplotlib.use("Agg")

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from shapely.geometry import Polygon

import lokigi


@pytest.fixture(autouse=True)
def close_figures():
    """Plotting methods create figures they do not own; close them after each
    test so a full run does not accumulate hundreds of open figures."""
    yield
    plt.close("all")


@pytest.fixture
def plottable_problem():
    """A small solved-able problem carrying everything the plotting methods
    need: demand, sites, a travel matrix, equity bands, and region geometry.

    Geometry is synthetic (three small squares near the site coordinates)
    rather than loaded from `sample_data`, to keep these tests fast and
    independent of the large boundary files.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 200, 150],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [40.0, 10.0, 8.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "imd_decile": [1, 5, 10],
        }
    )

    def square(x, y):
        return Polygon(
            [(x, y), (x + 0.05, y), (x + 0.05, y + 0.05), (x, y + 0.05)]
        )

    regions = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "geometry": [
                square(-0.10, 51.50),
                square(-0.20, 51.60),
                square(-0.30, 51.70),
            ],
        },
        crs="EPSG:4326",
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(
        demand_df, demand_col="demand", location_id_col="location_id"
    )
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    problem.add_region_geometry_layer(regions, common_col="location_id")
    return problem


@pytest.fixture
def solutions(plottable_problem):
    """mclp so the coverage-specific title branches are exercised."""
    return plottable_problem.solve(
        p=2,
        objectives="mclp",
        search_strategy="brute-force",
        show_progress=False,
        threshold_for_coverage=12,
    )


# --- one call per plotting method -----------------------------------------

# name -> callable taking the solved SiteSolutionSet. Parametrised rather
# than written out as separate tests so that adding a plotting method means
# adding one line here, and a failure names the method that broke.
PLOT_CALLS = {
    "plot_n_best_combinations_bar": lambda s: s.plot_n_best_combinations_bar(
        n_best=2
    ),
    "plot_best_combination": lambda s: s.plot_best_combination(),
    "plot_n_best_combinations": lambda s: s.plot_n_best_combinations(n_best=2),
    "plot_solution_comparison": lambda s: s.plot_solution_comparison(
        [{"solution_rank": 1}, {"solution_rank": 2}]
    ),
    "plot_travel_time_distribution": lambda s: s.plot_travel_time_distribution(
        top_n=2
    ),
    "check_solution_equity": lambda s: s.check_solution_equity(),
    "plot_top_n_solution_equity": lambda s: s.plot_top_n_solution_equity(n=2),
    "plot_combination_by_equity": lambda s: s.plot_combination_by_equity(),
    "plot_simple_pareto_front_pairs": lambda s: s.plot_simple_pareto_front_pairs(
        x_axis="max", y_axis="proportion_within_coverage_threshold"
    ),
    "plot_all_metric_pareto_front_pairs": (
        lambda s: s.plot_all_metric_pareto_front_pairs()
    ),
}


@pytest.mark.parametrize("method_name", sorted(PLOT_CALLS))
def test_plotting_method_runs_and_returns_something(solutions, method_name):
    result = PLOT_CALLS[method_name](solutions)
    assert result is not None, f"{method_name} returned None"


@pytest.mark.parametrize("method_name", sorted(PLOT_CALLS))
def test_plotting_method_output_renders(solutions, method_name):
    """Force a draw. Constructing a figure can succeed while rendering it
    fails (a bad artist, an unformattable value), and users hit the render.
    """
    result = PLOT_CALLS[method_name](solutions)

    figures = [
        figure
        for figure in (plt.figure(num) for num in plt.get_fignums())
        if figure.get_axes()
    ]
    for figure in figures:
        figure.canvas.draw()

    # Plotly-backed methods bypass matplotlib entirely; exercise their
    # serialisation instead, which is what rendering them ultimately needs.
    if type(result).__module__.startswith("plotly"):
        assert result.to_dict()


# --- the two specific crashes that motivated this file --------------------


def test_travel_time_distribution_with_bottom_n_does_not_raise(solutions):
    """Regression: `list.append(temp_bottom=...)` -- list.append() takes no
    keyword arguments, so passing bottom_n at all was an instant TypeError.
    """
    assert (
        solutions.plot_travel_time_distribution(top_n=1, bottom_n=1) is not None
    )


@pytest.mark.parametrize("solution_rank", [1, 2, 3])
def test_solution_comparison_handles_every_rank(solutions, solution_rank):
    """Regression: the ordinal-suffix helper was called as
    `self._get_ordinal_suffix(...)`, but it is a module-level function in
    lokigi.utils, so every rank except 1 raised AttributeError. Rank 1 took
    a different branch and worked, which is why this went unnoticed.
    """
    assert (
        solutions.plot_solution_comparison([{"solution_rank": solution_rank}])
        is not None
    )


def test_plot_solution_sets_comparison_handles_every_rank(solutions):
    """The same ordinal-suffix bug also sat in the standalone
    `plot_solution_sets_comparison` entry point in lokigi.plot_utils.
    """
    from lokigi.plot_utils import plot_solution_sets_comparison

    figure = plot_solution_sets_comparison(
        [solutions, solutions],
        [{"solution_rank": 2}, {"solution_rank": 3}],
    )
    assert figure is not None
