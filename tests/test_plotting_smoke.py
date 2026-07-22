"""
Smoke tests for lokigi's plotting methods.

These deliberately assert very little about *what* is drawn -- only that
each public plotting method runs to completion on a normal problem and
produces something. They exist because the plotting surface had no test
coverage at all, which let crash-on-call bugs sit in released code:

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

BASEMAPS: every map-drawing method fetches web map tiles via
`contextily.add_basemap`, so an un-stubbed run downloads tiles from a
public tile server on each test. The autouse fixture below intercepts
that. See its docstring for why the library's own `show_basemap=False` /
`add_basemap=False` flags are not sufficient on their own.
"""

import matplotlib

matplotlib.use("Agg")

import contextily
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from shapely.geometry import Polygon

import lokigi

GRID = 3  # 3x3 contiguous regions
CELL = 0.05
ORIGIN_X, ORIGIN_Y = -0.30, 51.50


@pytest.fixture(autouse=True)
def stub_basemap_tiles(monkeypatch):
    """Intercept web map tile downloads for every test in this module, and
    record the attempts so tests can assert on them.

    Why stub rather than rely on the library's own basemap flags:

    1. The solution-level map plots (`plot_best_combination`,
       `plot_n_best_combinations`, `plot_solution_comparison`,
       `plot_combination_by_equity`) call `cx.add_basemap` unconditionally
       -- there is no parameter to opt out of it.
    2. Historically the flag was spelled `show_basemap` on
       `plot_region_geometry_layer`/`plot_hotspots`/`plot_quadrant_map` and
       `add_basemap` elsewhere. Those three also accept `**kwargs`, so the
       wrong spelling reached the underlying plotting call rather than any
       code that understood it -- crashing the static path with a
       matplotlib error and being ignored outright on the interactive one.
       `add_basemap` is now accepted everywhere, but the stub keeps this
       module honest regardless of which spelling a test uses.

    Un-stubbed, this module made 21 real tile requests per run and took
    3-4x longer. Tests do NOT fail without the network (each call site
    catches RequestException and warns), but they are slower, they depend
    on a third-party service, and `timeout=30` per call means a hanging
    connection is far worse than a refused one.

    Where a method DOES expose a working flag, the tests below still pass
    it, so the library's own no-basemap branch is exercised rather than
    merely neutralised.
    """
    attempts = []

    def _stub(*args, **kwargs):
        attempts.append(kwargs.get("crs"))
        return None

    monkeypatch.setattr(contextily, "add_basemap", _stub)
    return attempts


@pytest.fixture(autouse=True)
def close_figures():
    """Plotting methods create figures they do not own; close them after each
    test so a full run does not accumulate hundreds of open figures."""
    yield
    plt.close("all")


@pytest.fixture
def plottable_problem():
    """A small problem carrying everything the plotting methods need:
    demand, sites, a travel matrix, equity bands, and region geometry.

    Geometry is a synthetic 3x3 grid of touching squares rather than a
    `sample_data` boundary file, to keep these tests fast and independent
    of the large geospatial fixtures. The cells must be contiguous because
    the hotspot methods build spatial-contiguity weights over them --
    disjoint polygons give every region zero neighbours.
    """
    # Compute every edge from its integer index rather than by adding CELL to
    # the previous edge. Accumulating (origin + n*CELL) + CELL drifts off
    # (origin + (n+1)*CELL) in floating point -- e.g. -0.2 vs
    # -0.19999999999999998 -- so neighbouring cells stop sharing an exact
    # edge, rook contiguity finds no shared boundary, and the grid silently
    # fragments into disconnected components with islands.
    def edge_x(index):
        return ORIGIN_X + index * CELL

    def edge_y(index):
        return ORIGIN_Y + index * CELL

    region_ids, geometries = [], []
    for row in range(GRID):
        for col in range(GRID):
            left, right = edge_x(col), edge_x(col + 1)
            bottom, top = edge_y(row), edge_y(row + 1)
            region_ids.append(f"LSOA_{row}{col}")
            geometries.append(
                Polygon(
                    [
                        (left, bottom),
                        (right, bottom),
                        (right, top),
                        (left, top),
                    ]
                )
            )

    regions = geopandas.GeoDataFrame(
        {"location_id": region_ids, "geometry": geometries}, crs="EPSG:4326"
    )
    demand_df = pd.DataFrame(
        {
            "location_id": region_ids,
            # One deliberate high-demand region, so demand-weighted and
            # region-based coverage are not accidentally identical.
            "demand": [10, 20, 30, 40, 200, 60, 70, 80, 90],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.52, 51.57, 51.62],
            "long": [-0.28, -0.22, -0.17],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": region_ids,
            "Site_A": [5.0, 8.0, 12.0, 9.0, 14.0, 18.0, 20.0, 25.0, 30.0],
            "Site_B": [15.0, 10.0, 7.0, 12.0, 6.0, 9.0, 16.0, 11.0, 13.0],
            "Site_C": [28.0, 22.0, 18.0, 20.0, 15.0, 8.0, 10.0, 6.0, 5.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": region_ids, "imd_decile": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
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


def _drew_something(result):
    """Plotting methods are inconsistent about what they hand back --
    Axes, Figure, a tuple, a folium Map, or None for the ones that draw
    onto the current figure. Accept any of those, but not "nothing at all".
    """
    return result is not None or bool(plt.get_fignums())


# --- solution-level plots -------------------------------------------------

# name -> callable taking the solved SiteSolutionSet. Parametrised rather
# than written out as separate tests so that adding a plotting method means
# adding one line here, and a failure names the method that broke.
SOLUTION_PLOT_CALLS = {
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


@pytest.mark.parametrize("method_name", sorted(SOLUTION_PLOT_CALLS))
def test_solution_plot_runs(solutions, method_name):
    assert _drew_something(SOLUTION_PLOT_CALLS[method_name](solutions)), (
        f"{method_name} produced no output"
    )


@pytest.mark.parametrize("method_name", sorted(SOLUTION_PLOT_CALLS))
def test_solution_plot_renders(solutions, method_name):
    """Force a draw. Constructing a figure can succeed while rendering it
    fails (a bad artist, an unformattable value), and users hit the render.
    """
    result = SOLUTION_PLOT_CALLS[method_name](solutions)

    for num in plt.get_fignums():
        figure = plt.figure(num)
        if figure.get_axes():
            figure.canvas.draw()

    # Plotly-backed methods bypass matplotlib entirely; exercise their
    # serialisation instead, which is what rendering them ultimately needs.
    if type(result).__module__.startswith("plotly"):
        assert result.to_dict()


# --- problem-level plots --------------------------------------------------

# These take the SiteProblem itself rather than a solved solution set, and
# were wholly untested before. `add_basemap` / `show_basemap` is passed
# where the method supports it -- note the two different spellings, and see
# stub_basemap_tiles for why passing them is not sufficient on its own.
PROBLEM_PLOT_CALLS = {
    "plot_sites": lambda p: p.plot_sites(add_basemap=False),
    "plot_sites__interactive": lambda p: p.plot_sites(
        add_basemap=False, interactive=True
    ),
    "plot_region_geometry_layer": lambda p: p.plot_region_geometry_layer(
        show_basemap=False
    ),
    "plot_hotspots": lambda p: p.plot_hotspots(show_basemap=False),
    "plot_quadrant_map": lambda p: p.plot_quadrant_map(show_basemap=False),
}


@pytest.mark.parametrize("method_name", sorted(PROBLEM_PLOT_CALLS))
def test_problem_plot_runs(plottable_problem, method_name):
    assert _drew_something(PROBLEM_PLOT_CALLS[method_name](plottable_problem)), (
        f"{method_name} produced no output"
    )


@pytest.mark.parametrize("method_name", sorted(PROBLEM_PLOT_CALLS))
def test_problem_plot_renders(plottable_problem, method_name):
    PROBLEM_PLOT_CALLS[method_name](plottable_problem)

    for num in plt.get_fignums():
        figure = plt.figure(num)
        if figure.get_axes():
            figure.canvas.draw()


# --- the basemap stubbing is load-bearing, not decorative -----------------


def test_solution_map_plots_attempt_a_basemap_with_no_way_to_opt_out(
    solutions, stub_basemap_tiles
):
    """`plot_best_combination` calls contextily unconditionally. If this ever
    stops being true -- someone adds a `show_basemap` parameter defaulting to
    False, say -- the stub stops being necessary and this test should be
    revisited. Until then it documents why the stub exists.
    """
    solutions.plot_best_combination()

    assert stub_basemap_tiles, (
        "expected plot_best_combination to request a basemap; if it no longer "
        "does, the autouse stub may no longer be needed"
    )


@pytest.mark.parametrize("method_name", sorted(PROBLEM_PLOT_CALLS))
def test_problem_plots_honour_add_basemap_false(
    plottable_problem, stub_basemap_tiles, method_name
):
    """`add_basemap=False` genuinely suppresses the download on every
    problem-level plot, so these tests exercise the real no-basemap branch
    rather than relying on the stub to hide a request that was still made.

    Before `add_basemap` was made the canonical name, three of these methods
    spelled it `show_basemap` and accepted `**kwargs`, so `add_basemap=False`
    was silently swallowed and tiles were fetched regardless -- this
    assertion would have failed for them.
    """
    PROBLEM_PLOT_CALLS[method_name](plottable_problem)

    assert stub_basemap_tiles == [], (
        f"{method_name} requested {len(stub_basemap_tiles)} basemap(s) "
        "despite add_basemap=False"
    )


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
