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
that. See its docstring for why the library's own `add_basemap=False`
flag is not sufficient on its own.
"""

import matplotlib

matplotlib.use("Agg")

import contextily
import geopandas
import json
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
    2. The methods that do expose `add_basemap` also accept `**kwargs` and
       forward them to the underlying plotting call, so a misspelling does
       not raise a normal TypeError naming the method -- it reaches
       matplotlib as a cryptic `PatchCollection.set()` error on static
       plots, or is ignored outright on interactive ones. The stub keeps
       this module hermetic even if such a slip creeps into a test.

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
            # Only read by plot_accessibility()/two_step_floating_catchment();
            # inert for every other test in this file.
            "supply": [5, 3, 8],
            # Only read by plot_site_utilisation()/site_utilisation_summary();
            # inert for every other test in this file.
            "capacity": [100, 200, 150],
            "current_load": [80, 200, 180],
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
    problem.add_sites(
        candidate_df,
        candidate_id_col="site_id",
        capacity_col="capacity",
        current_load_col="current_load",
    )
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
    "plot_site_allocation_summary": lambda s: s.plot_site_allocation_summary(),
    "plot_site_allocation_summary__average_travel_cost": (
        lambda s: s.plot_site_allocation_summary(metric="average_travel_cost")
    ),
    "plot_site_allocation_summary__allocated_demand": (
        lambda s: s.plot_site_allocation_summary(metric="allocated_demand")
    ),
    "plot_site_allocation_summary__n_regions": (
        lambda s: s.plot_site_allocation_summary(metric="n_regions")
    ),
    "plot_site_capacity_summary": lambda s: s.plot_site_capacity_summary(),
    "plot_site_capacity_summary__static": (
        lambda s: s.plot_site_capacity_summary(interactive=False)
    ),
    "plot_site_capacity_summary__incremental_headroom_ratio": (
        lambda s: s.plot_site_capacity_summary(metric="incremental_headroom_ratio")
    ),
    "plot_allocated_utilisation": lambda s: s.plot_allocated_utilisation(
        add_basemap=False
    ),
    "plot_allocated_utilisation__interactive": (
        lambda s: s.plot_allocated_utilisation(interactive=True, add_basemap=False)
    ),
    "plot_accessibility": lambda s: s.plot_accessibility(
        supply_col="supply", catchment_size=15, add_basemap=False
    ),
    "plot_accessibility__interactive": lambda s: s.plot_accessibility(
        supply_col="supply", catchment_size=15, add_basemap=False, interactive=True
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


# --- Colour bar / axis unit labelling ---------------------------------------
#
# plot_best_combination()'s default (cost-based) choropleth, plot_n_best_
# combinations()'s shared colorbar, check_solution_equity(), and
# plot_combination_by_equity() all used to show a bare, jargon-named colour
# bar/axis ("Min Cost", a raw min_cost/imd_decile column name) with no units
# and no plain-English label -- these pin the fix.


@pytest.fixture
def unit_labelled_problem():
    """Like plottable_problem, but with an explicit travel-matrix unit and
    a human-readable equity label, so the fixed labels can be checked for
    the actual unit/label text rather than just "present or absent"."""
    region_ids = ["LSOA_1", "LSOA_2", "LSOA_3"]
    regions = geopandas.GeoDataFrame(
        {
            "location_id": region_ids,
            "geometry": [
                Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)
            ],
        },
        crs="EPSG:4326",
    )
    demand_df = pd.DataFrame({"location_id": region_ids, "demand": [100, 200, 150]})
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": region_ids,
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )
    equity_df = pd.DataFrame({"location_id": region_ids, "imd_decile": [1, 5, 9]})

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="miles")
    problem.add_region_geometry_layer(regions, common_col="location_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD Decile",
        disadvantaged_end="low",
    )
    return problem


@pytest.fixture
def unit_labelled_solution(unit_labelled_problem):
    return unit_labelled_problem.solve(
        p=2, search_strategy="brute-force", show_progress=False
    )


def test_plot_best_combination_colorbar_labelled_with_unit(unit_labelled_solution):
    ax = unit_labelled_solution.plot_best_combination()
    colorbar_label = ax.figure.axes[-1].get_ylabel()
    assert colorbar_label == "Travel time to nearest site (miles)"


def test_plot_best_combination_colorbar_label_omits_unit_when_not_registered(
    solutions,
):
    """plottable_problem's travel matrix is registered with no unit=."""
    ax = solutions.plot_best_combination()
    colorbar_label = ax.figure.axes[-1].get_ylabel()
    assert colorbar_label == "Travel time to nearest site"


def test_plot_n_best_combinations_colorbar_labelled_with_unit(unit_labelled_solution):
    result = unit_labelled_solution.plot_n_best_combinations(n_best=2)
    fig = result[0] if isinstance(result, tuple) else result
    assert fig.axes[-1].get_ylabel() == "Travel time to nearest site (miles)"


def test_check_solution_equity_matplotlib_labels(unit_labelled_solution):
    fig = unit_labelled_solution.check_solution_equity(
        interactive=False, return_plot=True
    )
    axis = fig.axes[0]
    assert axis.get_xlabel() == "IMD Decile (most to least disadvantaged)"
    assert axis.get_ylabel() == "Average travel time (miles)"


def test_check_solution_equity_interactive_labels(unit_labelled_solution):
    fig = unit_labelled_solution.check_solution_equity(
        interactive=True, return_plot=True
    )
    assert fig.layout.xaxis.title.text == "IMD Decile (most to least disadvantaged)"
    assert fig.layout.yaxis.title.text == "Average travel time (miles)"


@pytest.fixture
def disadvantaged_high_solution(loaded_problem):
    """loaded_problem (Site_A/B/C, LSOA_1/2/3) with IMD deciles 1/5/10 on
    LSOA_1/2/3 respectively and disadvantaged_end="high" -- i.e. decile 10
    (not decile 1) is the most disadvantaged. unit_labelled_solution above
    uses disadvantaged_end="low", where ascending raw-bin order already
    happens to match most-to-least-disadvantaged order, so it can't tell
    a genuine reorder apart from an accidentally-correct default. This
    fixture can."""
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )
    loaded_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD Decile",
        disadvantaged_end="high",
    )
    return loaded_problem.solve(p=2, search_strategy="brute-force", show_progress=False)


def test_check_solution_equity_matplotlib_bars_reordered_for_disadvantaged_end_high(
    disadvantaged_high_solution,
):
    fig = disadvantaged_high_solution.check_solution_equity(
        interactive=False, return_plot=True
    )
    axis = fig.axes[0]
    assert [t.get_text() for t in axis.get_xticklabels()] == ["10", "5", "1"]


def test_check_solution_equity_plot_does_not_mutate_dataframe_return_order(
    disadvantaged_high_solution,
):
    """The non-plot DataFrame return is deliberately left in its original
    ascending order -- only the chart's bar order changes -- so calling
    check_solution_equity(return_plot=True) must not have side-mutated the
    order a subsequent return_plot=False call sees."""
    disadvantaged_high_solution.check_solution_equity(interactive=False, return_plot=True)
    table = disadvantaged_high_solution.check_solution_equity(return_plot=False)
    assert list(table["imd_decile"]) == [1, 5, 10]


def test_check_solution_equity_interactive_bars_reordered_for_disadvantaged_end_high(
    disadvantaged_high_solution,
):
    fig = disadvantaged_high_solution.check_solution_equity(
        interactive=True, return_plot=True
    )
    assert list(fig.layout.xaxis.categoryarray) == ["10", "5", "1"]


def test_plot_combination_by_equity_labels(unit_labelled_solution):
    fig, axes = unit_labelled_solution.plot_combination_by_equity()
    assert fig.axes[-1].get_ylabel() == "Travel time to nearest site (miles)"
    assert axes[0].get_title().startswith("IMD Decile:")


def test_plot_combination_by_equity_shows_region_and_people_counts(
    unit_labelled_solution,
):
    """A sparse panel (few regions/little demand) previously rendered as a
    near-empty map indistinguishable from "no problem here" -- the title now
    states the region and population headcount actually in that group."""
    fig, axes = unit_labelled_solution.plot_combination_by_equity()
    titles = [ax.get_title() for ax in axes[:3]]
    assert titles == [
        "IMD Decile: 1\n(1 regions, 100 people)",
        "IMD Decile: 5\n(1 regions, 200 people)",
        "IMD Decile: 9\n(1 regions, 150 people)",
    ]


# --- problem-level plots --------------------------------------------------

# These take the SiteProblem itself rather than a solved solution set, and
# were wholly untested before. `add_basemap=False` is passed throughout --
# it is the argument name on every plotting method as of v0.7.0 -- but see
# stub_basemap_tiles for why passing it is not sufficient on its own.
PROBLEM_PLOT_CALLS = {
    "plot_sites": lambda p: p.plot_sites(add_basemap=False),
    "plot_sites__interactive": lambda p: p.plot_sites(
        add_basemap=False, interactive=True
    ),
    "plot_region_geometry_layer": lambda p: p.plot_region_geometry_layer(
        add_basemap=False
    ),
    "plot_hotspots": lambda p: p.plot_hotspots(add_basemap=False),
    "plot_quadrant_map": lambda p: p.plot_quadrant_map(add_basemap=False),
    "plot_accessibility": lambda p: p.plot_accessibility(
        supply_col="supply", catchment_size=15, add_basemap=False
    ),
    "plot_accessibility__interactive": lambda p: p.plot_accessibility(
        supply_col="supply", catchment_size=15, add_basemap=False, interactive=True
    ),
    "plot_site_utilisation": lambda p: p.plot_site_utilisation(add_basemap=False),
    "plot_site_utilisation__interactive": lambda p: p.plot_site_utilisation(
        add_basemap=False, interactive=True
    ),
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


# --- plot_region_geometry_layer: colour bar label + title -----------------


def test_plot_region_geometry_layer_demand_colorbar_labelled(plottable_problem):
    ax = plottable_problem.plot_region_geometry_layer(
        plot_demand=True, add_basemap=False
    )
    assert ax.figure.axes[-1].get_ylabel() == "Demand"


def test_plot_region_geometry_layer_equity_colorbar_uses_registered_label(
    plottable_problem,
):
    ax = plottable_problem.plot_region_geometry_layer(
        plot_equity=True, add_basemap=False
    )
    assert ax.figure.axes[-1].get_ylabel() == "equity"


def test_plot_region_geometry_layer_equity_colorbar_falls_back_to_raw_column(
    plottable_problem,
):
    """If add_equity_data() was never given a label=, the colour bar
    falls back to the raw equity column name rather than showing nothing."""
    plottable_problem._equity_data_label = None
    ax = plottable_problem.plot_region_geometry_layer(
        plot_equity=True, add_basemap=False
    )
    assert ax.figure.axes[-1].get_ylabel() == "imd_decile"


def test_plot_region_geometry_layer_title_set_on_demand_plot(plottable_problem):
    ax = plottable_problem.plot_region_geometry_layer(
        plot_demand=True, add_basemap=False, title="Demand across LSOAs"
    )
    assert ax.get_title() == "Demand across LSOAs"


def test_plot_region_geometry_layer_title_set_on_equity_plot(plottable_problem):
    ax = plottable_problem.plot_region_geometry_layer(
        plot_equity=True, add_basemap=False, title="Equity across LSOAs"
    )
    assert ax.get_title() == "Equity across LSOAs"


def test_plot_region_geometry_layer_title_set_on_plain_plot(plottable_problem):
    ax = plottable_problem.plot_region_geometry_layer(
        add_basemap=False, title="Region boundaries"
    )
    assert ax.get_title() == "Region boundaries"


def test_plot_region_geometry_layer_no_title_by_default(plottable_problem):
    ax = plottable_problem.plot_region_geometry_layer(
        plot_demand=True, add_basemap=False
    )
    assert ax.get_title() == ""


# --- plot_region_geometry_layer: plot_region_of_interest_only (plain branch)
#
# The plain branch (neither plot_demand nor plot_equity) has its own
# region-of-interest filtering, separate from the plot_equity branch above.
# plottable_problem's demand covers every region 1:1, so it can't show
# filtering actually removing anything -- this fixture adds one extra
# region to the geometry layer with no corresponding demand row.


@pytest.fixture
def plottable_problem_with_extra_region(plottable_problem):
    extra = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_extra"],
            "geometry": [
                Polygon(
                    [
                        (ORIGIN_X + 3 * CELL, ORIGIN_Y),
                        (ORIGIN_X + 4 * CELL, ORIGIN_Y),
                        (ORIGIN_X + 4 * CELL, ORIGIN_Y + CELL),
                        (ORIGIN_X + 3 * CELL, ORIGIN_Y + CELL),
                    ]
                )
            ],
        },
        crs=plottable_problem.region_geometry_layer.crs,
    )
    plottable_problem.region_geometry_layer = pd.concat(
        [plottable_problem.region_geometry_layer, extra], ignore_index=True
    )
    return plottable_problem


def test_plot_region_geometry_layer_plain_plot_region_of_interest_only_filters(
    plottable_problem_with_extra_region,
):
    ax = plottable_problem_with_extra_region.plot_region_geometry_layer(
        add_basemap=False, plot_region_of_interest_only=True
    )
    assert len(ax.collections[0].get_paths()) == 9


def test_plot_region_geometry_layer_plain_plot_includes_all_regions_by_default(
    plottable_problem_with_extra_region,
):
    ax = plottable_problem_with_extra_region.plot_region_geometry_layer(
        add_basemap=False
    )
    assert len(ax.collections[0].get_paths()) == 10


def test_plot_region_geometry_layer_plain_plot_region_of_interest_only_does_not_raise(
    plottable_problem,
):
    """plot_region_of_interest_only=True previously raised UnboundLocalError
    in the plain branch (it referenced `plotting_df` before assigning it)."""
    ax = plottable_problem.plot_region_geometry_layer(
        add_basemap=False, plot_region_of_interest_only=True
    )
    assert len(ax.collections[0].get_paths()) == 9


# --- the basemap stubbing is load-bearing, not decorative -----------------


def test_solution_map_plots_attempt_a_basemap_with_no_way_to_opt_out(
    solutions, stub_basemap_tiles
):
    """`plot_best_combination` calls contextily unconditionally. If this ever
    stops being true -- someone adds an `add_basemap` parameter to the
    solution-level map plots, say -- the stub becomes less necessary and this
    test should be revisited. Until then it documents why the stub exists.
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

    Before `add_basemap` became the name on every plotting method, three of
    these spelled it `show_basemap` and accepted `**kwargs`, so
    `add_basemap=False` never reached code that understood it and tiles were
    fetched regardless -- this assertion would have failed for them.
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


def test_plot_best_combination_draws_site_markers_for_tabular_candidate_sites(
    solutions,
):
    """Regression: site markers were gated on
    `site_problem._candidate_sites_type == "geopandas"`, which records
    `add_sites()`'s *input* format, not whether `candidate_sites` ended up
    with real point geometry. `plottable_problem`'s `candidate_df` is
    tabular lat/long input (`_candidate_sites_type == "pandas"`), but
    `add_sites()` converts it to a real GeoDataFrame internally -- so the
    old check silently skipped site markers for this, the most common
    `add_sites()` usage pattern. Only the region choropleth was drawn
    (one collection); with the fix, chosen and unchosen site markers add
    two more.
    """
    ax = solutions.plot_best_combination()
    assert len(ax.collections) >= 3


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


def test_plot_site_allocation_summary_keeps_zero_allocation_bar_visible(
    five_site_problem,
):
    """A site closest to no region must still draw a bar (labelled 0%),
    not vanish from the chart the way it would from a plain groupby -- see
    the zero-allocation guard in `site_allocation_summary()`. Uses the
    shared `five_site_problem` fixture, where a p=3 solution over
    {Site_1, Site_2, Site_3} leaves Site_2 closest to nothing.
    """
    solved = five_site_problem.solve(p=3)
    site_names = ["Site_1", "Site_2", "Site_3"]

    static_fig = solved.plot_site_allocation_summary(
        site_names=site_names, interactive=False
    )
    assert isinstance(static_fig, plt.Figure)
    bars = static_fig.axes[0].containers[0]
    assert len(bars) == 3

    interactive_fig = solved.plot_site_allocation_summary(
        site_names=site_names, interactive=True
    )
    assert type(interactive_fig).__module__.startswith("plotly")
    assert len(interactive_fig.data[0].x) == 3


def test_plot_site_allocation_summary_average_travel_cost_shows_na_not_zero(
    five_site_problem,
):
    """With metric="average_travel_cost", a zero-allocation site (NaN in
    site_allocation_summary()) must be labelled "N/A", not a numeric "0.0"
    -- a real 0 would misleadingly read as "instant to reach" rather than
    "not applicable". The bar itself is still drawn (at zero length), so
    the site's row doesn't disappear from the chart."""
    solved = five_site_problem.solve(p=3)
    site_names = ["Site_1", "Site_2", "Site_3"]

    static_fig = solved.plot_site_allocation_summary(
        site_names=site_names, metric="average_travel_cost", interactive=False
    )
    ax = static_fig.axes[0]
    assert len(ax.containers[0]) == 3
    labels = [t.get_text() for t in ax.texts]
    assert "N/A" in labels
    assert sum(label == "N/A" for label in labels) == 1

    interactive_fig = solved.plot_site_allocation_summary(
        site_names=site_names, metric="average_travel_cost", interactive=True
    )
    assert len(interactive_fig.data[0].x) == 3
    assert "N/A" in interactive_fig.data[0].text
    # The NaN cost is filled to 0 for the bar's length (drawn, not hidden),
    # while its text label stays "N/A" -- these are deliberately different.
    na_index = list(interactive_fig.data[0].text).index("N/A")
    assert interactive_fig.data[0].x[na_index] == 0


def test_plot_site_allocation_summary_average_travel_cost_uses_registered_unit():
    """The bar labels and axis title should reflect whatever unit the
    travel matrix was registered with (e.g. "miles" for the average
    travel distance per patient use case), not just a bare number."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2"], "demand": [100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_A": [3.0, 12.0],
            "Site_B": [9.0, 4.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="miles")
    solved = problem.solve(p=2)

    fig = solved.plot_site_allocation_summary(
        metric="average_travel_cost", interactive=True
    )
    assert "miles" in fig.layout.xaxis.title.text
    assert all("miles" in text for text in fig.data[0].text)


def test_plot_site_allocation_summary_invalid_metric_raises(solutions):
    with pytest.raises(ValueError, match="metric must be"):
        solutions.plot_site_allocation_summary(metric="cost")


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


# --- plot_solution_sets_comparison() / SolutionComparator.plot_comparison() --
# shared colour scale --------------------------------------------------------
#
# Comparing e.g. a car-travel solution against a public-transport solution
# side by side used to let each panel autoscale its own colour bar
# independently -- a car map spanning 3-6 minutes and a PT map spanning
# 30-60 minutes rendered with visually IDENTICAL colour gradients, making
# the five-times-longer PT journeys look like a similar spread of outcomes.


@pytest.fixture
def differently_scaled_solutions():
    """Two SiteSolutionSets sharing sites/geometry but registered with
    travel matrices on very different scales (~3-12 vs ~30-120), like
    comparing car vs public transport travel times -- so a shared vs
    independent colour scale produces genuinely different, checkable
    vmin/vmax."""
    region_ids = ["LSOA_1", "LSOA_2", "LSOA_3"]
    regions = geopandas.GeoDataFrame(
        {
            "location_id": region_ids,
            "geometry": [
                Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(3)
            ],
        },
        crs="EPSG:4326",
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )

    base_problem = lokigi.site.SiteProblem(debug_mode=False)
    base_problem.add_sites(candidate_df, candidate_id_col="site_id")
    base_problem.add_region_geometry_layer(regions, common_col="location_id")

    problem_small_scale = base_problem.copy()
    problem_small_scale.add_travel_matrix(
        pd.DataFrame(
            {
                "source_id": region_ids,
                "Site_A": [5.0, 8.0, 12.0],
                "Site_B": [9.0, 3.0, 6.0],
                "Site_C": [11.0, 7.0, 4.0],
            }
        ),
        source_col="source_id",
        unit="minutes",
    )
    problem_large_scale = base_problem.copy()
    problem_large_scale.add_travel_matrix(
        pd.DataFrame(
            {
                "source_id": region_ids,
                "Site_A": [50.0, 80.0, 120.0],
                "Site_B": [90.0, 30.0, 60.0],
                "Site_C": [110.0, 70.0, 40.0],
            }
        ),
        source_col="source_id",
        unit="minutes",
    )

    with pytest.warns(UserWarning, match="No demand data was provided"):
        solution_small = problem_small_scale.solve(
            p=2, search_strategy="brute-force", show_progress=False
        )
        solution_large = problem_large_scale.solve(
            p=2, search_strategy="brute-force", show_progress=False
        )
    return solution_small, solution_large


def test_plot_comparison_shares_color_scale_by_default(differently_scaled_solutions):
    from lokigi.site_solutions import SolutionComparator

    solution_small, solution_large = differently_scaled_solutions
    comparator = SolutionComparator(solution_small, solution_large)

    fig, axes = comparator.plot_comparison()
    clims = [ax.collections[0].get_clim() for ax in axes]
    assert clims[0] == clims[1]
    assert clims[0] == (3.0, 60.0)


def test_plot_comparison_shared_color_scale_false_restores_independent_scales(
    differently_scaled_solutions,
):
    from lokigi.site_solutions import SolutionComparator

    solution_small, solution_large = differently_scaled_solutions
    comparator = SolutionComparator(solution_small, solution_large)

    fig, axes = comparator.plot_comparison(shared_color_scale=False)
    clims = [ax.collections[0].get_clim() for ax in axes]
    assert clims[0] != clims[1]
    assert clims[0] == (3.0, 6.0)
    assert clims[1] == (30.0, 60.0)


def test_plot_solution_sets_comparison_categorical_panel_unaffected_by_shared_scale(
    differently_scaled_solutions,
):
    """A plot_site_allocation panel colours by categorical site, not travel
    cost -- it must not be forced onto the cost-based shared vmin/vmax, and
    must not itself pollute the shared scale computed for the other,
    genuinely cost-based panel."""
    from lokigi.plot_utils import plot_solution_sets_comparison

    solution_small, solution_large = differently_scaled_solutions

    fig, axes = plot_solution_sets_comparison(
        [solution_small, solution_large],
        [{"plot_site_allocation": True}, {}],
    )
    assert fig is not None
    # The cost-based panel (index 1) is unaffected by the categorical one --
    # its own scale, not some mix that includes categorical site indices.
    assert axes[1].collections[0].get_clim() == (30.0, 60.0)


# --- SolutionComparator.plot_population_impact_summary ---------------------


@pytest.fixture
def population_impact_comparator(plottable_problem):
    from lokigi.site_solutions import SolutionComparator

    baseline = plottable_problem.evaluate_baseline(site_names=["Site_A"])
    candidate = plottable_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    return SolutionComparator(baseline, candidate, labels=("Current", "Proposed"))


def test_plot_population_impact_summary_runs_and_renders(population_impact_comparator):
    fig, axes = population_impact_comparator.plot_population_impact_summary()
    assert fig is not None
    assert len(axes) == 2
    fig.canvas.draw()


def test_plot_population_impact_summary_by_regions(population_impact_comparator):
    fig, axes = population_impact_comparator.plot_population_impact_summary(by="regions")
    assert fig is not None
    fig.canvas.draw()


def test_plot_population_impact_summary_invalid_by_raises(population_impact_comparator):
    with pytest.raises(ValueError, match="by must be"):
        population_impact_comparator.plot_population_impact_summary(by="nonsense")


def test_plot_population_impact_summary_reuses_given_axes(population_impact_comparator):
    fig, axes = plt.subplots(ncols=2)
    out_fig, out_axes = population_impact_comparator.plot_population_impact_summary(ax=axes)
    assert out_fig is fig
    assert out_axes is axes


# --- SolutionComparator.plot_site_reallocation_matrix -----------------------


def test_plot_site_reallocation_matrix_runs_and_renders(population_impact_comparator):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix()
    assert fig is not None
    fig.canvas.draw()


def test_plot_site_reallocation_matrix_by_regions(population_impact_comparator):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(by="regions")
    assert fig is not None
    fig.canvas.draw()


def test_plot_site_reallocation_matrix_invalid_by_raises(population_impact_comparator):
    with pytest.raises(ValueError, match="by must be"):
        population_impact_comparator.plot_site_reallocation_matrix(by="nonsense")


def test_plot_site_reallocation_matrix_reuses_given_axes(population_impact_comparator):
    fig, ax = plt.subplots()
    out_fig, out_ax = population_impact_comparator.plot_site_reallocation_matrix(ax=ax)
    assert out_fig is fig
    assert out_ax is ax


def test_plot_site_reallocation_matrix_axis_labels_are_comparator_labels(
    population_impact_comparator,
):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix()
    assert ax.get_xlabel() == "Proposed"
    assert ax.get_ylabel() == "Current"


def test_plot_site_reallocation_matrix_colorbar_label_matches_by(
    population_impact_comparator,
):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(by="demand")
    assert fig.axes[-1].get_ylabel() == "People"

    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(by="regions")
    assert fig.axes[-1].get_ylabel() == "Regions"


def test_plot_site_reallocation_matrix_default_caption_shown(
    population_impact_comparator,
):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix()
    assert len(fig.texts) == 1
    assert "How to read this" in fig.texts[0].get_text()


def test_plot_site_reallocation_matrix_caption_suppressed_with_empty_string(
    population_impact_comparator,
):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(caption="")
    assert len(fig.texts) == 0


def test_plot_site_reallocation_matrix_custom_caption(population_impact_comparator):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(
        caption="Custom explanation."
    )
    assert len(fig.texts) == 1
    assert "Custom explanation." in fig.texts[0].get_text()


def test_plot_site_reallocation_matrix_changed_only_runs_and_renders(
    population_impact_comparator,
):
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(
        changed_only=True
    )
    assert fig is not None
    fig.canvas.draw()


def test_plot_site_reallocation_matrix_changed_only_matches_dataframe_method(
    population_impact_comparator,
):
    expected = population_impact_comparator.site_reallocation_matrix(changed_only=True)
    fig, ax = population_impact_comparator.plot_site_reallocation_matrix(
        changed_only=True
    )
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert ytick_labels == list(expected.index)


def test_plot_population_impact_histogram_runs_and_renders(population_impact_comparator):
    fig, axis = population_impact_comparator.plot_population_impact_histogram()
    assert fig is not None
    fig.canvas.draw()

    legend_labels = [t.get_text() for t in axis.get_legend().get_texts()]
    assert "Current" in legend_labels
    assert "Proposed" in legend_labels
    assert any(label.startswith("Mean --") for label in legend_labels)
    assert any(label.startswith("Max --") for label in legend_labels)


def test_plot_population_impact_histogram_reuses_given_axis(population_impact_comparator):
    fig, axis = plt.subplots()
    out_fig, out_axis = population_impact_comparator.plot_population_impact_histogram(ax=axis)
    assert out_fig is fig
    assert out_axis is axis


def test_plot_population_impact_histogram_ylabel_reflects_demand_availability(
    plottable_problem,
):
    from lokigi.site_solutions import SolutionComparator

    baseline = plottable_problem.evaluate_baseline(site_names=["Site_A"])
    candidate = plottable_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    fig, axis = comparator.plot_population_impact_histogram(kind="hist")
    assert axis.get_ylabel() == "People"


@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_population_impact_histogram_kind_runs_and_renders(
    population_impact_comparator, kind
):
    fig, axis = population_impact_comparator.plot_population_impact_histogram(kind=kind)
    assert fig is not None
    fig.canvas.draw()


def test_plot_population_impact_histogram_invalid_kind_raises(population_impact_comparator):
    with pytest.raises(ValueError, match="kind must be"):
        population_impact_comparator.plot_population_impact_histogram(kind="bogus")


def test_plot_population_impact_histogram_kde_default_ylabel_is_plain_english(
    population_impact_comparator,
):
    """"Density" means nothing to a non-technical reader -- the default
    kind="kde" label should read as a plain-English share, not a stats term."""
    fig, axis = population_impact_comparator.plot_population_impact_histogram()
    assert axis.get_ylabel() == "Relative share of people"
    assert "density" not in axis.get_ylabel().lower()


def test_plot_population_impact_histogram_kde_default_caption_explains_curve(
    population_impact_comparator,
):
    fig, axis = population_impact_comparator.plot_population_impact_histogram()
    assert len(fig.texts) == 1
    caption = fig.texts[0].get_text().replace("\n", " ")
    assert "not a literal headcount" in caption
    assert 'kind="hist"' in caption


def test_plot_population_impact_histogram_hist_has_no_default_caption(
    population_impact_comparator,
):
    """A histogram's bar heights are already literal counts -- no caption
    needed to explain them, unlike the kde curve."""
    fig, axis = population_impact_comparator.plot_population_impact_histogram(kind="hist")
    assert len(fig.texts) == 0


def test_plot_population_impact_histogram_caption_can_be_suppressed(
    population_impact_comparator,
):
    fig, axis = population_impact_comparator.plot_population_impact_histogram(caption="")
    assert len(fig.texts) == 0


def test_plot_population_impact_histogram_custom_caption(population_impact_comparator):
    fig, axis = population_impact_comparator.plot_population_impact_histogram(
        caption="Custom explanation."
    )
    assert len(fig.texts) == 1
    assert "Custom explanation." in fig.texts[0].get_text()


# --- SolutionComparator.plot_population_impact_by_equity_group --------------


def test_plot_population_impact_by_equity_group_runs_and_renders(
    population_impact_comparator,
):
    fig, axis = population_impact_comparator.plot_population_impact_by_equity_group()
    assert fig is not None
    fig.canvas.draw()

    legend_labels = [t.get_text() for t in axis.get_legend().get_texts()]
    assert "Improved" in legend_labels
    assert "Worsened" in legend_labels
    # 9 imd_decile bands in plottable_problem -> 9 paired bars.
    assert len(axis.get_xticklabels()) == 9


def test_plot_population_impact_by_equity_group_reuses_given_axis(
    population_impact_comparator,
):
    fig, axis = plt.subplots()
    out_fig, out_axis = population_impact_comparator.plot_population_impact_by_equity_group(
        ax=axis
    )
    assert out_fig is fig
    assert out_axis is axis


def test_plot_population_impact_by_equity_group_xlabel_uses_equity_label(
    population_impact_comparator,
):
    """The x-axis previously used the raw equity column name (by_band.index
    .name, i.e. _equity_data_equity_col) -- broken/unreadable for a column
    like the real-world IMD dataset's
    "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived
    10% of LSOA" (note the unbalanced parenthesis -- a genuine raw column
    name, not a typo introduced here). Now prefers the human-readable
    add_equity_data(label=...) instead. plottable_problem registers
    label="equity"."""
    fig, axis = population_impact_comparator.plot_population_impact_by_equity_group()
    assert axis.get_xlabel() == "equity (most to least disadvantaged)"


def test_plot_population_impact_by_equity_group_no_equity_data_raises():
    from lokigi.site_solutions import SolutionComparator

    problem = lokigi.site.SiteProblem(debug_mode=False)
    demand_df = pd.DataFrame({"location_id": ["A", "B"], "demand": [10, 20]})
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.1, 51.2], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame({"source_id": ["A", "B"], "Site_A": [5.0, 8.0], "Site_B": [9.0, 3.0]})
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    baseline = problem.evaluate_baseline(site_names=["Site_A"])
    candidate = problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    with pytest.raises(ValueError, match="equity data"):
        comparator.plot_population_impact_by_equity_group()


# --- SolutionComparator.plot_population_impact_map --------------------------


@pytest.fixture
def closure_impact_comparator(plottable_problem):
    """Unlike `population_impact_comparator` (adding Site_B, a pure superset
    that only ever improves or leaves things unchanged), this closes Site_A
    -- baseline={Site_A, Site_B}, candidate={Site_B} -- so some regions
    genuinely worsen, exercising the "worsened" bucket/direction rather than
    always hitting the empty-subset path."""
    from lokigi.site_solutions import SolutionComparator

    baseline = plottable_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    candidate = plottable_problem.evaluate_baseline(site_names=["Site_B"])
    return SolutionComparator(baseline, candidate, labels=("Current", "Proposed"))


def test_plot_population_impact_map_all_runs_and_renders(population_impact_comparator):
    fig, axis = population_impact_comparator.plot_population_impact_map()
    assert fig is not None
    fig.canvas.draw()
    assert axis.get_title() == "Change in travel time by region"


def test_plot_population_impact_map_worsened_runs_and_renders(
    closure_impact_comparator,
):
    fig, axis = closure_impact_comparator.plot_population_impact_map(
        direction="worsened"
    )
    fig.canvas.draw()
    assert axis.get_title() == "Regions with a longer journey"


def test_plot_population_impact_map_improved_runs_and_renders(
    population_impact_comparator,
):
    fig, axis = population_impact_comparator.plot_population_impact_map(
        direction="improved"
    )
    fig.canvas.draw()
    assert axis.get_title() == "Regions with a shorter journey"


def test_plot_population_impact_map_empty_bucket_does_not_crash(
    population_impact_comparator,
):
    """population_impact_comparator only ever adds a site (Site_A -> Site_A
    + Site_B), so nothing ever worsens -- the "worsened" bucket is empty.
    Must render the background map, not raise or draw a broken colour bar."""
    fig, axis = population_impact_comparator.plot_population_impact_map(
        direction="worsened"
    )
    fig.canvas.draw()


def test_plot_population_impact_map_n_matches_worst_affected(closure_impact_comparator):
    worst_affected = closure_impact_comparator.population_impact_worst_affected(
        n=2, direction="worsened"
    )
    fig, axis = closure_impact_comparator.plot_population_impact_map(
        direction="worsened", n=2
    )
    fig.canvas.draw()
    assert len(worst_affected) == 2


def test_plot_population_impact_map_n_with_all_direction_raises(
    population_impact_comparator,
):
    with pytest.raises(ValueError, match="n is only valid"):
        population_impact_comparator.plot_population_impact_map(direction="all", n=5)


def test_plot_population_impact_map_invalid_direction_raises(
    population_impact_comparator,
):
    with pytest.raises(ValueError, match="direction must be"):
        population_impact_comparator.plot_population_impact_map(direction="sideways")


def test_plot_population_impact_map_show_sites_all_runs_and_renders(
    closure_impact_comparator,
):
    """closure_impact_comparator closes Site_A -- show_sites="all" should
    draw every candidate site, with Site_A picked out distinctly."""
    fig, axis = closure_impact_comparator.plot_population_impact_map(
        show_sites="all"
    )
    fig.canvas.draw()
    # 1 collection for the closed-site marker + 1 for the other (open) sites,
    # on top of however many the region choropleth itself drew.
    assert len(axis.collections) >= 2


def test_plot_population_impact_map_show_sites_closed_only(closure_impact_comparator):
    fig, axis = closure_impact_comparator.plot_population_impact_map(
        show_sites="closed"
    )
    fig.canvas.draw()
    assert len(axis.collections) >= 1


def test_plot_population_impact_map_show_sites_none_by_default_draws_no_markers(
    closure_impact_comparator,
):
    """The choropleth alone draws exactly one collection (the region
    polygons); show_sites=None (the default) must not add any more."""
    fig, axis = closure_impact_comparator.plot_population_impact_map()
    fig.canvas.draw()
    assert len(axis.collections) == 1


def test_plot_population_impact_map_invalid_show_sites_raises(
    population_impact_comparator,
):
    with pytest.raises(ValueError, match="show_sites must be"):
        population_impact_comparator.plot_population_impact_map(show_sites="bogus")


def test_plot_population_impact_map_show_sites_without_point_geometry_raises():
    """show_sites requires point geometry on candidate_sites -- the fallback
    path where add_sites() is never called (candidate_sites auto-derived
    from the travel matrix's own columns via _setup_sites_df_from_travel_matrix)
    leaves it a plain DataFrame with no geometry at all, matching the same
    scenario plot_best_combination() itself has to guard against."""
    from lokigi.site_solutions import SolutionComparator

    region_ids = ["LSOA_1", "LSOA_2"]
    regions = geopandas.GeoDataFrame(
        {
            "location_id": region_ids,
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    demand_df = pd.DataFrame({"location_id": region_ids, "demand": [100, 200]})
    travel_df = pd.DataFrame(
        {"location_id": region_ids, "Site_A": [10.0, 20.0], "Site_B": [25.0, 5.0]}
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_travel_matrix(travel_df, source_col="location_id")
    problem.add_region_geometry_layer(regions, common_col="location_id")

    with pytest.warns(UserWarning, match="No candidate site dataframe was given"):
        candidate = problem.solve(p=1, objectives="p_median", show_progress=False)
    baseline = problem.evaluate_baseline(site_names=["Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    with pytest.raises(ValueError, match="point geometry"):
        comparator.plot_population_impact_map(show_sites="all")

    # show_sites=None (the default) must still work fine without geometry.
    fig, axis = comparator.plot_population_impact_map()
    fig.canvas.draw()


def test_plot_population_impact_map_reuses_given_axis(population_impact_comparator):
    fig, axis = plt.subplots()
    out_fig, out_axis = population_impact_comparator.plot_population_impact_map(ax=axis)
    assert out_fig is fig
    assert out_axis is axis


def test_plot_population_impact_map_legend_mentions_unit():
    """The colour bar should reflect whatever unit the travel matrix was
    registered with, not just a bare "Change in travel time"."""
    from lokigi.site_solutions import SolutionComparator

    region_ids = ["LSOA_1", "LSOA_2"]
    regions = geopandas.GeoDataFrame(
        {
            "location_id": region_ids,
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                         Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    demand_df = pd.DataFrame({"location_id": region_ids, "demand": [100, 200]})
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.1, 51.2], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {"source_id": region_ids, "Site_A": [10.0, 20.0], "Site_B": [25.0, 5.0]}
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="minutes")
    problem.add_region_geometry_layer(regions, common_col="location_id")

    baseline = problem.evaluate_baseline(site_names=["Site_A"])
    candidate = problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    fig, axis = comparator.plot_population_impact_map()
    fig.canvas.draw()
    colorbar_label = fig.axes[-1].get_ylabel()
    assert "minutes" in colorbar_label


def test_plot_population_impact_map_no_region_geometry_raises():
    from lokigi.site_solutions import SolutionComparator

    problem = lokigi.site.SiteProblem(debug_mode=False)
    demand_df = pd.DataFrame({"location_id": ["A", "B"], "demand": [10, 20]})
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.1, 51.2], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame({"source_id": ["A", "B"], "Site_A": [5.0, 8.0], "Site_B": [9.0, 3.0]})
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    baseline = problem.evaluate_baseline(site_names=["Site_A"])
    candidate = problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, candidate)

    with pytest.raises(ValueError, match="region_geometry_layer"):
        comparator.plot_population_impact_map()


# --- interactive maps: hidden-container zoom guard ------------------------

# Leaflet takes its zoom from the pixel size of the map container. A map
# initialised inside a hidden container (a non-active reveal.js slide, an
# inactive Quarto tabset panel) measures 0x0 and fitBounds clamps to zoom
# 0 -- the whole world -- and nothing re-fits it when the container is
# later shown. Every interactive map therefore carries a one-shot
# ResizeObserver that re-applies the bounds on first real layout.

# Every code path that returns a folium map, including the three
# separate branches of plot_region_geometry_layer and the two in
# site_eda that build their map by hand rather than via .explore().
INTERACTIVE_MAP_CALLS = {
    **{
        name: call
        for name, call in PROBLEM_PLOT_CALLS.items()
        if name.endswith("__interactive")
    },
    "plot_region_geometry_layer__plain": lambda p: p.plot_region_geometry_layer(
        interactive=True, add_basemap=False
    ),
    "plot_region_geometry_layer__demand": lambda p: p.plot_region_geometry_layer(
        interactive=True, add_basemap=False, plot_demand=True
    ),
    "plot_region_geometry_layer__equity": lambda p: p.plot_region_geometry_layer(
        interactive=True, add_basemap=False, plot_equity=True
    ),
    "plot_hotspots": lambda p: p.plot_hotspots(
        interactive=True, add_basemap=False, verbose=False
    ),
    "plot_quadrant_map": lambda p: p.plot_quadrant_map(
        interactive=True, add_basemap=False, verbose=False
    ),
}


def _guard_children(folium_map):
    from lokigi.plot_utils import _DeferredFitBounds

    return [
        child
        for child in folium_map._children.values()
        if isinstance(child, _DeferredFitBounds)
    ]


@pytest.mark.parametrize("method_name", sorted(INTERACTIVE_MAP_CALLS))
def test_interactive_map_carries_zoom_guard(plottable_problem, method_name):
    folium_map = INTERACTIVE_MAP_CALLS[method_name](plottable_problem)

    assert len(_guard_children(folium_map)) == 1, (
        f"{method_name} returned a map with no hidden-container zoom guard"
    )

    # It must be added last: the guard reads the map's bounds off its
    # children, so anything added afterwards would not be fitted to.
    assert _guard_children(folium_map)[0] is list(folium_map._children.values())[-1]


@pytest.mark.parametrize("method_name", sorted(INTERACTIVE_MAP_CALLS))
def test_interactive_map_zoom_guard_renders(plottable_problem, method_name):
    """The guard is a Jinja template, so a bad reference renders silently
    wrong rather than raising -- assert on the emitted JavaScript."""
    folium_map = INTERACTIVE_MAP_CALLS[method_name](plottable_problem)
    html = folium_map.get_root().render()

    assert "ResizeObserver" in html
    assert "invalidateSize" in html

    # Bound to the right Leaflet map object, and re-fitting to the real
    # data extent rather than a placeholder.
    assert f"var map = {folium_map.get_name()};" in html
    bounds = folium_map.get_bounds()
    assert f"var bounds = {json.dumps(bounds)};" in html

    # Must run after Leaflet has created the map.
    assert html.index("ResizeObserver") > html.index("L.map(")


def test_zoom_guard_skipped_when_map_has_no_boundable_layers():
    """An empty map has bounds of [[None, None], [None, None]]; fitting to
    that would emit `fitBounds([[null, null], ...])` and break the map."""
    import folium

    from lokigi.plot_utils import _attach_deferred_fit_bounds

    empty = folium.Map()

    assert _attach_deferred_fit_bounds(empty) is empty
    assert _guard_children(empty) == []


def test_zoom_guard_kill_switch(plottable_problem, monkeypatch):
    import lokigi.plot_utils

    monkeypatch.setattr(lokigi.plot_utils, "DEFERRED_FIT_BOUNDS", False)

    folium_map = plottable_problem.plot_sites(add_basemap=False, interactive=True)

    assert _guard_children(folium_map) == []
    assert "ResizeObserver" not in folium_map.get_root().render()
