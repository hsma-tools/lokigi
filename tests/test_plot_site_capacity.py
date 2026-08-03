"""Targeted tests for `plot_site_capacity_summary()` and
`plot_allocated_utilisation()`, complementing the generic run/render smoke
coverage in `test_plotting_smoke.py` with behaviours specific to these two
methods -- especially the `inf`/NaN handling that a plain render can miss
(a broken axis or a flattened marker size doesn't raise, it just looks
wrong).
"""

import matplotlib

matplotlib.use("Agg")

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import lokigi


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _build_problem(candidate_df, capacity_kwargs):
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", **capacity_kwargs)
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def capacity_candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
            "capacity": [200, 250, 999],
        }
    )


@pytest.fixture
def capacitated_solution(capacity_candidate_df):
    problem = _build_problem(capacity_candidate_df, {"capacity_col": "capacity"})
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def capacitated_solution_with_load(capacity_candidate_df):
    df = capacity_candidate_df.copy()
    df["current_load"] = [50, 300, 0]
    problem = _build_problem(
        df, {"capacity_col": "capacity", "current_load_col": "current_load"}
    )
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def zero_capacity_solution(capacity_candidate_df):
    df = capacity_candidate_df.copy()
    df.loc[df["site_id"] == "Site_B", "capacity"] = 0
    problem = _build_problem(df, {"capacity_col": "capacity"})
    return problem.solve(p=2, search_strategy="brute-force", show_progress=False)


@pytest.fixture
def capacity_candidate_gdf():
    return geopandas.GeoDataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "capacity": [200, 250, 0],
            "geometry": [Point(-0.1, 51.5), Point(-0.2, 51.6), Point(-0.3, 51.7)],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def geo_capacitated_solution(capacity_candidate_gdf):
    problem = _build_problem(capacity_candidate_gdf, {"capacity_col": "capacity"})
    return problem.solve(p=3, search_strategy="brute-force", show_progress=False)


# --- plot_site_capacity_summary: validation ----------------------------------


def test_invalid_metric_message_lists_both_options(capacitated_solution):
    with pytest.raises(ValueError, match="allocated_utilisation_ratio"):
        capacitated_solution.plot_site_capacity_summary(metric="bogus")


def test_headroom_ratio_metric_raises_without_load_data_naming_add_sites(
    capacitated_solution,
):
    with pytest.raises(ValueError, match="add_sites"):
        capacitated_solution.plot_site_capacity_summary(
            metric="incremental_headroom_ratio"
        )


# --- plot_site_capacity_summary: rendering behaviour -------------------------


def test_over_capacity_bars_use_a_distinct_colour(capacitated_solution):
    fig = capacitated_solution.plot_site_capacity_summary(
        interactive=False, site_names=["Site_A", "Site_B"], sort=False
    )
    ax = fig.axes[0]
    bars = [p for p in ax.patches]
    colours = {bar.get_facecolor() for bar in bars}
    # Site_A is under capacity (0.5), Site_B is over (1.4) -- two bars,
    # two different colours.
    assert len(colours) == 2


def test_reference_line_drawn_at_one(capacitated_solution):
    fig = capacitated_solution.plot_site_capacity_summary(interactive=False)
    ax = fig.axes[0]
    line_positions = [line.get_xdata()[0] for line in ax.lines]
    assert any(pytest.approx(1.0) == pos for pos in line_positions)


def test_reference_line_omitted_when_disabled(capacitated_solution):
    fig = capacitated_solution.plot_site_capacity_summary(
        interactive=False, show_reference_line=False
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 0


def test_infinite_ratio_renders_without_breaking_the_axis(zero_capacity_solution):
    fig = zero_capacity_solution.plot_site_capacity_summary(
        interactive=False, site_names=["Site_A", "Site_B"]
    )
    fig.canvas.draw()
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    assert all(np.isfinite(xlim))

    labels = [t.get_text() for t in ax.texts]
    assert any("zero capacity" in label for label in labels)


def test_nan_ratio_bar_labelled_na_not_zero(capacitated_solution):
    candidate = capacitated_solution.site_problem.candidate_sites.copy()
    candidate.loc[candidate["site_id"] == "Site_B", "capacity"] = np.nan
    capacitated_solution.site_problem.candidate_sites = candidate

    with pytest.warns(UserWarning):
        fig = capacitated_solution.plot_site_capacity_summary(
            interactive=False, site_names=["Site_A", "Site_B"]
        )

    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.texts]
    assert "N/A" in labels


def test_unreachable_fragment_appears_in_default_title():
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.5, 51.6],
            "long": [-0.1, -0.2],
            "capacity": [500, 500],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 12.0, 14.0, np.nan],
            "Site_B": [20.0, 22.0, 24.0, np.nan],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", capacity_col="capacity")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)
    result = problem.solve(p=1, unreachable_cost=1000, show_progress=False)

    with pytest.warns(UserWarning):
        fig = result.plot_site_capacity_summary(interactive=False)
    ax = fig.axes[0]
    assert "excluded" in ax.get_title()


def test_precomputed_capacity_df_is_used_verbatim(capacitated_solution):
    doctored = capacitated_solution.site_capacity_summary(
        site_names=["Site_A", "Site_B"]
    ).copy()
    doctored.loc["Site_A", "allocated_utilisation_ratio"] = 0.99

    fig = capacitated_solution.plot_site_capacity_summary(
        capacity_df=doctored, interactive=False
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.texts]
    assert "99%" in labels


def test_ax_embeds_into_a_caller_owned_figure_without_closing_it(capacitated_solution):
    fig, axes = plt.subplots(ncols=2)
    other_ax = axes[1]
    other_ax.bar(["x", "y"], [1, 2])

    returned = capacitated_solution.plot_site_capacity_summary(
        interactive=False, ax=axes[0]
    )

    assert returned is fig
    assert plt.fignum_exists(fig.number)
    # The other panel is untouched -- this method only drew into axes[0].
    assert [b.get_height() for b in other_ax.patches] == [1, 2]
    fig.canvas.draw()


def test_ax_is_ignored_when_interactive(capacitated_solution):
    fig, ax = plt.subplots()
    result = capacitated_solution.plot_site_capacity_summary(interactive=True, ax=ax)
    assert type(result).__module__.startswith("plotly")


def test_rate_fragment_in_title_when_rate_not_default(capacitated_solution):
    fig = capacitated_solution.plot_site_capacity_summary(
        interactive=False, demand_to_capacity_rate=2.5
    )
    ax = fig.axes[0]
    assert "2.5" in ax.get_title()


def test_no_rate_fragment_at_default_rate(capacitated_solution):
    fig = capacitated_solution.plot_site_capacity_summary(interactive=False)
    ax = fig.axes[0]
    assert "capacity units" not in ax.get_title()


# --- plot_allocated_utilisation -----------------------------------------------


def test_map_raises_without_site_geometry():
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    travel_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_travel_matrix(travel_df, source_col="location_id")
    with pytest.warns(UserWarning):
        result = problem.solve(p=1, search_strategy="brute-force", show_progress=False)

    with pytest.raises(ValueError, match="real site geometry"):
        result.plot_allocated_utilisation()


def test_map_marker_sizes_stay_finite_with_an_infinite_ratio(geo_capacitated_solution):
    ax = geo_capacitated_solution.plot_allocated_utilisation(add_basemap=False)
    fig = ax.get_figure()
    fig.canvas.draw()
    site_collection = ax.collections[0]
    sizes = site_collection.get_sizes()
    assert all(np.isfinite(sizes))
    assert len(sizes) > 0


def test_map_draws_only_selected_sites():
    """4 candidate sites, p=3 -- one site is always left out of the
    solution, and its marker must not appear on the map."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_gdf = geopandas.GeoDataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "capacity": [200, 250, 999, 100],
            "geometry": [
                Point(-0.1, 51.5),
                Point(-0.2, 51.6),
                Point(-0.3, 51.7),
                Point(-0.4, 51.8),
            ],
        },
        crs="EPSG:4326",
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
            "Site_D": [40.0, 40.0, 40.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_gdf, candidate_id_col="site_id", capacity_col="capacity")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    result = problem.solve(p=3, search_strategy="brute-force", show_progress=False)

    selected = list(result.solution_df["site_names"].iloc[0])
    assert len(selected) == 3
    assert "Site_D" not in selected  # Site_D is dominated on every region

    ax = result.plot_allocated_utilisation(add_basemap=False)
    fig = ax.get_figure()
    fig.canvas.draw()
    assert len(ax.collections[0].get_offsets()) == 3


def test_interactive_map_returns_folium_map(geo_capacitated_solution):
    import folium

    m = geo_capacitated_solution.plot_allocated_utilisation(
        interactive=True, add_basemap=False
    )
    assert isinstance(m, folium.Map)
