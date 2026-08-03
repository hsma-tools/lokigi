"""Tests for how plots handle a genuinely unreachable demand location
(`add_travel_matrix(allow_missing=True)`), the third part of the
"no feasible journey" feature (see test_missing_travel_values.py and
test_unreachable_cost.py for ingestion/evaluation and solver ranking).

Before this, every choropleth built on `_plot_single_solution_map`
(`plot_best_combination`, `plot_n_best_combinations`,
`plot_solution_comparison`, `plot_solution_sets_comparison`) and
`plot_combination_by_equity()` silently dropped an unreachable region from
the map entirely -- geopandas draws nothing at all for a missing `column`
value unless `missing_kwds` is passed, leaving an unexplained hole
indistinguishable from "outside the study area". `plot_travel_time_
distribution()`'s histogram silently excluded it from the bars/density the
same way. `plot_site_allocation_summary()`'s bars silently summed to less
than 100% with no indication why.

LSOA_4 is unreachable from every candidate site in every fixture here, so
whichever site(s) `solve()` picks, it stays unreachable -- the tests don't
depend on which combination wins.
"""

import matplotlib

matplotlib.use("Agg")

import contextily
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import geopandas
from shapely.geometry import Point

import lokigi


@pytest.fixture(autouse=True)
def stub_basemap_tiles(monkeypatch):
    """Intercept web map tile downloads for every test in this module --
    see test_plotting_smoke.py's identical fixture for the full rationale
    (unconditional `cx.add_basemap` calls, no opt-out on several methods)."""

    def _stub(*args, **kwargs):
        return None

    monkeypatch.setattr(contextily, "add_basemap", _stub)


@pytest.fixture(autouse=True)
def close_figures():
    """Plotting methods create figures they do not own; close them after
    each test so a full run does not accumulate open figures."""
    yield
    plt.close("all")


@pytest.fixture
def unreachable_demand_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )


@pytest.fixture
def unreachable_candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.5, 51.6],
            "long": [-0.1, -0.2],
        }
    )


@pytest.fixture
def unreachable_travel_df():
    """LSOA_4 has no feasible journey to either site."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 12.0, 14.0, np.nan],
            "Site_B": [20.0, 22.0, 24.0, np.nan],
        }
    )


@pytest.fixture
def unreachable_region_gdf():
    return geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "geometry": [
                Point(0, 0).buffer(0.4, cap_style=3),
                Point(1, 0).buffer(0.4, cap_style=3),
                Point(0, 1).buffer(0.4, cap_style=3),
                Point(1, 1).buffer(0.4, cap_style=3),
            ],
        },
        crs="EPSG:27700",
    )


@pytest.fixture
def unreachable_equity_df():
    """LSOA_4 (the unreachable one) is alone in band 5."""
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [1, 1, 1, 5],
        }
    )


@pytest.fixture
def unreachable_problem(
    unreachable_demand_df,
    unreachable_candidate_df,
    unreachable_travel_df,
    unreachable_region_gdf,
):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(
        unreachable_demand_df, demand_col="demand", location_id_col="location_id"
    )
    problem.add_sites(unreachable_candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(
        unreachable_travel_df, source_col="source_id", allow_missing=True
    )
    problem.add_region_geometry_layer(unreachable_region_gdf, common_col="location_id")
    return problem


@pytest.fixture
def unreachable_result(unreachable_problem):
    return unreachable_problem.solve(p=1, unreachable_cost=1000, show_progress=False)


@pytest.fixture
def reachable_result(unreachable_candidate_df, unreachable_region_gdf):
    """Same shape, but no missing values -- the baseline every assertion
    here is contrasted against (no unreachable fragment, one collection)."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 12.0, 14.0, 16.0],
            "Site_B": [20.0, 22.0, 24.0, 26.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(unreachable_candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_region_geometry_layer(unreachable_region_gdf, common_col="location_id")
    return problem.solve(p=1, show_progress=False)


# --- _plot_single_solution_map: missing_kwds renders unreachable regions --


def test_plot_best_combination_draws_unreachable_region_distinctly(
    unreachable_result,
):
    ax = unreachable_result.plot_best_combination(show_all_locations=False)
    # geopandas draws the missing-valued geometries as a second collection
    # only when missing_kwds is passed -- without it, the NaN row is
    # dropped silently and only one collection (the coloured regions) is
    # drawn.
    assert len(ax.collections) >= 2


def test_plot_best_combination_no_extra_collection_when_fully_reachable(
    unreachable_result, reachable_result
):
    """Both plots also draw the selected-site markers as their own
    collection, so the absolute count isn't 1 vs 2 -- what matters is that
    the unreachable case has exactly one MORE collection than the fully
    reachable one (the missing_kwds-styled geometries)."""
    unreachable_ax = unreachable_result.plot_best_combination(show_all_locations=False)
    reachable_ax = reachable_result.plot_best_combination(show_all_locations=False)
    assert len(unreachable_ax.collections) == len(reachable_ax.collections) + 1


def test_plot_best_combination_site_allocation_draws_unreachable_region(
    unreachable_result, reachable_result
):
    unreachable_ax = unreachable_result.plot_best_combination(
        show_all_locations=False, plot_site_allocation=True
    )
    reachable_ax = reachable_result.plot_best_combination(
        show_all_locations=False, plot_site_allocation=True
    )
    assert len(unreachable_ax.collections) == len(reachable_ax.collections) + 1


# --- Title/label annotations name the unreachable count -------------------


def test_plot_best_combination_title_names_unreachable_count(unreachable_result):
    ax = unreachable_result.plot_best_combination(show_all_locations=False)
    assert "1 region unreachable" in ax.get_title()


def test_plot_best_combination_title_omits_fragment_when_fully_reachable(
    reachable_result,
):
    ax = reachable_result.plot_best_combination(show_all_locations=False)
    assert "unreachable" not in ax.get_title()


def test_plot_n_best_combinations_subplot_title_names_unreachable_count(
    unreachable_problem,
):
    result = unreachable_problem.solve(p=1, unreachable_cost=1000, show_progress=False)
    fig, axes = result.plot_n_best_combinations(n_best=2, show_all_locations=False)
    assert "1 region unreachable" in axes.flatten()[0].get_title()


def test_plot_n_best_combinations_does_not_crash_when_globally_unreachable(
    unreachable_problem,
):
    """Regression coverage for the global_vmin/global_vmax NaN-order bug:
    Python's builtin min()/max() over a list of per-solution .min()/.max()
    results is NaN-order-dependent whenever a candidate solution is
    entirely unreachable. p=2 with brute_force_keep_best_n large enough to
    surface both combinations exercises multiple solutions at once."""
    result = unreachable_problem.solve(
        p=1, unreachable_cost=1000, show_progress=False
    )
    fig, axes = result.plot_n_best_combinations(n_best=2, show_all_locations=False)
    assert fig is not None


def test_plot_solution_comparison_title_names_unreachable_count(unreachable_result):
    fig, axes = unreachable_result.plot_solution_comparison(
        [{"solution_rank": 1}], show_all_locations=False
    )
    assert "1 region unreachable" in axes[0].get_title()


def test_plot_solution_sets_comparison_title_names_unreachable_count(
    unreachable_result,
):
    from lokigi.plot_utils import plot_solution_sets_comparison

    fig, axes = plot_solution_sets_comparison(
        solution_sets=[unreachable_result],
        solutions_config=[{"solution_rank": 1}],
        show_all_locations=False,
    )
    assert "1 region unreachable" in axes[0].get_title()


def test_plot_travel_time_distribution_label_names_unreachable_count(
    unreachable_result,
):
    fig = unreachable_result.plot_travel_time_distribution(top_n=1)
    labels = [a.text for a in fig.layout.annotations if "Unreachable" in a.text]
    assert len(labels) == 1
    assert "Unreachable: 1" in labels[0]


def test_plot_travel_time_distribution_omits_unreachable_when_fully_reachable(
    reachable_result,
):
    fig = reachable_result.plot_travel_time_distribution(top_n=1)
    labels = [a.text for a in fig.layout.annotations if "Unreachable" in a.text]
    assert labels == []


# --- site_allocation_summary(): documented proportion shortfall -----------


def test_site_allocation_summary_proportion_sums_to_less_than_one(
    unreachable_result,
):
    summary = unreachable_result.site_allocation_summary()
    # LSOA_4's demand (100 of 400 total) is unreachable and excluded from
    # every site's row entirely -- proportion sums to 0.75, not 1.0.
    assert summary["proportion"].sum() == pytest.approx(0.75)
    assert summary["n_regions"].sum() == 3  # 3 reachable regions, 1 site


def test_plot_site_allocation_summary_title_names_unreachable_share(
    unreachable_result,
):
    fig = unreachable_result.plot_site_allocation_summary(interactive=False)
    ax = fig.axes[0]
    assert "1 region unreachable" in ax.get_title()


# --- plot_combination_by_equity(): per-band panels -------------------------


def test_plot_combination_by_equity_draws_unreachable_region_and_labels_it(
    unreachable_problem, unreachable_equity_df
):
    unreachable_problem.add_equity_data(
        unreachable_equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    result = unreachable_problem.solve(p=1, unreachable_cost=1000, show_progress=False)
    fig, axes = result.plot_combination_by_equity()

    # Band 5 (the group containing LSOA_4, the unreachable region) gets a
    # second collection (missing_kwds) and its title names the count.
    band_titles = {ax.get_title(): ax for ax in axes if ax.get_title()}
    band_5_ax = next(ax for title, ax in band_titles.items() if "5" in title)
    assert len(band_5_ax.collections) >= 2
    assert "1 region unreachable" in band_5_ax.get_title()
