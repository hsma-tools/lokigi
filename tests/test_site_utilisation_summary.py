"""Tests for `SiteProblem.site_utilisation_summary()` /
`SiteProblem.plot_site_utilisation()`.

Unlike `site_allocation_summary()` (solve-derived) or
`two_step_floating_catchment()` (catchment-derived), this is a baseline
diagnostic: it reports today's real-world utilisation exactly as registered
via `add_sites(current_load_col=..., capacity_col=...)` (raw counts) or
`add_sites(utilisation_col=...)` (a precomputed ratio), with no dependency on
demand data, a travel matrix, or `solve()`.

Fixture arithmetic used below:

- `raw_counts_problem`: Site_A cap=100/load=80 -> ratio 0.8, headroom 20.
  Site_B cap=200/load=200 -> ratio 1.0, headroom 0 (exactly full).
  Site_C cap=150/load=180 -> ratio 1.2, headroom -30 (genuinely over
  capacity -- deliberately not clipped). Site_D cap=NaN/load=NaN -> a
  simulated not-yet-built site with no baseline data at all.
- `precomputed_ratio_problem`: same four sites, registered instead with a
  single precomputed `utilisation` ratio column and no raw counts.
"""

import pandas as pd
import pytest
import geopandas
from shapely.geometry import Point

import lokigi


@pytest.fixture
def raw_counts_candidate_gdf():
    return geopandas.GeoDataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "capacity": [100.0, 200.0, 150.0, float("nan")],
            "current_load": [80.0, 200.0, 180.0, float("nan")],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        crs="EPSG:27700",
    )


@pytest.fixture
def raw_counts_problem(raw_counts_candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(
        raw_counts_candidate_gdf,
        candidate_id_col="site_id",
        capacity_col="capacity",
        current_load_col="current_load",
    )
    return problem


@pytest.fixture
def precomputed_ratio_candidate_gdf():
    return geopandas.GeoDataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C", "Site_D"],
            "capacity": [100.0, 200.0, 150.0, float("nan")],
            "utilisation": [0.8, 1.0, 1.2, float("nan")],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        },
        crs="EPSG:27700",
    )


@pytest.fixture
def precomputed_ratio_problem(precomputed_ratio_candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(
        precomputed_ratio_candidate_gdf,
        candidate_id_col="site_id",
        capacity_col="capacity",
        utilisation_col="utilisation",
    )
    return problem


@pytest.fixture
def precomputed_ratio_no_capacity_problem(precomputed_ratio_candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(
        precomputed_ratio_candidate_gdf.drop(columns=["capacity"]),
        candidate_id_col="site_id",
        utilisation_col="utilisation",
    )
    return problem


# ---------------------------------------------------------------------------
# add_sites() validation
# ---------------------------------------------------------------------------


def test_both_current_load_and_utilisation_col_raises(raw_counts_candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    with pytest.raises(ValueError, match="at most one"):
        problem.add_sites(
            raw_counts_candidate_gdf,
            candidate_id_col="site_id",
            capacity_col="capacity",
            current_load_col="current_load",
            utilisation_col="current_load",
        )


def test_current_load_col_without_capacity_col_raises(raw_counts_candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    with pytest.raises(ValueError, match="capacity_col"):
        problem.add_sites(
            raw_counts_candidate_gdf,
            candidate_id_col="site_id",
            current_load_col="current_load",
        )


def test_negative_current_load_raises(raw_counts_candidate_gdf):
    bad_gdf = raw_counts_candidate_gdf.copy()
    bad_gdf.loc[0, "current_load"] = -5.0
    problem = lokigi.site.SiteProblem(debug_mode=False)
    with pytest.raises(ValueError, match="Site_A"):
        problem.add_sites(
            bad_gdf,
            candidate_id_col="site_id",
            capacity_col="capacity",
            current_load_col="current_load",
        )


def test_negative_utilisation_raises(precomputed_ratio_candidate_gdf):
    bad_gdf = precomputed_ratio_candidate_gdf.copy()
    bad_gdf.loc[1, "utilisation"] = -0.1
    problem = lokigi.site.SiteProblem(debug_mode=False)
    with pytest.raises(ValueError, match="Site_B"):
        problem.add_sites(
            bad_gdf, candidate_id_col="site_id", utilisation_col="utilisation"
        )


def test_non_numeric_capacity_col_raises(raw_counts_candidate_gdf):
    """Regression guard: capacity_col was never included in add_sites()'s
    numeric validation, so a non-numeric capacity silently passed through.
    Reverting the numeric_col_names fix and rerunning this test confirms it
    actually catches the regression (fails without the fix, passes with
    it)."""
    bad_gdf = raw_counts_candidate_gdf.copy()
    bad_gdf["capacity"] = bad_gdf["capacity"].astype(str)
    problem = lokigi.site.SiteProblem(debug_mode=False)
    with pytest.raises(TypeError, match="capacity"):
        problem.add_sites(
            bad_gdf, candidate_id_col="site_id", capacity_col="capacity"
        )


# ---------------------------------------------------------------------------
# site_utilisation_summary() -- raw-counts path
# ---------------------------------------------------------------------------


def test_raw_counts_path_computes_ratio_and_headroom(raw_counts_problem):
    summary = raw_counts_problem.site_utilisation_summary(
        site_names=["Site_A", "Site_B", "Site_C"]
    )

    assert list(summary.index) == ["Site_A", "Site_B", "Site_C"]
    assert summary.index.name == "site"
    assert summary.loc["Site_A", "capacity"] == 100.0
    assert summary.loc["Site_A", "current_load"] == 80.0
    assert summary.loc["Site_A", "utilisation_ratio"] == pytest.approx(0.8)
    assert summary.loc["Site_A", "headroom"] == pytest.approx(20.0)

    assert summary.loc["Site_B", "utilisation_ratio"] == pytest.approx(1.0)
    assert summary.loc["Site_B", "headroom"] == pytest.approx(0.0)


def test_over_capacity_site_is_not_clipped(raw_counts_problem):
    """A ratio above 1.0 and a negative headroom are themselves the finding
    -- they must not be silently clamped to a "full" reading."""
    summary = raw_counts_problem.site_utilisation_summary(site_names=["Site_C"])

    assert summary.loc["Site_C", "utilisation_ratio"] == pytest.approx(1.2)
    assert summary.loc["Site_C", "headroom"] == pytest.approx(-30.0)


def test_site_with_no_baseline_data_is_nan_not_zero(raw_counts_problem):
    """A not-yet-built site (no capacity/current_load at all) must appear as
    an explicit NaN row, not be dropped or coerced to 0.0 -- 0.0 would
    misleadingly read as "measured, and currently idle"."""
    summary = raw_counts_problem.site_utilisation_summary()

    assert len(summary) == 4
    assert "Site_D" in summary.index
    assert pd.isna(summary.loc["Site_D", "utilisation_ratio"])
    assert pd.isna(summary.loc["Site_D", "headroom"])


def test_default_selection_is_every_site_in_canonical_order(raw_counts_problem):
    summary = raw_counts_problem.site_utilisation_summary()
    assert list(summary.index) == ["Site_A", "Site_B", "Site_C", "Site_D"]


# ---------------------------------------------------------------------------
# site_utilisation_summary() -- precomputed-ratio path
# ---------------------------------------------------------------------------


def test_precomputed_ratio_path_uses_values_as_is(precomputed_ratio_problem):
    summary = precomputed_ratio_problem.site_utilisation_summary(
        site_names=["Site_A", "Site_B", "Site_C"]
    )

    assert "current_load" not in summary.columns
    assert summary.loc["Site_A", "utilisation_ratio"] == pytest.approx(0.8)
    assert summary.loc["Site_C", "utilisation_ratio"] == pytest.approx(1.2)


def test_precomputed_ratio_with_capacity_derives_headroom(precomputed_ratio_problem):
    summary = precomputed_ratio_problem.site_utilisation_summary(
        site_names=["Site_A", "Site_C"]
    )

    assert "headroom" in summary.columns
    assert summary.loc["Site_A", "headroom"] == pytest.approx(
        100.0 * (1 - 0.8)
    )
    assert summary.loc["Site_C", "headroom"] == pytest.approx(150.0 * (1 - 1.2))


def test_precomputed_ratio_without_capacity_has_no_headroom_or_capacity(
    precomputed_ratio_no_capacity_problem,
):
    summary = precomputed_ratio_no_capacity_problem.site_utilisation_summary(
        site_names=["Site_A", "Site_B"]
    )

    assert "capacity" not in summary.columns
    assert "headroom" not in summary.columns
    assert "current_load" not in summary.columns
    assert list(summary.columns) == ["utilisation_ratio"]


# ---------------------------------------------------------------------------
# Call-time errors when no baseline data was registered
# ---------------------------------------------------------------------------


def test_no_utilisation_data_registered_raises(candidate_gdf):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_gdf, candidate_id_col="site_id")

    with pytest.raises(ValueError, match="add_sites"):
        problem.site_utilisation_summary()


def test_capacity_only_registered_raises(candidate_gdf):
    """capacity_col alone (e.g. registered for a different purpose, like
    solve()'s future capacitated mode) is not enough on its own -- there is
    no current load or precomputed ratio to report."""
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_gdf, candidate_id_col="site_id", capacity_col="capacity")

    with pytest.raises(ValueError, match="add_sites"):
        problem.site_utilisation_summary()


# ---------------------------------------------------------------------------
# Selection arguments
# ---------------------------------------------------------------------------


def test_site_names_and_site_indices_mutually_exclusive(raw_counts_problem):
    with pytest.raises(ValueError, match="at most one"):
        raw_counts_problem.site_utilisation_summary(
            site_names=["Site_A"], site_indices=[0]
        )


def test_site_indices_resolve_the_same_as_site_names(raw_counts_problem):
    by_names = raw_counts_problem.site_utilisation_summary(
        site_names=["Site_A", "Site_B"]
    )
    by_indices = raw_counts_problem.site_utilisation_summary(site_indices=[0, 1])

    pd.testing.assert_frame_equal(by_names, by_indices)


def test_out_of_range_site_indices_raises(raw_counts_problem):
    with pytest.raises(IndexError):
        raw_counts_problem.site_utilisation_summary(site_indices=[99])


def test_unknown_site_names_raises(raw_counts_problem):
    with pytest.raises(KeyError):
        raw_counts_problem.site_utilisation_summary(site_names=["Nonexistent_Site"])


def test_duplicate_site_indices_raises(raw_counts_problem):
    with pytest.raises(ValueError, match="duplicate"):
        raw_counts_problem.site_utilisation_summary(site_indices=[0, 0])


def test_duplicate_site_names_raises(raw_counts_problem):
    with pytest.raises(ValueError, match="duplicate"):
        raw_counts_problem.site_utilisation_summary(
            site_names=["Site_A", "Site_A"]
        )


def test_site_names_preserve_given_order_not_canonical_order(raw_counts_problem):
    summary = raw_counts_problem.site_utilisation_summary(
        site_names=["Site_C", "Site_A"]
    )
    assert list(summary.index) == ["Site_C", "Site_A"]


# ---------------------------------------------------------------------------
# plot_site_utilisation()
# ---------------------------------------------------------------------------


def test_plot_requires_real_geometry():
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(
        pd.DataFrame({"location_id": ["L1"], "demand": [1]}),
        demand_col="demand",
        location_id_col="location_id",
    )
    problem.add_travel_matrix(
        pd.DataFrame({"location_id": ["L1"], "Site_A": [1.0]}),
        source_col="location_id",
    )
    problem._setup_sites_df_from_travel_matrix()

    with pytest.raises(ValueError, match="geometry"):
        problem.plot_site_utilisation()


def test_plot_static_returns_axes(raw_counts_problem):
    import matplotlib

    ax = raw_counts_problem.plot_site_utilisation(add_basemap=False)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_accepts_precomputed_utilisation_df(raw_counts_problem):
    import matplotlib

    precomputed = raw_counts_problem.site_utilisation_summary()
    ax = raw_counts_problem.plot_site_utilisation(
        utilisation_df=precomputed, add_basemap=False
    )
    assert isinstance(ax, matplotlib.axes.Axes)
