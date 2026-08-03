"""Tests for `plot_accessibility()` (`SiteProblem` and `SiteSolutionSet`).

Complements the generic run/render/basemap smoke coverage in
`test_plotting_smoke.py` with a few behaviours specific to this method:

- Site markers are drawn from `candidate_sites`' actual geometry, not the
  `_candidate_sites_type` string `add_sites()` stores -- that string
  records the *input* format ("pandas" for tabular lat/long input), even
  though tabular input is converted to a real GeoDataFrame internally.
  Trusting it directly would silently skip site markers for the most
  common `add_sites()` usage pattern (this was caught and fixed while
  building this method -- `plot_best_combination()` has the same
  underlying gap, out of scope here and reported separately).
- A site with an undefined (NaN) ratio needs an explicit legend entry,
  since geopandas' `missing_kwds` label only surfaces on a discrete
  `scheme=` legend, not the continuous colorbar this method uses.
- `region_frame`/`site_frame` passed in directly should be used as-is,
  without a redundant call to `two_step_floating_catchment()`.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_raises_without_region_geometry_layer(sfca_problem):
    with pytest.raises(ValueError, match="add_region_geometry_layer"):
        sfca_problem.plot_accessibility(supply_col="supply", catchment_size=15)


def test_raises_without_frames_or_params(sfca_problem_with_geometry):
    with pytest.raises(ValueError, match="supply_col"):
        sfca_problem_with_geometry.plot_accessibility()


def test_site_markers_use_actual_geometry_not_candidate_sites_type(
    sfca_problem_with_geometry,
):
    """Regression: `_candidate_sites_type` stays "pandas" for tabular
    lat/long input (as `sfca_problem`'s candidate_df is) even though
    `add_sites()` converts it to a real GeoDataFrame. Forcing
    `candidate_sites` to a plain DataFrame (stripping only its
    GeoDataFrame-ness, not its data) must suppress site markers; leaving
    it alone must draw them."""
    ax_with_geometry = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        add_basemap=False,
    )
    n_collections_with_geometry = len(ax_with_geometry.collections)
    assert n_collections_with_geometry >= 2  # region choropleth + site markers

    real_candidate_sites = sfca_problem_with_geometry.candidate_sites
    sfca_problem_with_geometry.candidate_sites = pd.DataFrame(real_candidate_sites)
    try:
        ax_without_geometry = sfca_problem_with_geometry.plot_accessibility(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2"],
            add_basemap=False,
        )
    finally:
        sfca_problem_with_geometry.candidate_sites = real_candidate_sites

    assert len(ax_without_geometry.collections) < n_collections_with_geometry


def test_site_marker_count_matches_scored_sites(sfca_problem_with_geometry):
    ax = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        add_basemap=False,
    )
    site_markers = ax.collections[-1]
    assert len(site_markers.get_offsets()) == 2


def test_nan_ratio_site_gets_explicit_legend_entry(sfca_problem_with_geometry):
    with pytest.warns(UserWarning, match="Site_Isolated"):
        ax = sfca_problem_with_geometry.plot_accessibility(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2", "Site_Isolated"],
            show_site_ratio=True,
            add_basemap=False,
        )
    legend = ax.get_legend()
    assert legend is not None
    assert "No catchment demand" in [t.get_text() for t in legend.get_texts()]


def test_no_legend_when_every_site_has_a_ratio(sfca_problem_with_geometry):
    """geopandas' `.plot(legend=True)` draws a colorbar, not an
    `ax.legend()` -- so with no NaN-ratio site to explain, `get_legend()`
    should stay None rather than picking up a leftover/empty legend."""
    ax = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        show_site_ratio=True,
        add_basemap=False,
    )
    assert ax.get_legend() is None


def test_default_site_markers_are_plain_and_uniform(sfca_problem_with_geometry):
    """`show_site_ratio` defaults to False: site markers should be plain,
    uniformly coloured dots -- no ratio-driven colour/size, and no
    "No catchment demand" legend even for a site with a NaN ratio (it's
    still warned about by `two_step_floating_catchment()`, since that
    warning is unrelated to how the map chooses to render it)."""
    with pytest.warns(UserWarning, match="Site_Isolated"):
        ax = sfca_problem_with_geometry.plot_accessibility(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2", "Site_Isolated"],
            add_basemap=False,
        )
    site_markers = ax.collections[-1]
    assert len(site_markers.get_offsets()) == 3
    # A single shared facecolor (rather than one per point, as the
    # ratio-cmap path produces) is itself evidence markers are uniformly
    # coloured, not driven by `ratio`.
    assert len(site_markers.get_facecolor()) == 1
    assert ax.get_legend() is None


def test_site_colour_overrides_default_marker_colour(sfca_problem_with_geometry):
    import matplotlib.colors as mcolors

    ax = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        site_colour="red",
        add_basemap=False,
    )
    site_markers = ax.collections[-1]
    facecolors = site_markers.get_facecolor()
    assert (facecolors[0] == mcolors.to_rgba("red")).all()


def test_caption_reflects_show_site_ratio(sfca_problem_with_geometry):
    ax_default = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        add_basemap=False,
    )
    ax_ratio = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        show_site_ratio=True,
        add_basemap=False,
    )
    # Caption text is line-wrapped for display; collapse back to a single
    # line so substring checks aren't sensitive to where the wrap falls.
    default_caption = " ".join(
        ax_default.get_figure().texts[-1].get_text().split()
    )
    ratio_caption = " ".join(ax_ratio.get_figure().texts[-1].get_text().split())
    assert "plain markers" in default_caption
    assert "plain markers" not in ratio_caption
    assert "coloured and sized by its own supply" in ratio_caption
    assert "coloured and sized by its own supply" not in default_caption


def test_precomputed_frames_skip_autocompute(sfca_problem_with_geometry, monkeypatch):
    region_frame, site_frame = sfca_problem_with_geometry.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "two_step_floating_catchment should not be called when "
            "region_frame/site_frame are both supplied"
        )

    monkeypatch.setattr(
        sfca_problem_with_geometry, "two_step_floating_catchment", _fail_if_called
    )

    ax = sfca_problem_with_geometry.plot_accessibility(
        region_frame=region_frame, site_frame=site_frame, add_basemap=False
    )
    assert ax is not None


def test_site_problem_and_solution_set_plot_agree_on_marker_count(
    sfca_problem_with_geometry,
):
    result = sfca_problem_with_geometry.solve(p=2, show_progress=False)

    ax = result.plot_accessibility(
        supply_col="supply", catchment_size=15, add_basemap=False
    )
    site_markers = ax.collections[-1]
    assert len(site_markers.get_offsets()) == 2


def test_distance_decay_is_forwarded_and_differs_from_catchment_size(
    sfca_problem_with_geometry,
):
    """`plot_accessibility(distance_decay=...)` auto-computes via the
    step-decay/Gaussian path rather than raising or silently falling back
    to a hard cutoff."""
    ax = sfca_problem_with_geometry.plot_accessibility(
        supply_col="supply",
        distance_decay=[(10, 1.0), (20, 0.5), (30, 0.2)],
        site_names=["Site_1", "Site_2"],
        add_basemap=False,
    )
    assert ax is not None

    region_catchment_size = sfca_problem_with_geometry.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    region_distance_decay = sfca_problem_with_geometry.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=[(10, 1.0), (20, 0.5), (30, 0.2)],
        site_names=["Site_1", "Site_2"],
    )
    assert not region_catchment_size["accessibility"].equals(
        region_distance_decay["accessibility"]
    )
