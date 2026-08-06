"""
Tests for the `add_basemap` argument, unified across every plotting method
in v0.7.0.

Three methods -- plot_region_geometry_layer, plot_hotspots and
plot_quadrant_map -- previously spelled it `show_basemap`. Because they
also accept `**kwargs` and forward them to the underlying plotting call,
the wrong spelling never reached code that understood it:

  - static path: ``AttributeError: PatchCollection.set() got an
    unexpected keyword argument`` -- an error blaming matplotlib
    internals that never mentions the real problem, that this method
    spells the argument differently;
  - interactive path: silently ignored, because
    ``GeoDataFrame.explore()`` accepts unknown keywords, so the tile
    layer was drawn anyway with no signal to the caller.

`show_basemap` has been removed outright rather than aliased (lokigi is
pre-1.0). That same `**kwargs` forwarding means a bare removal would give
old callers the identical cryptic failure, so the three methods explicitly
reject the removed name with a message naming the replacement.

plot_sites picked up the same guard later for a different reason: it never
carried the old spelling, but once it gained **kwargs forwarding of its own
it was exposed to the identical silent-absorption failure, so it rejects
the removed name too. See test_plot_sites_rejects_the_removed_show_basemap_kwarg.
"""

import matplotlib

matplotlib.use("Agg")

import contextily
import matplotlib.pyplot as plt
import pytest

from lokigi.utils import _reject_removed_basemap_alias

from tests.test_plotting_smoke import plottable_problem  # noqa: F401

# The three methods that carried the removed spelling.
RENAMED_METHODS = [
    "plot_region_geometry_layer",
    "plot_hotspots",
    "plot_quadrant_map",
]


@pytest.fixture(autouse=True)
def stub_basemap_tiles(monkeypatch):
    """Record basemap requests without downloading anything. See
    tests/test_plotting_smoke.py::stub_basemap_tiles for the rationale."""
    attempts = []
    monkeypatch.setattr(
        contextily, "add_basemap", lambda *a, **k: attempts.append(k.get("crs"))
    )
    return attempts


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- the guard in isolation -----------------------------------------------


def test_guard_passes_when_the_removed_name_is_absent():
    assert _reject_removed_basemap_alias({"cmap": "Blues"}, "plot_x") is None


def test_guard_rejects_the_removed_name():
    with pytest.raises(TypeError, match="no longer accepts 'show_basemap'"):
        _reject_removed_basemap_alias({"show_basemap": False}, "plot_x")


def test_guard_message_names_the_method_and_the_replacement():
    """The caller needs to know which call to fix and what to write instead."""
    with pytest.raises(TypeError) as excinfo:
        _reject_removed_basemap_alias({"show_basemap": True}, "plot_quadrant_map")

    message = str(excinfo.value)
    assert "plot_quadrant_map" in message
    assert "add_basemap" in message


# --- end to end, through the real methods ---------------------------------


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_add_basemap_false_suppresses_the_download(
    plottable_problem, stub_basemap_tiles, method_name  # noqa: F811
):
    getattr(plottable_problem, method_name)(add_basemap=False)

    assert stub_basemap_tiles == [], (
        f"{method_name}(add_basemap=False) still requested a basemap"
    )


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_add_basemap_true_still_requests_tiles(
    plottable_problem, stub_basemap_tiles, method_name  # noqa: F811
):
    """The mirror case -- the flag must not be stuck off."""
    getattr(plottable_problem, method_name)(add_basemap=True)

    assert stub_basemap_tiles, (
        f"{method_name}(add_basemap=True) did not request a basemap"
    )


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_add_basemap_defaults_to_true(
    plottable_problem, stub_basemap_tiles, method_name  # noqa: F811
):
    getattr(plottable_problem, method_name)()

    assert stub_basemap_tiles, f"{method_name}() did not default to a basemap"


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_removed_show_basemap_raises_a_useful_error(
    plottable_problem, method_name  # noqa: F811
):
    """Without the explicit guard this would reach matplotlib and surface as
    `PatchCollection.set() got an unexpected keyword argument`, which names
    neither the method nor the replacement -- or be silently ignored on the
    interactive path.
    """
    with pytest.raises(TypeError, match="no longer accepts 'show_basemap'"):
        getattr(plottable_problem, method_name)(show_basemap=False)


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_removed_name_is_rejected_before_any_analysis_runs(
    plottable_problem, method_name  # noqa: F811
):
    """The hotspot methods run a spatial analysis before plotting. Rejecting
    the removed argument first means a bad call fails immediately instead of
    after that work.
    """
    called = []
    original = type(plottable_problem).get_hotspots

    def _spy(self, *args, **kwargs):
        called.append(1)
        return original(self, *args, **kwargs)

    type(plottable_problem).get_hotspots = _spy
    try:
        with pytest.raises(TypeError):
            getattr(plottable_problem, method_name)(show_basemap=False)
    finally:
        type(plottable_problem).get_hotspots = original

    assert called == [], f"{method_name} ran analysis before rejecting the call"


def test_plot_sites_always_used_add_basemap(
    plottable_problem, stub_basemap_tiles  # noqa: F811
):
    """plot_sites always spelled it add_basemap."""
    plottable_problem.plot_sites(add_basemap=False)
    assert stub_basemap_tiles == []


def test_plot_sites_rejects_the_removed_show_basemap_kwarg(
    plottable_problem  # noqa: F811
):
    """plot_sites now forwards **kwargs to the underlying plot/explore call
    (see plot_region_geometry_layer, plot_hotspots, plot_quadrant_map for
    the same problem), so without the guard `show_basemap` would be
    silently absorbed and forwarded to matplotlib, surfacing there as
    `PathCollection.set() got an unexpected keyword argument` instead of a
    message naming the real problem."""
    with pytest.raises(TypeError, match="no longer accepts 'show_basemap'"):
        plottable_problem.plot_sites(show_basemap=False)
