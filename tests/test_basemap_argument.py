"""
Tests for the `add_basemap` / `show_basemap` argument unification.

`add_basemap` is the canonical name on every plotting method. Three
methods -- plot_region_geometry_layer, plot_hotspots, plot_quadrant_map --
previously spelled it `show_basemap`, and because they also accept
`**kwargs` and forward them to the underlying plotting call, passing
`add_basemap=False` never reached code that understood it:

  - static path: ``AttributeError: PatchCollection.set() got an
    unexpected keyword argument 'add_basemap'`` -- an error blaming
    matplotlib internals that never mentions the real problem, that this
    method spells the argument differently;
  - interactive path: silently ignored, because
    ``GeoDataFrame.explore()`` accepts unknown keywords, so the tile
    layer was drawn anyway with no signal to the caller.

`show_basemap` still works for one release, emitting a FutureWarning,
following the same deprecation shape as the `direction` ->
`disadvantaged_end` alias on `add_equity_data()`.
"""

import matplotlib

matplotlib.use("Agg")

import contextily
import matplotlib.pyplot as plt
import pytest

from lokigi.utils import _resolve_basemap_argument

from tests.test_plotting_smoke import plottable_problem  # noqa: F401

# The three methods that carried the deprecated spelling.
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


# --- the resolver in isolation --------------------------------------------


def test_defaults_to_true_when_neither_given():
    assert _resolve_basemap_argument(None, None, "plot_x") is True


@pytest.mark.parametrize("value", [True, False])
def test_canonical_argument_is_passed_through(value):
    assert _resolve_basemap_argument(value, None, "plot_x") is value


@pytest.mark.parametrize("value", [True, False])
def test_deprecated_alias_is_honoured_with_a_warning(value):
    with pytest.warns(FutureWarning, match="'show_basemap'.*deprecated"):
        assert _resolve_basemap_argument(None, value, "plot_x") is value


def test_passing_both_is_rejected_as_ambiguous():
    with pytest.raises(ValueError, match="passing both is ambiguous"):
        _resolve_basemap_argument(True, False, "plot_x")


def test_error_and_warning_name_the_calling_method():
    """The messages have to say which method was misused -- these arguments
    span five different plotting methods."""
    with pytest.raises(ValueError, match="plot_quadrant_map"):
        _resolve_basemap_argument(True, True, "plot_quadrant_map")

    with pytest.warns(FutureWarning, match="plot_hotspots"):
        _resolve_basemap_argument(None, False, "plot_hotspots")


# --- end to end, through the real methods ---------------------------------


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_add_basemap_false_is_honoured_not_forwarded_to_kwargs(
    plottable_problem, stub_basemap_tiles, method_name  # noqa: F811
):
    """The regression this rename exists to prevent: `add_basemap=False`
    landing in **kwargs, where it either crashed the static path with a
    matplotlib error or was quietly ignored on the interactive one."""
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
def test_deprecated_show_basemap_still_works_on_every_renamed_method(
    plottable_problem, stub_basemap_tiles, method_name  # noqa: F811
):
    with pytest.warns(FutureWarning, match=method_name):
        getattr(plottable_problem, method_name)(show_basemap=False)

    assert stub_basemap_tiles == [], (
        f"{method_name}(show_basemap=False) still requested a basemap"
    )


@pytest.mark.parametrize("method_name", RENAMED_METHODS)
def test_passing_both_names_raises_on_every_renamed_method(
    plottable_problem, method_name  # noqa: F811
):
    with pytest.raises(ValueError, match="passing both is ambiguous"):
        getattr(plottable_problem, method_name)(
            add_basemap=False, show_basemap=False
        )


def test_methods_that_always_used_add_basemap_are_unchanged(
    plottable_problem, stub_basemap_tiles  # noqa: F811
):
    """plot_sites never had the alias, so it should accept add_basemap
    without warning and reject show_basemap outright (it has no **kwargs)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plottable_problem.plot_sites(add_basemap=False)

    assert stub_basemap_tiles == []

    with pytest.raises(TypeError):
        plottable_problem.plot_sites(show_basemap=False)
