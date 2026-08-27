"""Tests for `SiteSolutionSet._resolve_equity_view()` -- the single entry
point behind the `equity=` selector on the reporting methods.

Two paths reach the same shape: the primary dimension's bands are read
straight off `solution_df` (the numbers the solve stored), any other
dimension's are recomputed from that solution's `problem_df`. The
equivalence tests here are what keep them honest -- a recompute that
disagrees with the stored value would surface as a chart quietly
contradicting the number printed beside it.
"""

import math

import pytest
import pandas as pd

import lokigi
from lokigi.site_solutions import _EQUITY_BAND_STORED_NAMES


def _same(a, b):
    """Equality that treats NaN as equal to NaN.

    Several band values are legitimately NaN -- every coverage entry is,
    whenever no `threshold_for_coverage` was given -- and plain `==` would
    report two structurally identical breakdowns as different. Comparing
    the stored dict to itself happens to pass by object identity, which
    would make these assertions weaker than they look; this compares by
    value instead."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


@pytest.fixture
def two_dimension_problem(loaded_problem_with_equity):
    """`loaded_problem_with_equity` (IMD deciles 1/5/10, disadvantaged_end
    "low") plus a second dimension banded so a *different* region is the
    most disadvantaged one -- LSOA_3 rather than LSOA_1.

    It takes both a band-value change and a `disadvantaged_end` flip to
    achieve that: mirroring the values while also flipping the end would
    cancel out and leave the two dimensions describing the same ordering,
    which would make every assertion below pass vacuously."""
    loaded_problem_with_equity.add_equity_data(
        pd.DataFrame(
            {
                "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
                "rurality_band": [1, 5, 10],
            }
        ),
        equity_col="rurality_band",
        common_col="location_id",
        label="rurality",
        disadvantaged_end="high",
        primary=False,
    )
    return loaded_problem_with_equity


@pytest.fixture
def multi_region_band_problem():
    """A problem where equity band 1 holds TWO regions with different
    demand, which `loaded_problem_with_equity` does not.

    That distinction is load-bearing rather than incidental. With one
    region per band, a band's weighted mean collapses to that single
    region's cost whatever the weights are -- so any test of *how* the
    bands are weighted passes no matter what the implementation does. This
    fixture was added after exactly that: the compound-weights test below
    kept passing with `_row_weight` disabled entirely.

    Hand-computed, min_cost = [10, 30, 20, 40] for L1..L4 (Site_X always
    wins -- Site_Y is 100 everywhere), bands 1/1/5/10, demand 300/100/100/100:

      demand-weighted     band 1 = (10*300 + 30*100) / 400 = 15.0
      weights={"demand": 0.6, "equity": 0.4}
                          row weights = [1.0, 0.4, 0.2222, 0.0]
                          band 1 = (10*1.0 + 30*0.4) / 1.4 = 15.71

    Bands 5 and 10 hold one region each, so they read 20.0 and 40.0 under
    both weightings -- band 1 is the whole test.
    """
    ids = ["L1", "L2", "L3", "L4"]
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(
        pd.DataFrame({"location_id": ids, "demand": [300, 100, 100, 100]}),
        demand_col="demand",
        location_id_col="location_id",
    )
    problem.add_sites(
        pd.DataFrame(
            {
                "site_id": ["Site_X", "Site_Y"],
                "lat": [51.5, 51.6],
                "long": [-0.1, -0.2],
            }
        ),
        candidate_id_col="site_id",
    )
    problem.add_travel_matrix(
        pd.DataFrame(
            {
                "source_id": ids,
                "Site_X": [10.0, 30.0, 20.0, 40.0],
                "Site_Y": [100.0] * 4,
            }
        ),
        source_col="source_id",
    )
    problem.add_equity_data(
        pd.DataFrame({"location_id": ids, "imd_decile": [1, 1, 5, 10]}),
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    return problem


# --- Equivalence: the recompute must reproduce what the solve stored -------


def test_primary_view_matches_stored_solution_df_values(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    row = result.solution_df.iloc[0]
    meta, bands = result._resolve_equity_view()

    assert meta["label"] == "equity"
    for key, stored_name in _EQUITY_BAND_STORED_NAMES.items():
        assert _same(bands[key], row[stored_name]), (
            f"{key} diverged from {stored_name}"
        )


def test_naming_the_primary_dimension_explicitly_is_identical(
    loaded_problem_with_equity,
):
    """equity="equity" and equity=None must not take observably different
    routes -- both describe the primary dimension."""
    result = loaded_problem_with_equity.solve(p=1)
    _, implicit = result._resolve_equity_view()
    _, explicit = result._resolve_equity_view(equity="equity")
    assert implicit.keys() == explicit.keys()
    assert all(_same(implicit[k], explicit[k]) for k in implicit)


def test_recompute_reproduces_the_stored_primary_values(multi_region_band_problem):
    """The load-bearing one. Forces the recompute path for the *primary*
    dimension by temporarily clearing the registry's primary flag, then
    checks it lands on exactly the numbers the solve stored. This is what
    proves `_equity_band_metrics()` behaves identically whether it is
    called mid-solve or afterwards against problem_df."""
    result = multi_region_band_problem.solve(p=1)
    row = result.solution_df.iloc[0]
    # Band 1 must be a genuine weighted mean of two regions, not one
    # region's cost wearing a weighted mean's clothing.
    assert row["weighted_by_equity_group"][1] == pytest.approx(15.0)

    result.site_problem._primary_equity_label = None
    _, recomputed = result._resolve_equity_view(equity="equity")

    for key, stored_name in _EQUITY_BAND_STORED_NAMES.items():
        assert _same(recomputed[key], row[stored_name]), (
            f"recomputed {key} != stored {stored_name}"
        )


def test_recompute_reproduces_stored_values_under_compound_weights(
    multi_region_band_problem,
):
    """The case `_row_weight` exists for. With a multi-key weights= dict,
    the per-band means are computed against a compound demand/equity blend,
    not raw demand -- so a recompute that reached for the demand column
    would silently produce different numbers than the primary dimension's
    stored ones."""
    result = multi_region_band_problem.solve(
        p=1, weights={"demand": 0.6, "equity": 0.4}
    )
    row = result.solution_df.iloc[0]

    # The two weightings must actually disagree, or a recompute using the
    # wrong one would still land on the right answer and this would pass
    # while testing nothing. Band 1 is where they differ: 15.71 compound
    # against 15.0 demand-weighted (see the fixture's docstring).
    assert row["weighted_by_equity_group"][1] == pytest.approx(15.71)
    demand_weighted = (
        multi_region_band_problem.copy()
        .solve(p=1)
        .solution_df.iloc[0]["weighted_by_equity_group"][1]
    )
    assert demand_weighted == pytest.approx(15.0)

    result.site_problem._primary_equity_label = None
    _, recomputed = result._resolve_equity_view(equity="equity")

    assert _same(
        recomputed["weighted_by_equity_group"], row["weighted_by_equity_group"]
    )
    assert recomputed["inter_tertile_ratio"] == pytest.approx(
        row["inter_tertile_ratio"]
    )
    assert recomputed["gap_absolute_weighted"] == pytest.approx(
        row["gap_absolute_weighted"]
    )


# --- Selecting a second dimension ------------------------------------------


def test_second_dimension_gives_its_own_bands(two_dimension_problem):
    result = two_dimension_problem.solve(p=1)

    meta_imd, imd = result._resolve_equity_view()
    meta_rural, rural = result._resolve_equity_view(equity="rurality")

    assert meta_imd["label"] == "equity"
    assert meta_rural["label"] == "rurality"

    # Winning site's costs are 25 / 5 / 15 for LSOA_1 / 2 / 3, one region
    # per band under both dimensions -- so the per-band means coincide...
    assert imd["weighted_by_equity_group"] == {1: 25.0, 5: 5.0, 10: 15.0}
    assert rural["weighted_by_equity_group"] == {1: 25.0, 5: 5.0, 10: 15.0}

    # ...while the tertile ratio does not, because the two dimensions
    # disagree about which end is disadvantaged: IMD divides decile 1 (25)
    # by decile 10 (15), rurality divides band 10 (15) by band 1 (25).
    assert imd["inter_tertile_ratio"] == pytest.approx(25 / 15)
    assert rural["inter_tertile_ratio"] == pytest.approx(15 / 25)


def test_both_dimensions_return_the_same_keys(two_dimension_problem):
    """A stored breakdown and a recomputed one must be interchangeable to
    callers -- including the three verbal descriptors, which solution_df
    stores under `*_description` while the helper returns `*_desc`."""
    result = two_dimension_problem.solve(p=1)
    _, primary = result._resolve_equity_view()
    _, secondary = result._resolve_equity_view(equity="rurality")

    required = set(_EQUITY_BAND_STORED_NAMES)
    assert required <= set(primary)
    assert required <= set(secondary)

    for key in ("gap_absolute_desc", "gap_relative_desc", "inter_tertile_desc"):
        assert isinstance(primary[key], str)
        assert isinstance(secondary[key], str)


def test_unknown_dimension_raises_listing_registered(two_dimension_problem):
    result = two_dimension_problem.solve(p=1)
    with pytest.raises(ValueError, match=r"Registered: \['equity', 'rurality'\]"):
        result._resolve_equity_view(equity="ethnicity")


def test_returns_none_pair_when_no_equity_registered(loaded_problem):
    """Callers keep their existing "no equity data" handling rather than
    each having to catch an exception."""
    result = loaded_problem.solve(p=1)
    meta, bands = result._resolve_equity_view()
    assert meta is None
    assert bands is None


def test_dimension_registered_after_solving_raises_clearly(
    loaded_problem_with_equity,
):
    """problem_df is a snapshot: a dimension registered afterwards is not
    in it, and its bands genuinely cannot be computed. Say so, rather than
    surfacing a bare KeyError from the groupby."""
    result = loaded_problem_with_equity.solve(p=1)
    loaded_problem_with_equity.add_equity_data(
        pd.DataFrame(
            {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "late_band": [1, 5, 10]}
        ),
        equity_col="late_band",
        common_col="location_id",
        label="late",
        primary=False,
    )
    with pytest.raises(ValueError, match="was produced before that dimension"):
        result._resolve_equity_view(equity="late")


def test_row_argument_wins_over_solution_rank(loaded_problem_with_equity):
    """Callers that have already selected a row must not risk this
    resolving a different one."""
    result = loaded_problem_with_equity.solve(p=1)
    chosen = result.solution_df.iloc[-1]
    _, bands = result._resolve_equity_view(row=chosen, solution_rank=1)
    assert _same(
        bands["weighted_by_equity_group"], chosen["weighted_by_equity_group"]
    )


def test_reserved_row_weight_column_name_is_rejected(loaded_problem):
    """`_row_weight` carries the weights each combination was scored with.
    A user column of that name can reach the same frame via
    drop_extra_cols=False, and silently overwriting it would leave
    something that still looks like the user's column while holding the
    solver's numbers."""
    ids = ["LSOA_1", "LSOA_2", "LSOA_3"]
    loaded_problem.add_demand(
        pd.DataFrame(
            {
                "location_id": ids,
                "demand": [100, 200, 150],
                "_row_weight": [7, 8, 9],
            }
        ),
        demand_col="demand",
        location_id_col="location_id",
        drop_extra_cols=False,
    )
    with pytest.raises(ValueError, match="'_row_weight' is reserved"):
        loaded_problem.solve(p=1)
