"""Tests for `SiteSolutionSet.site_allocation_summary()`.

Every uncapacitated evaluation already assigns each demand region to its
closest selected site (`site.py`'s `idxmin`-based `selected_site` column),
but nothing in the package aggregated that by site until now.
`site_allocation_summary()` reports the share of demand (or of regions)
closest to each site in a chosen solution -- e.g. showing that a proposed
third site only picks up 11% of demand, weakening the case for opening it
even where it improves the average travel time -- and the average travel
cost incurred by each site's group, e.g. showing that closing a site would
roughly double the average distance travelled by the patients who used to
be closest to it (the `average_travel_cost` column, inspired by Gill
Baker's work on average travel distance per patient).

Fixture arithmetic used below (hand-derived from `conftest.py`):

- `loaded_problem`: p=2 {Site_A, Site_B}. `travel_df` gives
  LSOA_1 -> A (10 < 25), LSOA_2 -> B (5 < 20), LSOA_3 -> B (15 < 30).
  `demand_df` is 100/200/150 (total 450). Demand-weighted:
  A = 100/450, B = 350/450. Region-count: A = 1/3, B = 2/3.
- `loaded_problem`: p=3 {Site_A, Site_B, Site_C}. LSOA_1 -> A (10 < 25 < 30),
  LSOA_2 -> B (5 < 10 < 20), LSOA_3 -> C (8 < 15 < 30). Uniform one region
  each, so demand-weighted = region-count = 100/450, 200/450, 150/450.
- `loaded_problem_with_secondary_matrix`: the `public_transport` matrix
  has uniform per-site cost (Site_C=5, Site_B=50, Site_A=90), so on that
  matrix *every* region's closest site is Site_C regardless of solution.
- `five_site_problem`: p=3 {Site_1, Site_2, Site_3}. `travel_df` gives
  LSOA_1 -> Site_3 (24 < 25/38), LSOA_2 -> Site_3 (13 < 18/40),
  LSOA_3 -> Site_1 (10 < 24/29), LSOA_4 -> Site_3 (13 < 28/29). Demand is
  uniform (100 each), so Site_1 = 1/4, Site_2 = 0 (closest to nothing),
  Site_3 = 3/4. This is the headline zero-allocation case.
- `tied_score_problem`: p=2 {Site_A, Site_B}. Both columns are identical
  (10.0/20.0), so `idxmin`'s first-column tiebreak sends every region to
  Site_A; Site_B is closest to nothing.
"""

import pandas as pd
import pytest


def test_demand_weighted_differs_from_region_count(loaded_problem):
    """Trap: a region-count implementation under by="demand" would return
    exactly 1/3 for Site_A here, matching the by="regions" case below. The
    correct, demand-weighted answer is 100/450, which is a different
    number, so this fails loudly if the two bases are ever conflated."""
    result = loaded_problem.solve(p=2)

    demand_summary = result.site_allocation_summary(
        by="demand", site_names=["Site_A", "Site_B"]
    )
    assert demand_summary.loc["Site_A", "proportion"] == pytest.approx(100 / 450)
    assert demand_summary.loc["Site_B", "proportion"] == pytest.approx(350 / 450)
    assert demand_summary.loc["Site_A", "proportion"] != pytest.approx(1 / 3)
    assert demand_summary.loc["Site_A", "n_regions"] == 1
    assert demand_summary.loc["Site_B", "n_regions"] == 2

    region_summary = result.site_allocation_summary(
        by="regions", site_names=["Site_A", "Site_B"]
    )
    assert region_summary.loc["Site_A", "proportion"] == pytest.approx(1 / 3)
    assert region_summary.loc["Site_B", "proportion"] == pytest.approx(2 / 3)


def test_zero_allocation_site_appears_as_explicit_zero_row(five_site_problem):
    """The headline regression guard: a selected site closest to no region
    must not be silently dropped by the underlying groupby."""
    result = five_site_problem.solve(p=3)

    summary = result.site_allocation_summary(
        by="demand", site_names=["Site_1", "Site_2", "Site_3"]
    )

    assert list(summary.index) == ["Site_1", "Site_2", "Site_3"]
    assert len(summary) == 3
    assert summary.loc["Site_2", "n_regions"] == 0
    assert summary.loc["Site_2", "proportion"] == 0.0
    assert summary.loc["Site_1", "proportion"] == pytest.approx(0.25)
    assert summary.loc["Site_3", "proportion"] == pytest.approx(0.75)

    # Demand is uniform here, so by="regions" should match exactly.
    region_summary = result.site_allocation_summary(
        by="regions", site_names=["Site_1", "Site_2", "Site_3"]
    )
    pd.testing.assert_series_equal(
        summary["proportion"], region_summary["proportion"]
    )


def test_matrix_argument_switches_to_secondary_allocation(
    loaded_problem_with_secondary_matrix,
):
    """Every region is closest to Site_C on the uniform-cost secondary
    matrix, regardless of what the primary matrix says -- pins that
    `matrix=` actually reads `selected_site__public_transport` rather than
    the primary `selected_site`."""
    result = loaded_problem_with_secondary_matrix.solve(p=3)

    primary = result.site_allocation_summary(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    assert primary.loc["Site_A", "proportion"] == pytest.approx(100 / 450)
    assert primary.loc["Site_B", "proportion"] == pytest.approx(200 / 450)
    assert primary.loc["Site_C", "proportion"] == pytest.approx(150 / 450)

    secondary = result.site_allocation_summary(
        site_names=["Site_A", "Site_B", "Site_C"], matrix="public_transport"
    )
    assert secondary.loc["Site_C", "proportion"] == pytest.approx(1.0)
    assert secondary.loc["Site_A", "proportion"] == 0.0
    assert secondary.loc["Site_B", "proportion"] == 0.0


def test_exact_ties_go_to_lowest_indexed_site_not_split(tied_score_problem):
    result = tied_score_problem.solve(p=2)

    summary = result.site_allocation_summary(site_names=["Site_A", "Site_B"])

    assert summary.loc["Site_A", "proportion"] == pytest.approx(1.0)
    assert summary.loc["Site_B", "proportion"] == 0.0
    assert summary.loc["Site_B", "n_regions"] == 0


def test_proportions_sum_to_one_on_real_data(brighton_problem):
    result = brighton_problem.solve(p=3, objectives="p_median", threshold_for_coverage=10)

    for rank in range(1, 6):
        summary = result.site_allocation_summary(solution_rank=rank)
        assert summary["proportion"].sum() == pytest.approx(1.0)
        assert len(summary) == 3


def test_invalid_by_raises(loaded_problem):
    result = loaded_problem.solve(p=2)

    with pytest.raises(ValueError, match="by must be"):
        result.site_allocation_summary(by="both")
    with pytest.raises(ValueError, match="by must be"):
        result.site_allocation_summary(by="Demand")
    with pytest.raises(ValueError, match="by must be"):
        result.site_allocation_summary(by=None)


def test_by_demand_without_demand_data_raises(loaded_problem, monkeypatch):
    """Deliberately diverges from `_coverage_proportion`'s silent
    region-count fallback: `by="demand"` is an explicit request, so a
    missing demand column raises rather than quietly reporting region
    shares under a demand label. Unreachable via `solve()` (which installs
    a uniform-demand fallback), so this stands in with a direct
    monkeypatch of the problem attribute."""
    result = loaded_problem.solve(p=2)
    monkeypatch.setattr(result.site_problem, "_demand_data_demand_col", None)

    with pytest.raises(ValueError, match="add_demand"):
        result.site_allocation_summary(by="demand")

    # by="regions" never needed demand data, so it still works.
    summary = result.site_allocation_summary(by="regions")
    assert summary["proportion"].sum() == pytest.approx(1.0)
    assert "total_demand" not in summary.columns


def test_selection_arguments_take_priority_site_indices_over_site_names(loaded_problem):
    result = loaded_problem.solve(p=2)

    by_indices = result.site_allocation_summary(site_indices=[0, 1])
    by_names = result.site_allocation_summary(site_names=["Site_A", "Site_B"])

    pd.testing.assert_frame_equal(by_indices, by_names)


def test_average_travel_cost_is_demand_weighted_by_default(loaded_problem):
    """`average_travel_cost` for a site with more than one region assigned
    must be the demand-weighted mean, not the plain mean -- Site_B here
    serves LSOA_2 (cost=5, demand=200) and LSOA_3 (cost=15, demand=150), so
    a plain mean would give 10.0, same as the by="regions" case below. The
    correct, demand-weighted answer is (5*200 + 15*150) / 350 = 9.2857,
    which is a different number."""
    result = loaded_problem.solve(p=2)

    summary = result.site_allocation_summary(site_names=["Site_A", "Site_B"])

    assert summary.loc["Site_A", "average_travel_cost"] == pytest.approx(10.0)
    assert summary.loc["Site_B", "average_travel_cost"] == pytest.approx(
        (5 * 200 + 15 * 150) / 350
    )
    assert summary.loc["Site_B", "average_travel_cost"] != pytest.approx(10.0)


def test_average_travel_cost_by_regions_is_plain_mean(loaded_problem):
    result = loaded_problem.solve(p=2)

    summary = result.site_allocation_summary(by="regions", site_names=["Site_A", "Site_B"])

    assert summary.loc["Site_A", "average_travel_cost"] == pytest.approx(10.0)
    assert summary.loc["Site_B", "average_travel_cost"] == pytest.approx((5 + 15) / 2)


def test_average_travel_cost_is_nan_not_zero_for_zero_allocation_site(five_site_problem):
    """The same zero-allocation site that gets an explicit 0 in `proportion`
    must be `NaN` in `average_travel_cost` -- there is no travel cost to
    average over zero regions, and 0 would misleadingly read as "instant
    to reach" rather than "not applicable"."""
    result = five_site_problem.solve(p=3)

    summary = result.site_allocation_summary(site_names=["Site_1", "Site_2", "Site_3"])

    assert pd.isna(summary.loc["Site_2", "average_travel_cost"])
    assert summary.loc["Site_1", "average_travel_cost"] == pytest.approx(10.0)
    assert summary.loc["Site_3", "average_travel_cost"] == pytest.approx((24 + 13 + 13) / 3)


def test_average_travel_cost_matrix_argument_switches_matrix(
    loaded_problem_with_secondary_matrix,
):
    """Every region's closest site on the uniform-cost secondary matrix is
    Site_C at cost 5, regardless of the primary matrix's own per-site
    averages -- pins that `matrix=` reaches `min_cost__public_transport`."""
    result = loaded_problem_with_secondary_matrix.solve(p=3)

    primary = result.site_allocation_summary(site_names=["Site_A", "Site_B", "Site_C"])
    assert primary.loc["Site_A", "average_travel_cost"] == pytest.approx(10.0)
    assert primary.loc["Site_B", "average_travel_cost"] == pytest.approx(5.0)
    assert primary.loc["Site_C", "average_travel_cost"] == pytest.approx(8.0)

    secondary = result.site_allocation_summary(
        site_names=["Site_A", "Site_B", "Site_C"], matrix="public_transport"
    )
    assert secondary.loc["Site_C", "average_travel_cost"] == pytest.approx(5.0)
    assert pd.isna(secondary.loc["Site_A", "average_travel_cost"])
    assert pd.isna(secondary.loc["Site_B", "average_travel_cost"])
