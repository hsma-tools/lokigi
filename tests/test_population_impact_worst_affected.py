"""Tests for `SolutionComparator.population_impact_worst_affected()` --
the top-N drill-down naming which specific demand locations were hit
hardest by a change, rather than leaving a reader to compute it from
`population_impact_summary()`'s region-wide totals or the raw
`return_per_region=True` frame themselves.

Uses `loaded_problem` (Site_A/B/C, LSOA_1/2/3, demand 100/200/150) --
shared with `test_sites_vs_baseline.py`. Baseline = {Site_A, Site_B},
candidate = {Site_A, Site_C}, evaluated directly via
`evaluate_baseline(site_names=...)` for a deterministic set (not solver
search) so exact before/after numbers are hand-computable:

    LSOA_1: Site_A=10, Site_B=25, Site_C=30 -> baseline min=10, candidate min=10 (unchanged)
    LSOA_2: Site_A=20, Site_B=5,  Site_C=10 -> baseline min=5,  candidate min=10 (worsened +5)
    LSOA_3: Site_A=30, Site_B=15, Site_C=8  -> baseline min=15, candidate min=8  (improved -7)
"""

import pytest

from lokigi.site_solutions import SolutionComparator


@pytest.fixture
def comparator(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    candidate = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_C"])
    return SolutionComparator(baseline, candidate, labels=("Current", "Proposed"))


def test_worsened_direction_default(comparator):
    result = comparator.population_impact_worst_affected()

    assert list(result.index) == ["LSOA_2"]
    assert result["Before"].loc["LSOA_2"] == 5.0
    assert result["After"].loc["LSOA_2"] == 10.0
    assert result["Change"].loc["LSOA_2"] == 5.0
    assert result["Previous site"].loc["LSOA_2"] == "Site_B"
    assert result["New site"].loc["LSOA_2"] == "Site_C"
    assert result["People affected"].loc["LSOA_2"] == 200


def test_improved_direction(comparator):
    result = comparator.population_impact_worst_affected(direction="improved")

    assert list(result.index) == ["LSOA_3"]
    assert result["Before"].loc["LSOA_3"] == 15.0
    assert result["After"].loc["LSOA_3"] == 8.0
    assert result["Change"].loc["LSOA_3"] == -7.0
    assert result["Previous site"].loc["LSOA_3"] == "Site_B"
    assert result["New site"].loc["LSOA_3"] == "Site_C"
    assert result["People affected"].loc["LSOA_3"] == 150


def test_previous_and_new_site_unaffected_by_secondary_travel_matrix(
    loaded_problem_with_secondary_matrix,
):
    """previous_site/new_site should reflect the SAME matrix= being diffed
    (selected_site__<label> for a secondary matrix), not silently fall
    back to the primary matrix's selected_site."""
    baseline = loaded_problem_with_secondary_matrix.evaluate_baseline(
        site_names=["Site_A", "Site_B"]
    )
    candidate = loaded_problem_with_secondary_matrix.evaluate_baseline(
        site_names=["Site_A", "Site_C"]
    )
    comparator = SolutionComparator(baseline, candidate, labels=("Current", "Proposed"))

    result = comparator.population_impact_worst_affected(matrix="public_transport")
    assert "Previous site" in result.columns
    assert "New site" in result.columns
    assert result["Previous site"].notna().all()
    assert result["New site"].notna().all()


def test_unchanged_location_never_appears(comparator):
    worsened = comparator.population_impact_worst_affected(direction="worsened")
    improved = comparator.population_impact_worst_affected(direction="improved")

    assert "LSOA_1" not in worsened.index
    assert "LSOA_1" not in improved.index


def test_n_caps_number_of_rows_returned(comparator):
    result = comparator.population_impact_worst_affected(direction="worsened", n=0)
    assert len(result) == 0


def test_fewer_than_n_available_returns_all(comparator):
    result = comparator.population_impact_worst_affected(direction="worsened", n=10)
    assert len(result) == 1


def test_index_named_after_demand_location_id_col(comparator):
    result = comparator.population_impact_worst_affected()
    assert result.index.name == "location_id"


def test_invalid_direction_raises(comparator):
    with pytest.raises(ValueError, match="direction"):
        comparator.population_impact_worst_affected(direction="sideways")


def test_columns_named_after_registered_unit(basic_problem, demand_df, candidate_df, travel_df):
    """Before/After/Change columns previously said "(mins)" unconditionally,
    regardless of what unit (if any) was actually registered via
    add_travel_matrix(unit=...) -- wrong for e.g. a distance matrix in
    miles, and misleading even when no unit was registered at all."""
    basic_problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
    basic_problem.add_travel_matrix(travel_df, source_col="source_id", unit="miles")

    baseline = basic_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    candidate = basic_problem.evaluate_baseline(site_names=["Site_A", "Site_C"])
    comparator = SolutionComparator(baseline, candidate, labels=("Current", "Proposed"))

    result = comparator.population_impact_worst_affected()
    assert "Before (miles)" in result.columns
    assert "After (miles)" in result.columns
    assert "Change (miles)" in result.columns
    assert "Before" not in result.columns
