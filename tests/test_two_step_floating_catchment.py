"""Tests for `SiteProblem.two_step_floating_catchment()` and
`SiteSolutionSet.two_step_floating_catchment()`.

Binary threshold coverage (`within_threshold`) treats every demand region
within the threshold identically, regardless of how many other regions
compete for the same site's supply. 2SFCA (Rushton 2003; Luo & Wang 2003)
fixes that by computing, for each site, a supply-to-demand ratio over its
own catchment (step 1), then summing the ratios of every site reachable
from a demand region (step 2) -- so a region that shares its nearest site
with many others scores lower than one with the same travel time but less
competition.

Fixture arithmetic (hand-derived, see `sfca_problem` in `conftest.py`):
p=2 sites {Site_1, Site_2}, catchment_size=15.
  LSOA_1 reaches both sites (10, 12); LSOA_2 only Site_1 (8 < 15 < 30);
  LSOA_3 only Site_2 (5 < 15 < 30); LSOA_Isolated reaches neither (1000, 1000).
  Demand: 100, 100, 50, 75. Supply: Site_1=10, Site_2=5.
  R_1 = 10 / (100+100) = 0.05
  R_2 = 5 / (100+50) = 1/30
  accessibility: LSOA_1 = R_1+R_2 = 0.08333..., LSOA_2 = R_1 = 0.05,
  LSOA_3 = R_2 = 0.03333..., LSOA_Isolated = 0
"""

import numpy as np
import pandas as pd
import pytest

import lokigi


R_1 = 10 / 200
R_2 = 5 / 150


def test_step1_and_step2_match_hand_computation(sfca_problem):
    region_frame, site_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )

    assert site_frame.loc["Site_1", "catchment_demand"] == 200
    assert site_frame.loc["Site_2", "catchment_demand"] == 150
    assert site_frame.loc["Site_1", "ratio"] == pytest.approx(R_1)
    assert site_frame.loc["Site_2", "ratio"] == pytest.approx(R_2)
    assert site_frame.loc["Site_1", "n_regions_in_catchment"] == 2
    assert site_frame.loc["Site_2", "n_regions_in_catchment"] == 2

    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(R_1 + R_2)
    assert region_frame.loc["LSOA_2", "accessibility"] == pytest.approx(R_1)
    assert region_frame.loc["LSOA_3", "accessibility"] == pytest.approx(R_2)
    assert region_frame.loc["LSOA_1", "n_sites_in_catchment"] == 2
    assert region_frame.loc["LSOA_2", "n_sites_in_catchment"] == 1
    assert region_frame.loc["LSOA_3", "n_sites_in_catchment"] == 1


def test_binary_coverage_treats_lsoa_1_and_lsoa_2_identically_2sfca_does_not(
    sfca_problem,
):
    """The motivating case: LSOA_1 (min_cost=10) and LSOA_2 (min_cost=8) are
    both under the coverage threshold of 15, so binary coverage rates them
    identically as "covered". 2SFCA does not, because LSOA_2 has no second
    site and shares Site_1's supply with LSOA_1."""
    result = sfca_problem.evaluate_single_solution_single_objective(
        objective="p_median",
        site_names=["Site_1", "Site_2"],
        threshold_for_coverage=15,
    )
    problem_df = result.show_result_df()
    assert bool(problem_df.set_index("location_id").loc["LSOA_1", "within_threshold"])
    assert bool(problem_df.set_index("location_id").loc["LSOA_2", "within_threshold"])

    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    assert region_frame.loc["LSOA_1", "accessibility"] != pytest.approx(
        region_frame.loc["LSOA_2", "accessibility"]
    )
    assert region_frame.loc["LSOA_1", "accessibility"] > region_frame.loc["LSOA_2", "accessibility"]


def test_conservation_invariant_holds_with_no_empty_catchments(sfca_problem):
    """sum(demand * accessibility) == sum(supply) whenever every scored site
    has at least one region in its catchment: summing demand-weighted
    ratios over all regions is exactly redistributing the supply that
    produced them."""
    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    total = (region_frame["demand"] * region_frame["accessibility"]).sum()
    assert total == pytest.approx(10 + 5)


def test_empty_catchment_site_warns_and_gets_nan_ratio_without_affecting_others(
    sfca_problem,
):
    with pytest.warns(UserWarning, match="Site_Isolated"):
        region_frame, site_frame = sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2", "Site_Isolated"],
            return_site_ratios=True,
        )

    assert site_frame.loc["Site_Isolated", "catchment_demand"] == 0
    assert np.isnan(site_frame.loc["Site_Isolated", "ratio"])

    # Site_Isolated's undefined ratio must not poison the other regions'
    # accessibility -- these match test_step1_and_step2_match_hand_computation
    # exactly, with Site_Isolated included or not.
    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(R_1 + R_2)
    assert region_frame.loc["LSOA_2", "accessibility"] == pytest.approx(R_1)
    assert region_frame.loc["LSOA_3", "accessibility"] == pytest.approx(R_2)


def test_unreachable_region_is_zero_not_nan(sfca_problem):
    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    accessibility = region_frame.loc["LSOA_Isolated", "accessibility"]
    assert accessibility == 0
    assert not np.isnan(accessibility)
    assert region_frame.loc["LSOA_Isolated", "n_sites_in_catchment"] == 0


def test_widening_catchment_size_never_decreases_sites_in_catchment(sfca_problem):
    narrow = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=5, site_names=["Site_1", "Site_2"]
    )
    medium = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    wide = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=1000, site_names=["Site_1", "Site_2"]
    )

    for region in narrow.index:
        assert (
            narrow.loc[region, "n_sites_in_catchment"]
            <= medium.loc[region, "n_sites_in_catchment"]
            <= wide.loc[region, "n_sites_in_catchment"]
        )
    # And it's a real effect, not a coincidental tie throughout the range.
    assert wide["n_sites_in_catchment"].sum() > narrow["n_sites_in_catchment"].sum()


def test_per_capita_scales_accessibility_but_not_site_frame(sfca_problem):
    region_frame, site_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )
    scaled_region_frame, scaled_site_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        per_capita=100_000,
        return_site_ratios=True,
    )

    pd.testing.assert_series_equal(
        scaled_region_frame["accessibility"],
        region_frame["accessibility"] * 100_000,
    )
    pd.testing.assert_frame_equal(scaled_site_frame, site_frame)


def test_matrix_argument_uses_secondary_costs(sfca_problem_with_secondary_matrix):
    """On the `public_transport` secondary matrix, every region reaches
    only Site_2 (cost 5 < 15), unlike the primary matrix where reachability
    varies by region -- so every region ends up with the same, uniform
    accessibility, and it differs from the primary-matrix result."""
    primary = sfca_problem_with_secondary_matrix.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    secondary = sfca_problem_with_secondary_matrix.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
        matrix="public_transport",
    )

    expected_ratio = 5 / (100 + 100 + 50 + 75)
    for region in secondary.index:
        assert secondary.loc[region, "accessibility"] == pytest.approx(expected_ratio)

    assert not np.isclose(
        primary.loc["LSOA_2", "accessibility"], secondary.loc["LSOA_2", "accessibility"]
    )


def test_no_solve_required(sfca_problem):
    """Scoring the full candidate site set works without ever calling
    solve() -- travel_and_demand_df is built lazily on first use."""
    assert sfca_problem.travel_and_demand_df is None

    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15
    )

    assert sfca_problem.travel_and_demand_df is not None
    # Default (no site_names/site_indices) scores every candidate site,
    # including Site_Isolated.
    assert len(region_frame) == 4


def test_site_problem_and_solution_set_agree(sfca_problem):
    result = sfca_problem.solve(p=2, objectives="p_median")

    problem_side = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    solution_side = result.two_step_floating_catchment(
        supply_col="supply",
        catchment_size=15,
        site_names=["Site_1", "Site_2"],
    )

    pd.testing.assert_frame_equal(problem_side, solution_side)


def test_site_indices_resolve_the_same_as_site_names(sfca_problem):
    by_names = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"]
    )
    by_indices = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_indices=[0, 1]
    )
    pd.testing.assert_frame_equal(by_names, by_indices)


# --- Error handling -------------------------------------------------------


def test_both_site_names_and_site_indices_raises(sfca_problem):
    with pytest.raises(ValueError, match="at most one"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1"],
            site_indices=[0],
        )


def test_no_demand_data_raises():
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
            "supply": [10, 5],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_1": [10.0, 8.0],
            "Site_2": [12.0, 30.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    with pytest.raises(ValueError, match="add_demand"):
        problem.two_step_floating_catchment(supply_col="supply", catchment_size=15)


def test_unknown_matrix_label_raises(sfca_problem):
    with pytest.raises(ValueError, match="Unknown secondary travel matrix"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2"],
            matrix="does_not_exist",
        )


def test_missing_supply_col_raises(sfca_problem):
    with pytest.raises(ValueError, match="not found in candidate_sites"):
        sfca_problem.two_step_floating_catchment(
            supply_col="does_not_exist", catchment_size=15
        )


def test_non_numeric_supply_raises():
    problem = _sfca_problem_with_supply_values(["ten", "five"])
    with pytest.raises(TypeError, match="numeric"):
        problem.two_step_floating_catchment(supply_col="supply", catchment_size=15)


def test_null_supply_raises():
    problem = _sfca_problem_with_supply_values([10, np.nan])
    with pytest.raises(ValueError, match="Site_2"):
        problem.two_step_floating_catchment(supply_col="supply", catchment_size=15)


def test_negative_supply_raises():
    problem = _sfca_problem_with_supply_values([10, -5])
    with pytest.raises(ValueError, match="Site_2"):
        problem.two_step_floating_catchment(supply_col="supply", catchment_size=15)


def _sfca_problem_with_supply_values(supply_values):
    """Minimal 2-site/2-demand problem with an arbitrary `supply` column,
    for exercising `_resolve_supply`'s validation in isolation."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2"], "demand": [100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
            "supply": supply_values,
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_1": [10.0, 8.0],
            "Site_2": [12.0, 30.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem
