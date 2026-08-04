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

import math

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


def test_nan_travel_cost_treated_as_unreachable_not_propagated():
    """A NaN travel cost is rejected at registration time unless
    `add_travel_matrix(allow_missing=True)` opts in -- passed here so a
    genuinely missing origin-destination route reaches
    `two_step_floating_catchment()`, rather than raising at registration.

    This underwrites the safety of building `weight_matrix` from boolean
    comparisons against `cost_frame` (`cost_frame <= x`, `.where(cost_frame
    <= x, 0.0)`): a NaN comparison is always False in pandas/numpy, so a
    NaN cost cell always resolves to weight 0.0 ("not in catchment") and
    never leaks a raw NaN into `weight_matrix` -- a precondition for
    `_two_step_floating_catchment`'s internals to use `.dot()` (which
    propagates NaN) instead of `.mul().sum()` (which silently skips it)
    without changing behaviour on valid data.

    LSOA_1's cost to Site_1 is NaN (route unknown); LSOA_1 can still reach
    Site_2 (cost 12 <= 15). Site_1's catchment must exclude LSOA_1 entirely
    (catchment_demand = LSOA_2's 100 only) rather than becoming NaN itself.
    """
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 100, 50]}
    )
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
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_1": [np.nan, 8.0, 30.0],
            "Site_2": [12.0, 30.0, 5.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)

    region_frame, site_frame = problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, return_site_ratios=True
    )

    assert site_frame.loc["Site_1", "catchment_demand"] == 100
    assert site_frame.loc["Site_1", "ratio"] == pytest.approx(10 / 100)

    # LSOA_1 reaches only Site_2 -- Site_1's NaN cost must not make its own
    # accessibility NaN, nor silently count it as "in catchment" for Site_1.
    expected_r2 = 5 / (100 + 50)  # Site_2's catchment: LSOA_1 (12) + LSOA_3 (5)
    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(expected_r2)
    assert not np.isnan(region_frame.loc["LSOA_1", "accessibility"])
    assert region_frame.loc["LSOA_1", "n_sites_in_catchment"] == 1


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


@pytest.mark.parametrize("bad_per_capita", [0, -1.0])
def test_non_positive_per_capita_raises(sfca_problem, bad_per_capita):
    with pytest.raises(ValueError, match="per_capita"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2"],
            per_capita=bad_per_capita,
        )


def test_non_numeric_per_capita_raises(sfca_problem):
    with pytest.raises(ValueError, match="per_capita"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            catchment_size=15,
            site_names=["Site_1", "Site_2"],
            per_capita="fast",
        )


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
    # Also pins "supply quantity" specifically, not just the site name --
    # this method's validation is shared with site_capacity_summary()'s via
    # a common resolver (_resolve_site_numeric_column), parametrised by a
    # quantity_label ("supply quantity" here, "capacity" there); a
    # regression that mixed the two labels up would still match on
    # "Site_2" alone.
    with pytest.raises(ValueError, match="not a valid supply quantity.*Site_2"):
        problem.two_step_floating_catchment(supply_col="supply", catchment_size=15)


# --- Enhanced 2SFCA: step-decay bands --------------------------------------


def test_step_decay_matches_hand_computation(sfca_problem):
    """Bands [(10, 1.0), (15, 0.5), (30, 0.2)] against sfca_problem's costs
    (Site_1: LSOA_1=10, LSOA_2=8, LSOA_3=30; Site_2: LSOA_1=12, LSOA_2=30,
    LSOA_3=5):

      weight(Site_1): LSOA_1=1.0 (<=10), LSOA_2=1.0 (<=10), LSOA_3=0.2 (<=30)
      weight(Site_2): LSOA_1=0.5 (<=15), LSOA_2=0.2 (<=30), LSOA_3=1.0 (<=10)

      catchment_demand(Site_1) = 1.0*100 + 1.0*100 + 0.2*50 = 210 -> R_1 = 10/210
      catchment_demand(Site_2) = 0.5*100 + 0.2*100 + 1.0*50 = 120 -> R_2 = 5/120
      accessibility(LSOA_1) = 1.0*R_1 + 0.5*R_2
      accessibility(LSOA_2) = 1.0*R_1 + 0.2*R_2
      accessibility(LSOA_3) = 0.2*R_1 + 1.0*R_2
    """
    bands = [(10, 1.0), (15, 0.5), (30, 0.2)]
    region_frame, site_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=bands,
        site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )

    expected_r1 = 10 / 210
    expected_r2 = 5 / 120
    assert site_frame.loc["Site_1", "ratio"] == pytest.approx(expected_r1)
    assert site_frame.loc["Site_2", "ratio"] == pytest.approx(expected_r2)

    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(
        1.0 * expected_r1 + 0.5 * expected_r2
    )
    assert region_frame.loc["LSOA_2", "accessibility"] == pytest.approx(
        1.0 * expected_r1 + 0.2 * expected_r2
    )
    assert region_frame.loc["LSOA_3", "accessibility"] == pytest.approx(
        0.2 * expected_r1 + 1.0 * expected_r2
    )

    total = (region_frame["demand"] * region_frame["accessibility"]).sum()
    assert total == pytest.approx(10 + 5)


def test_step_decay_bands_need_not_be_presorted(sfca_problem):
    sorted_bands = [(10, 1.0), (15, 0.5), (30, 0.2)]
    shuffled_bands = [(30, 0.2), (10, 1.0), (15, 0.5)]

    sorted_result = sfca_problem.two_step_floating_catchment(
        supply_col="supply", distance_decay=sorted_bands, site_names=["Site_1", "Site_2"]
    )
    shuffled_result = sfca_problem.two_step_floating_catchment(
        supply_col="supply", distance_decay=shuffled_bands, site_names=["Site_1", "Site_2"]
    )
    pd.testing.assert_frame_equal(sorted_result, shuffled_result)


def test_single_band_distance_decay_equals_classic_catchment_size(sfca_problem):
    """The concrete proof that classic 2SFCA is the single-band special
    case of the generalised weight-matrix engine, not a diverged path."""
    via_catchment_size = sfca_problem.two_step_floating_catchment(
        supply_col="supply", catchment_size=15, site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )
    via_distance_decay = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=[(15, 1.0)],
        site_names=["Site_1", "Site_2"],
        return_site_ratios=True,
    )
    pd.testing.assert_frame_equal(via_catchment_size[0], via_distance_decay[0])
    pd.testing.assert_frame_equal(via_catchment_size[1], via_distance_decay[1])


def test_step_decay_band_boundary_is_inclusive(sfca_problem):
    """A cost exactly equal to a band's upper_bound belongs to that band,
    not the next one up -- matching catchment_size's existing inclusive
    `<=` convention. LSOA_1's and LSOA_2's cost to Site_1 are 10 and 8
    (both <= 10, the first band); only Site_1 is scored, so
    catchment_demand = 100 + 100 = 200 if the boundary is inclusive, not
    100 (LSOA_2 only) if a cost of exactly 10 were wrongly pushed into the
    second band."""
    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=[(10, 1.0), (20, 0.5)],
        site_names=["Site_1"],
    )
    expected = 10 / (100 + 100) * 1.0
    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(expected)


def test_step_decay_empty_list_raises(sfca_problem):
    with pytest.raises(ValueError, match="at least one"):
        sfca_problem.two_step_floating_catchment(supply_col="supply", distance_decay=[])


def test_step_decay_duplicate_upper_bounds_raises(sfca_problem):
    with pytest.raises(ValueError, match="unique"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", distance_decay=[(10, 1.0), (10, 0.5)]
        )


def test_step_decay_non_positive_upper_bound_raises(sfca_problem):
    with pytest.raises(ValueError, match="positive"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", distance_decay=[(0, 1.0), (10, 0.5)]
        )


def test_step_decay_negative_weight_raises(sfca_problem):
    with pytest.raises(ValueError, match="non-negative"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", distance_decay=[(10, -0.5)]
        )


# --- Enhanced 2SFCA: continuous Gaussian decay ------------------------------


def test_gaussian_decay_boundary_and_truncation_values(gaussian_decay_problem):
    """Dai (2010)'s truncated Gaussian: weight is exactly 1.0 at distance 0
    and exactly 0.0 at distance == catchment_size (the truncation radius),
    both algebraically exact (not approximations of the general formula).
    LSOA_Beyond (distance 100) is also exactly 0.0 -- truncation beyond the
    radius, not decay towards it."""
    region_frame = gaussian_decay_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay={"method": "gaussian", "catchment_size": 30, "bandwidth": 15},
    )
    assert region_frame.loc["LSOA_Boundary", "accessibility"] == 0.0
    assert region_frame.loc["LSOA_Beyond", "accessibility"] == 0.0
    assert region_frame.loc["LSOA_Beyond", "n_sites_in_catchment"] == 0

    # LSOA_Zero (distance 0) has weight exactly 1.0, so its accessibility
    # equals the site's ratio R directly.
    d0_term = math.exp(-0.5 * (30 / 15) ** 2)
    weight_mid = (math.exp(-0.5 * (15 / 15) ** 2) - d0_term) / (1 - d0_term)
    catchment_demand = 1.0 * 100 + weight_mid * 100
    expected_r = 10 / catchment_demand
    assert region_frame.loc["LSOA_Zero", "accessibility"] == pytest.approx(expected_r)
    assert region_frame.loc["LSOA_Mid", "accessibility"] == pytest.approx(
        weight_mid * expected_r
    )


def test_gaussian_decay_weight_is_monotonically_non_increasing_with_distance(
    gaussian_decay_problem,
):
    region_frame = gaussian_decay_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay={"method": "gaussian", "catchment_size": 30, "bandwidth": 15},
    )
    ordered = region_frame.loc[["LSOA_Zero", "LSOA_Mid", "LSOA_Boundary", "LSOA_Beyond"]]
    accessibility = ordered["accessibility"].tolist()
    assert accessibility == sorted(accessibility, reverse=True)
    # And it's a real effect, not every value coincidentally tied.
    assert accessibility[0] > accessibility[1] > 0


def test_gaussian_decay_unknown_method_raises(sfca_problem):
    with pytest.raises(ValueError, match="Unknown distance_decay method"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", distance_decay={"method": "exponential"}
        )


def test_gaussian_decay_missing_keys_raises(sfca_problem):
    with pytest.raises(ValueError, match="catchment_size"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", distance_decay={"method": "gaussian", "bandwidth": 10}
        )


def test_gaussian_decay_non_positive_catchment_size_raises(sfca_problem):
    with pytest.raises(ValueError, match="catchment_size"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "gaussian", "catchment_size": 0, "bandwidth": 10},
        )


def test_gaussian_decay_non_positive_bandwidth_raises(sfca_problem):
    with pytest.raises(ValueError, match="bandwidth"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "gaussian", "catchment_size": 30, "bandwidth": 0},
        )


# --- Enhanced 2SFCA: continuous power/gravity decay -------------------------


def test_power_decay_matches_hand_computation(sfca_problem):
    """weight(d) = (d/scale)**alpha, truncated beyond catchment_size. With
    scale=1, alpha=-1 this is a plain inverse-distance weight, 1/d. Using
    only Site_1 (LSOA_1=10, LSOA_2=8, LSOA_3=30; demand 100/100/50):

      catchment_demand(Site_1) = 100/10 + 100/8 + 50/30 = 10 + 12.5 + 1.6667 = 24.1667
      R_1 = 10 / 24.1667
      accessibility(LSOA_1) = R_1 / 10, accessibility(LSOA_2) = R_1 / 8,
      accessibility(LSOA_3) = R_1 / 30
    """
    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay={"method": "power", "catchment_size": 30, "scale": 1, "alpha": -1},
        site_names=["Site_1"],
    )

    catchment_demand = 100 / 10 + 100 / 8 + 50 / 30
    expected_r1 = 10 / catchment_demand
    assert region_frame.loc["LSOA_1", "accessibility"] == pytest.approx(expected_r1 / 10)
    assert region_frame.loc["LSOA_2", "accessibility"] == pytest.approx(expected_r1 / 8)
    assert region_frame.loc["LSOA_3", "accessibility"] == pytest.approx(expected_r1 / 30)


def test_power_decay_truncates_beyond_catchment_size(sfca_problem):
    region_frame = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay={"method": "power", "catchment_size": 20, "scale": 1, "alpha": -1},
        site_names=["Site_1"],
    )
    # LSOA_3's cost to Site_1 is 30, beyond catchment_size=20.
    assert region_frame.loc["LSOA_3", "accessibility"] == 0.0
    assert region_frame.loc["LSOA_3", "n_sites_in_catchment"] == 0


def test_power_decay_missing_keys_raises(sfca_problem):
    with pytest.raises(ValueError, match="scale"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "power", "catchment_size": 30, "alpha": -1},
        )


def test_power_decay_non_positive_catchment_size_raises(sfca_problem):
    with pytest.raises(ValueError, match="catchment_size"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "power", "catchment_size": 0, "scale": 1, "alpha": -1},
        )


def test_power_decay_non_positive_scale_raises(sfca_problem):
    with pytest.raises(ValueError, match="scale"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "power", "catchment_size": 30, "scale": 0, "alpha": -1},
        )


def test_power_decay_negative_min_dist_raises(sfca_problem):
    with pytest.raises(ValueError, match="min_dist"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={
                "method": "power", "catchment_size": 30, "scale": 1, "alpha": -1,
                "min_dist": -1,
            },
        )


def test_power_decay_zero_cost_with_zero_min_dist_raises():
    """A cost of exactly 0 combined with min_dist=0 and negative alpha is a
    singularity (1/0); this must raise rather than silently produce inf.
    sfca_problem never has a zero cost, so this needs its own fixture."""
    demand_df = pd.DataFrame({"location_id": ["LSOA_1", "LSOA_2"], "demand": [100, 100]})
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
            "Site_1": [0.0, 8.0],
            "Site_2": [12.0, 30.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    with pytest.raises(ValueError, match="infinite"):
        problem.two_step_floating_catchment(
            supply_col="supply",
            distance_decay={"method": "power", "catchment_size": 30, "scale": 1, "alpha": -1},
        )


# --- Cross-validation against pysal/access's published hospital example ----
#
# Dataset and expected values are from pysal/access's own hospital-accessibility
# test fixture (3 locations, gravity-weighted 2SFCA) -- see THIRD_PARTY_LICENCES.md
# for the licence and exactly what was and wasn't reused.


def _pysal_hospital_problem(costs):
    """`costs` is a dict of {(origin, dest): cost} matching pysal/access's
    test_hospital_example.py scenarios. Locations 1/2/3 double as both
    demand regions and candidate sites, as in the original fixture."""
    demand_df = pd.DataFrame({"loc": [1, 2, 3], "pop": [100, 50, 10]})
    candidate_df = pd.DataFrame(
        {
            "loc": [1, 2, 3],
            "lat": [41.0, 41.1, 41.2],
            "long": [-87.0, -87.1, -87.2],
            "doc": [15, 20, 100],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source": [1, 2, 3],
            1: [costs[(1, 1)], costs[(2, 1)], costs[(3, 1)]],
            2: [costs[(1, 2)], costs[(2, 2)], costs[(3, 2)]],
            3: [costs[(1, 3)], costs[(2, 3)], costs[(3, 3)]],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="pop", location_id_col="loc")
    problem.add_sites(candidate_df, candidate_id_col="loc")
    problem.add_travel_matrix(travel_df, source_col="source")
    return problem


_PYSAL_SCENARIOS = [
    {  # Scenario 0: gridlock
        (1, 1): 1, (2, 2): 1, (3, 3): 1,
        (1, 3): 40, (2, 3): 40, (3, 1): 40, (3, 2): 40,
        (1, 2): 80, (2, 1): 80,
    },
    {  # Scenario 1: faster inbound to location 3
        (1, 1): 1, (2, 2): 1, (3, 3): 1,
        (1, 3): 20, (2, 3): 20, (3, 1): 40, (3, 2): 40,
        (1, 2): 80, (2, 1): 80,
    },
    {  # Scenario 2: faster outbound from location 3
        (1, 1): 1, (2, 2): 1, (3, 3): 1,
        (1, 3): 40, (2, 3): 40, (3, 1): 20, (3, 2): 20,
        (1, 2): 80, (2, 1): 80,
    },
    {  # Scenario 3: symmetric, faster
        (1, 1): 1, (2, 2): 1, (3, 3): 1,
        (1, 3): 20, (2, 3): 20, (3, 1): 20, (3, 2): 20,
        (1, 2): 40, (2, 1): 40,
    },
]

_PYSAL_EXPECTED_ACCESSIBILITY = [
    {1: 0.3314441169802766, 2: 0.5798281320669381, 3: 7.286418169862544},
    {1: 0.43534022087638047, 2: 0.683724235963042, 3: 5.727976611420986},
    {1: 0.3310719131614654, 2: 0.5778577857785778, 3: 7.299991939492457},
    {1: 0.44256839539858406, 2: 0.6667582799658271, 3: 5.7405246461850234},
]


@pytest.mark.parametrize("scenario_index", [0, 1, 2, 3])
def test_matches_pysal_access_hospital_example(scenario_index):
    problem = _pysal_hospital_problem(_PYSAL_SCENARIOS[scenario_index])
    region_frame = problem.two_step_floating_catchment(
        supply_col="doc",
        distance_decay={"method": "power", "catchment_size": 61, "scale": 1, "alpha": -1},
    )

    expected = _PYSAL_EXPECTED_ACCESSIBILITY[scenario_index]
    for loc, expected_accessibility in expected.items():
        assert region_frame.loc[loc, "accessibility"] == pytest.approx(
            expected_accessibility
        )


# --- catchment_size / distance_decay mutual exclusivity ---------------------


def test_both_catchment_size_and_distance_decay_raises(sfca_problem):
    with pytest.raises(ValueError, match="exactly one"):
        sfca_problem.two_step_floating_catchment(
            supply_col="supply", catchment_size=15, distance_decay=[(15, 1.0)]
        )


def test_neither_catchment_size_nor_distance_decay_raises(sfca_problem):
    with pytest.raises(ValueError, match="exactly one"):
        sfca_problem.two_step_floating_catchment(supply_col="supply")


def test_site_problem_and_solution_set_agree_under_distance_decay(sfca_problem):
    result = sfca_problem.solve(p=2, objectives="p_median")

    problem_side = sfca_problem.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=[(10, 1.0), (20, 0.68), (30, 0.22)],
        site_names=["Site_1", "Site_2"],
    )
    solution_side = result.two_step_floating_catchment(
        supply_col="supply",
        distance_decay=[(10, 1.0), (20, 0.68), (30, 0.22)],
        site_names=["Site_1", "Site_2"],
    )

    pd.testing.assert_frame_equal(problem_side, solution_side)


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
