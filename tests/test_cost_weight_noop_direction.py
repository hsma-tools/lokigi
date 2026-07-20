"""
Tests pinning that greedy and GRASP still sort/compare in the correct
direction when cost weighting is REQUESTED but silently NO-OPS -- for
both a maximising objective (mclp) and a minimising one (p_median).

_apply_cost_weighting (utils.py) blends the primary ranking column with
site cost onto a lower-is-better "composite_score" scale -- but only when
it actually has usable cost data. It no-ops (returns the ORIGINAL
ranking column, on the objective's own natural scale, unchanged) when
e.g. every candidate's total_cost is NaN (add_sites(..., cost_col=...,
allow_missing_cost=True) with no cost values ever provided).

Four call sites -- _greedy's final sort, _grasp's RCL construction and
cost-weighted local search (mixins/site_solvers.py), and site.py's
shared final cross-strategy sort (_solve_pmedian_pcenter_mclp_problem,
which runs after brute-force/greedy/grasp all return) -- assumed that
"a positive cost weight was requested" implied "blending happened,
therefore the score is lower-is-better", and hardcoded the
ascending/minimising direction accordingly. That assumption breaks when
_apply_cost_weighting no-ops: for mclp (higher-is-better coverage
proportion), treating its raw score as lower-is-better inverted the
search, so greedy/GRASP/the final sort picked the WORST-coverage
combinations as "best".

The site.py final-sort instance is only visible when there's more than
one candidate solution to actually sort -- a single-solution result
(greedy, or GRASP with num_solutions=1) makes "sort direction" moot, so
it stayed hidden behind the earlier per-solver tests until specifically
tested with brute-force's full (unpruned) output and GRASP with
num_solutions > 1.

Test design: rather than asserting the search reaches the GLOBAL optimum
(brute-force's best), which greedy and a single GRASP construction
attempt aren't guaranteed to reach regardless of this fix -- that's a
separate, pre-existing property of those search strategies, not what
this bug is about -- these tests assert the more precise invariant that
actually follows from _apply_cost_weighting's documented no-op contract:
a request that no-ops must behave IDENTICALLY to not requesting cost
weighting at all, same random_seed included. That's exactly what would
differ if the direction fallback were wrong, and it isn't dependent on
whether the search strategy itself happens to find the global optimum.
"""

import numpy as np
import pandas as pd
import pytest

import lokigi


@pytest.fixture
def problem_with_noop_cost():
    """Same adversarial travel data as `five_site_problem` (the combination
    with the best weighted_average is NOT the one with the best mclp
    coverage), plus a cost_col registered with every site's cost missing
    (allow_missing_cost=True) -- so weights={"cost": ...} passes solve()'s
    validation (a cost_col IS configured) but _apply_cost_weighting no-ops
    at evaluation time (every candidate's total_cost is NaN). Used for
    both the mclp (maximising) and p_median (minimising) tests below."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"],
            "lat": [51.1, 51.2, 51.3, 51.4, 51.5],
            "long": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "build_cost": [np.nan] * 5,
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [38.0, 18.0, 10.0, 28.0],
            "Site_2": [25.0, 40.0, 11.0, 31.0],
            "Site_3": [24.0, 13.0, 29.0, 13.0],
            "Site_4": [29.0, 16.0, 17.0, 16.0],
            "Site_5": [17.0, 15.0, 17.0, 36.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(
        candidate_df,
        candidate_id_col="site_id",
        cost_col="build_cost",
        allow_missing_cost=True,
    )
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def _assert_noop_cost_matches_no_cost(problem, solve_kwargs):
    """Solve once with a cost weight that's requested but no-ops (NaN
    costs), and once with cost omitted entirely -- same everything else,
    including random_seed for GRASP. If the no-op direction fallback is
    correct, these must produce the identical top solution."""
    with_noop_cost = problem.solve(
        weights={"demand": 0.5, "cost": 0.5}, **solve_kwargs
    )
    without_cost = problem.solve(weights={"demand": 1.0}, **solve_kwargs)

    assert (
        with_noop_cost.solution_df.iloc[0]["site_names"]
        == without_cost.solution_df.iloc[0]["site_names"]
    )
    assert with_noop_cost.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        without_cost.solution_df.iloc[0]["weighted_average"]
    )


# --- mclp (maximising) ---


def test_greedy_mclp_noop_cost_matches_no_cost_weight(problem_with_noop_cost):
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="mclp",
            search_strategy="greedy",
            show_progress=False,
            threshold_for_coverage=15,
        ),
    )


def test_grasp_construction_mclp_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    """local_search_chance=0.0 isolates GRASP's construction/RCL phase --
    a mismatch here is attributable solely to the RCL sort-direction fix,
    since local search never runs."""
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="mclp",
            search_strategy="grasp",
            show_progress=False,
            threshold_for_coverage=15,
            grasp_num_solutions=1,
            grasp_max_attempts=30,
            grasp_local_search_chance=0.0,
            random_seed=1,
        ),
    )


def test_grasp_with_local_search_mclp_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    """local_search_chance=1.0 forces the cost-weighted local-search swap
    comparison to run every attempt -- if its direction were still wrong,
    it could swap a matching-so-far solution away to a different one."""
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="mclp",
            search_strategy="grasp",
            show_progress=False,
            threshold_for_coverage=15,
            grasp_num_solutions=1,
            grasp_max_attempts=30,
            grasp_local_search_chance=1.0,
            random_seed=1,
        ),
    )


# --- p_median (minimising) ---
#
# For a minimising objective, the no-op fallback (`is_minimization`) and
# the blended case (always "lower is better") agree, so this direction
# was never actually wrong in practice -- these tests exist to confirm
# that positively, rather than only inferring it from the mclp case above
# and the fact the full suite doesn't regress.


def test_greedy_p_median_noop_cost_matches_no_cost_weight(problem_with_noop_cost):
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="p_median",
            search_strategy="greedy",
            show_progress=False,
        ),
    )


def test_grasp_construction_p_median_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="p_median",
            search_strategy="grasp",
            show_progress=False,
            grasp_num_solutions=1,
            grasp_max_attempts=30,
            grasp_local_search_chance=0.0,
            random_seed=1,
        ),
    )


def test_grasp_with_local_search_p_median_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="p_median",
            search_strategy="grasp",
            show_progress=False,
            grasp_num_solutions=1,
            grasp_max_attempts=30,
            grasp_local_search_chance=1.0,
            random_seed=1,
        ),
    )


# --- site.py's shared final cross-strategy sort ---
#
# This is a SEPARATE bug site from the three above: it runs once, after
# brute-force/greedy/grasp all return, and re-sorts/ranks WHATEVER list of
# solutions that strategy produced. It only matters when there's more than
# one solution to sort -- greedy and single-solution GRASP calls (used
# throughout the tests above) never exercise it, since sorting a 1-row
# result is a no-op regardless of direction. Brute-force with no
# brute_force_keep_best_n/worst_n (returns every combination) and GRASP
# with grasp_num_solutions > 1 both genuinely exercise it.


def test_brute_force_full_results_mclp_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    """Brute-force with no keep_best_n/worst_n returns all 10
    combinations -- enough for the final sort's direction to actually
    matter."""
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="mclp",
            search_strategy="brute-force",
            show_progress=False,
            threshold_for_coverage=15,
        ),
    )


def test_grasp_multiple_solutions_mclp_noop_cost_matches_no_cost_weight(
    problem_with_noop_cost,
):
    """grasp_num_solutions=3 (with local search enabled, the default
    grasp_local_search_chance=0.8) gives the final sort multiple distinct
    solutions to rank -- GRASP's own construction/local-search direction
    is already proven identical between the noop-cost and no-cost calls
    by the tests above, so any remaining mismatch here is attributable to
    the final sort specifically."""
    _assert_noop_cost_matches_no_cost(
        problem_with_noop_cost,
        dict(
            p=2,
            objectives="mclp",
            search_strategy="grasp",
            show_progress=False,
            threshold_for_coverage=15,
            grasp_num_solutions=3,
            grasp_max_attempts=30,
            random_seed=1,
        ),
    )
