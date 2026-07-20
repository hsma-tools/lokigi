"""
Tests pinning that max_value_cutoff (the hybrid objectives' "safety net")
is honoured by EVERY search strategy, not just brute-force.

These pin the fix for a silent-constraint-violation bug: solve() insisted
hybrid objectives provide max_value_cutoff, but the dispatcher only
forwarded it to `_brute_force` -- `_greedy` and `_grasp` didn't even have
the parameter. Since the hybrid objectives' ranking columns equal their
non-hybrid counterparts, hybrid_p_median + greedy/grasp silently behaved
exactly like plain p_median: solutions violating the promised worst-case
travel guarantee were returned, ranked normally, with no warning.

The fixed behaviour:

- Greedy applies the cutoff when choosing the final site (with fewer than
  p sites the worst-case travel is usually still shrinking, so only the
  final step is filtered), and raises a clear ValueError if no feasible
  completion of its earlier choices exists.
- GRASP rejects finished (post-local-search) solutions that violate the
  cutoff, costing an attempt like a diversity reject.
- A cutoff strict enough to rule out every combination now raises a clear
  "No feasible solutions" ValueError from the dispatcher for any strategy,
  instead of crashing with a cryptic KeyError on an empty DataFrame.
"""

import pandas as pd
import pytest

import lokigi


@pytest.fixture
def greedy_trap_problem():
    """
    Three sites, four equal-demand LSOAs, engineered so unconstrained
    greedy walks into a cutoff violation that a different final site
    would avoid:

    - Site_A: [10, 20, 20, 30] -> weighted_average 20.00, max 30
    - Site_B: [30,  6,  6, 40] -> weighted_average 20.50, max 40
    - Site_C: [25, 25, 25, 12] -> weighted_average 21.75, max 25

    Step 1 always picks Site_A (best weighted_average). At step 2:

    - {A, B}: mins [10, 6, 6, 30] -> weighted_average 13.0, max 30
    - {A, C}: mins [10, 20, 20, 12] -> weighted_average 15.5, max 20

    Unconstrained greedy prefers {A, B} (13.0 < 15.5), which violates a
    cutoff of 20 (LSOA_4 is left 30 away); {A, C} satisfies it. The
    final-step filter must steer greedy to {A, C}.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 20.0, 20.0, 30.0],
            "Site_B": [30.0, 6.0, 6.0, 40.0],
            "Site_C": [25.0, 25.0, 25.0, 12.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


# --- greedy ---


def test_unconstrained_greedy_walks_into_the_trap(greedy_trap_problem):
    """Sanity check that the fixture discriminates: without a cutoff,
    greedy picks {A, B} (max 30). If this stops holding, the cutoff tests
    below no longer prove anything."""
    result = greedy_trap_problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )
    assert result.solution_df.iloc[0]["site_names"] == ["Site_A", "Site_B"]
    assert result.solution_df.iloc[0]["max"] == 30.0


@pytest.mark.parametrize("objective", ["hybrid_p_median", "hybrid_simple_p_median"])
def test_greedy_final_step_respects_cutoff(greedy_trap_problem, objective):
    """With max_value_cutoff=20, greedy must reject {A, B} (max 30) at the
    final step and return {A, C} (max 20) instead of silently violating
    the guarantee."""
    result = greedy_trap_problem.solve(
        p=2,
        objectives=objective,
        search_strategy="greedy",
        show_progress=False,
        max_value_cutoff=20,
    )

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_A", "Site_C"]
    assert best["max"] == 20.0
    assert all(result.solution_df["max"] <= 20)


def test_greedy_cutoff_applies_when_p_is_one(greedy_trap_problem):
    """With p=1 the first step IS the final step: Site_A (best
    weighted_average, max 30) and Site_B (max 40) must be filtered out by
    a cutoff of 25, leaving Site_C (max 25) as the only feasible single
    site."""
    result = greedy_trap_problem.solve(
        p=1,
        objectives="hybrid_p_median",
        search_strategy="greedy",
        show_progress=False,
        max_value_cutoff=25,
    )

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_C"]
    assert best["max"] == 25.0


def test_greedy_raises_clearly_when_no_feasible_completion_exists(
    greedy_trap_problem,
):
    """A cutoff of 15 rules out both completions of greedy's fixed first
    pick ({A, B} max 30, {A, C} max 20), so greedy must fail loudly and
    point at strategies that can still succeed."""
    with pytest.raises(ValueError, match="max_value_cutoff=15"):
        greedy_trap_problem.solve(
            p=2,
            objectives="hybrid_p_median",
            search_strategy="greedy",
            show_progress=False,
            max_value_cutoff=15,
        )


# --- grasp ---


def test_grasp_only_returns_solutions_within_cutoff(five_site_problem):
    """On the adversarial five-site fixture, unconstrained GRASP's pool for
    hybrid_p_median includes a solution with max 24; with a cutoff of 20,
    every returned solution must satisfy it (only combinations with max 17
    are feasible)."""
    result = five_site_problem.solve(
        p=2,
        objectives="hybrid_p_median",
        search_strategy="grasp",
        show_progress=False,
        max_value_cutoff=20,
        grasp_num_solutions=2,
        grasp_max_attempts=50,
        random_seed=42,
    )

    assert len(result.solution_df) >= 1
    assert all(result.solution_df["max"] <= 20)


def test_grasp_raises_clearly_when_cutoff_is_infeasible(five_site_problem):
    """No two-site combination on the five-site fixture achieves max <= 5,
    so GRASP must exhaust its attempts and surface a clear error rather
    than crashing with a KeyError on an empty result set."""
    with pytest.raises(ValueError, match="No feasible solutions"):
        five_site_problem.solve(
            p=2,
            objectives="hybrid_p_median",
            search_strategy="grasp",
            show_progress=False,
            max_value_cutoff=5,
            grasp_num_solutions=2,
        )


# --- brute force: infeasible cutoff now errors clearly instead of crashing ---


def test_brute_force_raises_clearly_when_cutoff_is_infeasible(five_site_problem):
    """Previously a cutoff that filtered out every combination crashed
    with KeyError('weighted_average') when ranking the empty DataFrame."""
    with pytest.raises(ValueError, match="No feasible solutions"):
        five_site_problem.solve(
            p=2,
            objectives="hybrid_p_median",
            search_strategy="brute-force",
            show_progress=False,
            max_value_cutoff=5,
        )
