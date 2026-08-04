"""
Tests pinning that required_sites_col (sites that MUST appear in every
solution) is honoured by EVERY search strategy, not just brute-force.

These pin the fix for a silent-constraint-violation bug: required sites
were only enforced inside `_generate_all_combinations` (utils.py), which
brute-force and greedy call -- but GRASP builds solutions incrementally
from `range(total_n_sites)` and never calls it. GRASP therefore ignored
required_sites_col completely: its randomised construction was free to
omit required sites, and even when one was picked by chance, the 1-opt
local search would happily swap it back out for anything better.

The fixed behaviour:

- GRASP seeds every construction with the required sites and only fills
  the remaining p - n_required slots.
- GRASP's local search never proposes swapping a required site out (in
  both the plain and cost-weighted branches).
- More required sites than p raises a clear ValueError up front.
- The attempt budget's "total combinations" reflects only the free slots.
"""

import pandas as pd
import pytest

import lokigi


def _problem_with_terrible_required_site(mark_required=True, also_require=()):
    """
    Five sites, four equal-demand LSOAs. Site_Req is uniformly terrible
    (90 minutes from everywhere), so no search would ever pick it
    voluntarily -- and local search would always want to swap it out.
    That makes it the sharpest possible probe: Site_Req appears in a
    solution if and only if the required-sites constraint is enforced.

    With Site_Req required and p=2, only 4 combinations are feasible, and
    each scores the partner site's own travel times (Site_Req's 90s never
    win a min). Best partner by weighted_average: Site_4 (19.5), then
    Site_3 (20.75), Site_1 (23.5), Site_2 (26.75).

    Unconstrained, the p=2 optimum is {Site_1, Site_3} (15.0).

    `also_require` marks additional site_ids as required, for exercising
    the multiple-required-sites paths.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    site_ids = ["Site_1", "Site_2", "Site_3", "Site_4", "Site_Req"]
    candidate_df = pd.DataFrame(
        {
            "site_id": site_ids,
            "lat": [51.1, 51.2, 51.3, 51.4, 51.5],
            "long": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "must_build": [
                "yes"
                if (site in also_require or (site == "Site_Req" and mark_required))
                else "no"
                for site in site_ids
            ],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [38.0, 18.0, 10.0, 28.0],
            "Site_2": [25.0, 40.0, 11.0, 31.0],
            "Site_3": [24.0, 13.0, 29.0, 13.0],
            "Site_4": [29.0, 16.0, 17.0, 16.0],
            "Site_Req": [90.0, 90.0, 90.0, 90.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(
        candidate_df, candidate_id_col="site_id", required_sites_col="must_build"
    )
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


# --- sanity: the fixture discriminates ---


def test_unconstrained_grasp_omits_the_terrible_site():
    """With no site marked required, GRASP must never return Site_Req --
    if this stops holding, the constraint tests below no longer prove the
    constraint (rather than luck) put Site_Req in the solutions."""
    problem = _problem_with_terrible_required_site(mark_required=False)
    result = problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
        random_seed=42,
    )

    for site_names in result.solution_df["site_names"]:
        assert "Site_Req" not in site_names


# --- GRASP must honour the required site ---


def test_grasp_includes_required_site_in_every_solution():
    problem = _problem_with_terrible_required_site()
    result = problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
        random_seed=42,
    )

    assert len(result.solution_df) >= 1
    for site_names in result.solution_df["site_names"]:
        assert "Site_Req" in site_names


def test_grasp_local_search_never_swaps_out_a_required_site():
    """Site_Req is strictly worse than every alternative, so any 1-opt
    swap replacing it improves the score -- if local search were allowed
    to touch required sites it would ALWAYS remove Site_Req. Forcing
    local search on every attempt makes this deterministic."""
    problem = _problem_with_terrible_required_site()
    result = problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=4,
        grasp_max_attempts=50,
        grasp_local_search_chance=1.0,
        random_seed=42,
    )

    assert len(result.solution_df) >= 1
    for site_names in result.solution_df["site_names"]:
        assert "Site_Req" in site_names


def test_grasp_finds_the_best_solution_containing_the_required_site():
    """The constrained optimum is {Site_4, Site_Req} (weighted_average
    19.5) -- GRASP's top-ranked solution must match brute-force's."""
    problem = _problem_with_terrible_required_site()
    brute_force = problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )
    grasp = problem.solve(
        p=2,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
        grasp_max_attempts=50,
        random_seed=42,
    )

    assert sorted(brute_force.solution_df.iloc[0]["site_names"]) == [
        "Site_4",
        "Site_Req",
    ]
    assert sorted(grasp.solution_df.iloc[0]["site_names"]) == ["Site_4", "Site_Req"]
    assert grasp.solution_df.iloc[0]["weighted_average"] == pytest.approx(
        brute_force.solution_df.iloc[0]["weighted_average"]
    )


def test_grasp_raises_clearly_when_required_sites_exceed_p():
    """Two required sites cannot fit in a p=1 solution: GRASP must fail
    fast with a clear message rather than looping or silently dropping
    one of them."""
    problem = _problem_with_terrible_required_site(
        mark_required=False, also_require=("Site_1", "Site_2")
    )

    with pytest.raises(ValueError, match="required"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="grasp",
            show_progress=False,
        )


# --- regression: the shared helper must not change existing strategies ---


def test_brute_force_still_only_returns_combinations_with_the_required_site():
    problem = _problem_with_terrible_required_site()
    result = problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    # 4 of the 10 possible pairs contain Site_Req
    assert len(result.solution_df) == 4
    for site_names in result.solution_df["site_names"]:
        assert "Site_Req" in site_names


def test_greedy_still_honours_a_single_required_site():
    problem = _problem_with_terrible_required_site()
    result = problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )

    best = result.solution_df.iloc[0]
    assert "Site_Req" in best["site_names"]
    assert sorted(best["site_names"]) == ["Site_4", "Site_Req"]


# --- greedy with MULTIPLE required sites ---
#
# Greedy grows solutions one site at a time, but every size-i combination
# is also filtered to contain ALL required sites -- so with two or more
# required sites, the early steps (i < n_required) had no valid
# combinations at all and greedy crashed with KeyError('weighted_average')
# on the empty DataFrame. The fix seeds greedy's build with the required
# sites and starts the loop at that size.


def test_greedy_with_two_required_sites_returns_the_best_completion():
    """With Site_Req and Site_2 required and p=3, the three possible
    completions are hand-checkable (Site_Req's 90s never win a min, so
    each scores the elementwise mins of Site_2 and the free site):

      + Site_1: mins [25, 18, 10, 28] -> weighted_average 20.25
      + Site_3: mins [24, 13, 11, 13] -> weighted_average 15.25
      + Site_4: mins [25, 16, 11, 16] -> weighted_average 17.00

    Greedy's single free choice sees all three completions at once, so it
    must find the true constrained optimum here, matching brute-force."""
    problem = _problem_with_terrible_required_site(also_require=("Site_2",))
    greedy = problem.solve(
        p=3, objectives="p_median", search_strategy="greedy", show_progress=False
    )
    brute_force = problem.solve(
        p=3, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    best = greedy.solution_df.iloc[0]
    assert sorted(best["site_names"]) == ["Site_2", "Site_3", "Site_Req"]
    assert best["weighted_average"] == pytest.approx(15.25)
    assert sorted(brute_force.solution_df.iloc[0]["site_names"]) == sorted(
        best["site_names"]
    )


def test_greedy_when_p_equals_the_number_of_required_sites():
    """With every slot taken by a required site there is nothing to
    choose: greedy must return exactly the required set (elementwise mins
    of Site_2 and Site_Req = Site_2's own times, weighted_average 26.75)
    rather than crashing."""
    problem = _problem_with_terrible_required_site(also_require=("Site_2",))
    result = problem.solve(
        p=2, objectives="p_median", search_strategy="greedy", show_progress=False
    )

    best = result.solution_df.iloc[0]
    assert sorted(best["site_names"]) == ["Site_2", "Site_Req"]
    assert best["weighted_average"] == pytest.approx(26.75)


def test_brute_force_raises_clearly_when_required_sites_exceed_p():
    """Two required sites cannot fit in a p=1 solution: brute-force must
    fail fast with a clear message rather than falling through to the
    generic, misleadingly-worded 'No feasible solutions' guard."""
    problem = _problem_with_terrible_required_site(
        mark_required=False, also_require=("Site_1", "Site_2")
    )

    with pytest.raises(ValueError, match="required"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )


def test_greedy_raises_clearly_when_required_sites_exceed_p():
    problem = _problem_with_terrible_required_site(also_require=("Site_2",))

    with pytest.raises(ValueError, match="required"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="greedy",
            show_progress=False,
        )


def test_grasp_honours_two_required_sites():
    """GRASP's construction seeding must handle any number of required
    sites, not just one: every returned solution contains both."""
    problem = _problem_with_terrible_required_site(also_require=("Site_2",))
    result = problem.solve(
        p=3,
        objectives="p_median",
        search_strategy="grasp",
        show_progress=False,
        grasp_num_solutions=3,
        grasp_max_attempts=50,
        random_seed=42,
    )

    assert len(result.solution_df) >= 1
    for site_names in result.solution_df["site_names"]:
        assert "Site_Req" in site_names
        assert "Site_2" in site_names


# --- additional_site_names (site_names minus the required sites) ---


def test_additional_site_names_present_and_correct_with_one_required_site():
    """4 candidates, Site_Req required, p=2 -- every solution has exactly
    one free slot, so additional_site_names is that single partner site."""
    problem = _problem_with_terrible_required_site()
    result = problem.solve(p=2, objectives="p_median", show_progress=False)

    assert "additional_site_names" in result.solution_df.columns
    for _, row in result.solution_df.iterrows():
        non_required = [n for n in row["site_names"] if n != "Site_Req"]
        assert row["additional_site_names"] == non_required


def test_additional_site_names_correct_with_two_required_sites():
    problem = _problem_with_terrible_required_site(also_require=("Site_2",))
    result = problem.solve(p=3, objectives="p_median", show_progress=False)

    for _, row in result.solution_df.iterrows():
        assert "Site_Req" not in row["additional_site_names"]
        assert "Site_2" not in row["additional_site_names"]
        assert set(row["additional_site_names"]) == set(row["site_names"]) - {
            "Site_Req",
            "Site_2",
        }


def test_additional_site_names_absent_without_required_sites():
    problem = _problem_with_terrible_required_site(mark_required=False)
    result = problem.solve(p=2, objectives="p_median", show_progress=False)
    assert "additional_site_names" not in result.solution_df.columns


def test_additional_site_names_in_show_solutions_summary():
    problem = _problem_with_terrible_required_site()
    result = problem.solve(p=2, objectives="p_median", show_progress=False)
    df = result.show_solutions_summary()

    assert "Additional sites chosen" in df.columns
    raw_row = result.show_solutions(rounding=None).iloc[0]
    assert df["Additional sites chosen"].iloc[0] == ", ".join(
        raw_row["additional_site_names"]
    )


def test_additional_sites_chosen_absent_from_summary_without_required_sites():
    problem = _problem_with_terrible_required_site(mark_required=False)
    result = problem.solve(p=2, objectives="p_median", show_progress=False)
    df = result.show_solutions_summary()
    assert "Additional sites chosen" not in df.columns
