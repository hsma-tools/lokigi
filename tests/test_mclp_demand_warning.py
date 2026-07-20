"""
Tests pinning that the "no demand data" warning correctly exempts mclp,
regardless of whether the caller passes objectives as a plain string or
as a list.

solve() resolves the caller's `objectives` argument (str or list of str)
into a single `objective` string early on, but the missing-demand warning
check further down compared the ORIGINAL `objectives` parameter to the
literal string "mclp" -- so `objectives="mclp"` correctly matched and
suppressed the warning, but `objectives=["mclp"]` never could (a list is
never equal to a string), and the warning fired anyway for the one
objective it was supposed to exempt.
"""

import warnings

import pytest

import lokigi


@pytest.fixture
def problem_without_demand():
    """Sites and travel matrix, but demand deliberately not added -- the
    condition that triggers solve()'s auto-equal-demand warning."""
    import pandas as pd

    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.1, 51.2], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_A": [10.0, 20.0],
            "Site_B": [20.0, 10.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def _demand_warnings(caught):
    return [w for w in caught if "No demand data was provided" in str(w.message)]


def test_mclp_as_a_list_is_still_exempt_from_the_demand_warning(
    problem_without_demand,
):
    """The bug: objectives=["mclp"] used to still trigger the warning
    because a list is never equal to the string "mclp"."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem_without_demand.solve(
            p=1,
            objectives=["mclp"],
            search_strategy="brute-force",
            show_progress=False,
            threshold_for_coverage=15,
        )

    assert _demand_warnings(caught) == []


def test_mclp_as_a_string_is_exempt_from_the_demand_warning(problem_without_demand):
    """Sanity check / regression: the plain-string case already worked
    before this fix and must keep working."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem_without_demand.solve(
            p=1,
            objectives="mclp",
            search_strategy="brute-force",
            show_progress=False,
            threshold_for_coverage=15,
        )

    assert _demand_warnings(caught) == []


def test_non_mclp_objectives_still_warn_about_missing_demand(problem_without_demand):
    """Sanity check that the exemption is specific to mclp: a non-exempt
    objective (as a plain string) must still warn -- if this stops
    holding, the fixture no longer proves the exemption is selective."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem_without_demand.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )

    assert len(_demand_warnings(caught)) == 1


def test_non_mclp_objectives_as_a_list_still_warn_about_missing_demand(
    problem_without_demand,
):
    """The multi-objective list path (only the first element is used,
    with its own separate warning) must still warn about missing demand
    when that first element isn't mclp."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem_without_demand.solve(
            p=1,
            objectives=["p_median"],
            search_strategy="brute-force",
            show_progress=False,
        )

    assert len(_demand_warnings(caught)) == 1
