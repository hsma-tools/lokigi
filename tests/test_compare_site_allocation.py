"""Tests for `SolutionComparator.compare_site_allocation()`.

Puts two solutions' `site_allocation_summary()` results side by side --
e.g. a 2-site solution against a 3-site one, to see whether the new site
captured a meaningful share of demand or took over from an existing site.

Uses `five_site_problem` (see `tests/conftest.py`), whose p=2 optimum is
{Site_1, Site_3} (weighted_average-best pair) -- adding Site_2 to reach
p=3 turns out to capture nothing (see `test_site_allocation_summary.py`'s
"headline regression guard"), so this is a real instance of the exact
scenario the method exists to surface, not a contrived one.
"""

import pandas as pd
import pytest

from lokigi.site_solutions import SolutionComparator


def test_new_site_that_captures_nothing_is_nan_not_zero(five_site_problem):
    sols_2 = five_site_problem.solve(p=2)
    sols_3 = five_site_problem.solve(p=3)

    comparator = SolutionComparator(sols_2, sols_3, labels=("2 sites", "3 sites"))
    result = comparator.compare_site_allocation(
        config_a={"solution_rank": 1},
        config_b={"site_names": ["Site_1", "Site_2", "Site_3"]},
    )

    assert list(result.index) == ["Site_1", "Site_2", "Site_3"]
    # Site_2 isn't part of the 2-site solution at all -- NaN, not opened.
    assert pd.isna(result.loc["Site_2", "2 sites"])
    # But it IS opened in the 3-site solution, just closest to nothing.
    assert result.loc["Site_2", "3 sites"] == 0.0
    # Sites 1 and 3 are unaffected by Site_2's addition -- same proportions
    # in both solutions, difference exactly 0.
    assert result.loc["Site_1", "difference"] == pytest.approx(0.0)
    assert result.loc["Site_3", "difference"] == pytest.approx(0.0)


def test_difference_direction_matches_get_metric_summary(five_site_problem):
    """`difference` = labels[0] - labels[1], the same direction
    `get_metric_summary` already uses, for consistency within the class."""
    sols_2 = five_site_problem.solve(p=2)
    sols_3 = five_site_problem.solve(p=3)

    comparator = SolutionComparator(sols_2, sols_3, labels=("2 sites", "3 sites"))
    result = comparator.compare_site_allocation(
        config_a={"solution_rank": 1},
        config_b={"site_names": ["Site_1", "Site_2", "Site_3"]},
    )

    for site in ("Site_1", "Site_3"):
        expected = result.loc[site, "2 sites"] - result.loc[site, "3 sites"]
        assert result.loc[site, "difference"] == pytest.approx(expected)


def test_by_regions_argument_is_forwarded(five_site_problem):
    sols_2 = five_site_problem.solve(p=2)
    sols_3 = five_site_problem.solve(p=3)

    comparator = SolutionComparator(sols_2, sols_3, labels=("2 sites", "3 sites"))
    result = comparator.compare_site_allocation(
        by="regions",
        config_a={"solution_rank": 1},
        config_b={"site_names": ["Site_1", "Site_2", "Site_3"]},
    )

    # Demand is uniform in this fixture, so by="regions" gives the same
    # numbers as by="demand" -- just confirms the argument reaches through.
    assert result.loc["Site_2", "3 sites"] == 0.0
