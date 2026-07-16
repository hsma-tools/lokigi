"""
Tests for SolutionComparatorMethodsMixin (lokigi/mixins/solution_comparator_methods.py),
exercised via lokigi.site_solutions.SolutionComparator. Solution sets are
hand-built SiteSolutionSet instances (rather than solved end-to-end) so each
scenario -- overlap, disjoint sets, an empty set -- is fully controlled.
"""

import pandas as pd
import pytest

from lokigi.site_solutions import SiteSolutionSet, SolutionComparator


def _make_set(site_indices, site_names, weighted_average, max_cost):
    df = pd.DataFrame(
        {
            "site_indices": site_indices,
            "site_names": site_names,
            "weighted_average": weighted_average,
            "max": max_cost,
        }
    )
    return SiteSolutionSet(
        solution_df=df, site_problem=None, objectives="p_median", n_sites=2
    )


@pytest.fixture
def comparator():
    set_a = _make_set(
        site_indices=[[0, 1], [0, 2]],
        site_names=[["A", "B"], ["A", "C"]],
        weighted_average=[10.0, 12.0],
        max_cost=[20.0, 22.0],
    )
    set_b = _make_set(
        site_indices=[[1, 2], [0, 1]],
        site_names=[["B", "C"], ["A", "B"]],
        weighted_average=[11.0, 13.0],
        max_cost=[21.0, 23.0],
    )
    return SolutionComparator(set_a, set_b, labels=("Set A", "Set B"))


def test_site_overlap_top_1(comparator):
    result = comparator.site_overlap(top_n=1)

    assert result["common_sites_count"] == 1
    assert result["common_sites_indices"] == [1]
    assert result["unique_to_a"] == [0]
    assert result["unique_to_b"] == [2]
    assert result["jaccard_similarity"] == pytest.approx(1 / 3)


def test_site_overlap_top_2_full_overlap(comparator):
    result = comparator.site_overlap(top_n=2)

    assert result["common_sites_count"] == 3
    assert result["jaccard_similarity"] == pytest.approx(1.0)


def test_site_overlap_disjoint_sets_does_not_divide_by_zero():
    set_c = _make_set([[5, 6]], [["X", "Y"]], [9.0], [19.0])
    set_d = _make_set([[7, 8]], [["Z", "W"]], [9.0], [19.0])
    comparator = SolutionComparator(set_c, set_d)

    result = comparator.site_overlap(top_n=1)

    assert result["common_sites_count"] == 0
    assert result["jaccard_similarity"] == 0


def test_get_metric_summary_difference_column(comparator):
    summary = comparator.get_metric_summary("weighted_average")

    assert summary.loc["mean", "Set A"] == pytest.approx(11.0)
    assert summary.loc["mean", "Set B"] == pytest.approx(12.0)
    assert summary.loc["mean", "difference"] == pytest.approx(-1.0)


def test_find_balanced_solution_jaccard_method(comparator):
    result = comparator.find_balanced_solution(top_n=2, method="jaccard")

    assert result is not None
    solution_a, solution_b = result
    # Both sets contain a solution with site_indices [0, 1] -- the perfect
    # Jaccard match (1.0) should win over any partial overlap.
    assert solution_a["site_indices"] == [0, 1]
    assert solution_b["site_indices"] == [0, 1]


def test_find_balanced_solution_combined_method_does_not_crash(comparator):
    result = comparator.find_balanced_solution(top_n=2, method="combined")
    assert result is not None


def test_find_balanced_solution_returns_none_for_empty_set():
    set_a = _make_set([[0, 1]], [["A", "B"]], [10.0], [20.0])
    set_empty = _make_set([], [], [], [])
    comparator = SolutionComparator(set_a, set_empty)

    result = comparator.find_balanced_solution(top_n=2, method="jaccard")

    assert result is None
