"""
Direct unit tests for the internal utils.py helpers that back solve()'s
combination generation, ranking, selection, and GRASP diversity checks.

None of these were previously tested directly -- only indirectly through
default-path solve() calls -- so branches like required-site filtering,
tie-break ranking, and the priority order in `_select_solution` had no
dedicated coverage.
"""

import pandas as pd
import pytest

from lokigi.utils import (
    _add_rank_column,
    _generate_all_combinations,
    _get_ranking_by_objective,
    _select_solution,
    _too_similar_to_accepted,
)


class _StubSiteProblem:
    """Minimal duck-typed stand-in for SiteProblem -- _generate_all_combinations
    only reads .candidate_sites and ._candidate_sites_required_sites_col."""

    def __init__(self, candidate_sites, required_sites_col=None):
        self.candidate_sites = candidate_sites
        self._candidate_sites_required_sites_col = required_sites_col


# --- _generate_all_combinations ---


def test_generate_all_combinations_basic_count():
    combos = _generate_all_combinations(n_facilities=4, p=2)
    assert len(combos) == 6  # C(4,2)


def test_generate_all_combinations_filters_by_required_site():
    candidates = pd.DataFrame(
        {"site_id": ["A", "B", "C", "D"], "required": ["yes", "no", "no", "no"]}
    )
    stub = _StubSiteProblem(candidates, required_sites_col="required")

    combos = _generate_all_combinations(n_facilities=4, p=2, site_problem=stub)

    assert len(combos) == 3
    assert all(0 in combo for combo in combos)


def test_generate_all_combinations_returns_empty_when_required_sites_exceed_p():
    """Two required sites can never both fit into a p=1 solution -- the
    function silently returns an empty list rather than raising, which this
    test pins as the current (accepted) behaviour."""
    candidates = pd.DataFrame(
        {"site_id": ["A", "B", "C", "D"], "required": ["yes", "yes", "no", "no"]}
    )
    stub = _StubSiteProblem(candidates, required_sites_col="required")

    combos = _generate_all_combinations(n_facilities=4, p=1, site_problem=stub)

    assert combos == []


def test_generate_all_combinations_force_include_indices():
    combos = _generate_all_combinations(n_facilities=4, p=2, force_include_indices=[1])

    assert len(combos) == 3
    assert all(1 in combo for combo in combos)


# --- _add_rank_column ---


def test_add_rank_column_dense_rank_with_tie_ascending():
    df = pd.DataFrame({"score": [10, 10, 5], "tiebreak": [1, 1, 2]})

    ranked = _add_rank_column(df, score_col="score", tiebreaker_col="tiebreak", ascending=True)

    # The two identical (score, tiebreak) rows share rank 2; the lone lowest
    # score gets rank 1 -- dense ranking, no gaps.
    ranks_by_score = dict(zip(ranked["score"], ranked["solution_rank"]))
    assert ranks_by_score[5] == 1
    assert (ranked[ranked["score"] == 10]["solution_rank"] == 2).all()


def test_add_rank_column_descending_direction():
    df = pd.DataFrame({"score": [10, 10, 5], "tiebreak": [1, 1, 2]})

    ranked = _add_rank_column(
        df, score_col="score", tiebreaker_col="tiebreak", ascending=[False, True]
    )

    assert ranked.iloc[0]["score"] == 10
    assert ranked.iloc[-1]["score"] == 5


def test_add_rank_column_puts_rank_as_first_column():
    df = pd.DataFrame({"score": [3, 1, 2], "tiebreak": [0, 0, 0]})
    ranked = _add_rank_column(df, score_col="score", tiebreaker_col="tiebreak")
    assert ranked.columns[0] == "solution_rank"


# --- _select_solution ---


@pytest.fixture
def solution_df():
    return pd.DataFrame(
        {
            "solution_rank": [1, 2, 3],
            "weighted_average": [10.0, 5.0, 8.0],
            "site_indices": [[0, 1], [0, 2], [1, 2]],
            "site_names": [["A", "B"], ["A", "C"], ["B", "C"]],
        }
    )


def test_select_solution_by_site_indices_takes_priority(solution_df):
    result = _select_solution(
        solution_df, rank_on="weighted_average", solution_rank=1, site_indices=[2, 1]
    )
    assert result.iloc[0]["site_names"] == ["B", "C"]


def test_select_solution_by_site_names_when_no_indices_given(solution_df):
    result = _select_solution(solution_df, site_names=["C", "A"])
    assert result.iloc[0]["site_indices"] == [0, 2]


def test_select_solution_by_rank_on_and_solution_rank(solution_df):
    """rank_on='weighted_average' should sort ascending first (5.0 is lowest,
    so solution_rank=1 after re-sorting is the row with weighted_average 5.0)."""
    result = _select_solution(solution_df, rank_on="weighted_average", solution_rank=1)
    assert result.iloc[0]["weighted_average"] == 5.0


def test_select_solution_no_matching_site_indices_raises(solution_df):
    with pytest.raises(ValueError, match="No solution found with site_indices"):
        _select_solution(solution_df, site_indices=[0, 1, 2])


def test_select_solution_no_matching_site_names_raises(solution_df):
    with pytest.raises(ValueError, match="No solution found with site_names"):
        _select_solution(solution_df, site_names=["X", "Y"])


@pytest.mark.parametrize("bad_rank", [0, 4])
def test_select_solution_out_of_range_rank_raises(solution_df, bad_rank):
    with pytest.raises(ValueError, match="solution_rank must be between 1 and 3"):
        _select_solution(solution_df, solution_rank=bad_rank)


# --- _too_similar_to_accepted ---


def test_too_similar_exact_match_branch():
    assert _too_similar_to_accepted({0, 1}, [{0, 1}], min_jaccard_distance=0.0) is True
    assert _too_similar_to_accepted({0, 1}, [{0, 2}], min_jaccard_distance=0.0) is False


def test_too_similar_jaccard_distance_threshold():
    # intersection=1, p=2 -> union = 2*2-1 = 3, similarity=1/3, distance=2/3
    assert _too_similar_to_accepted({0, 1}, [{1, 2}], min_jaccard_distance=0.5) is False
    assert _too_similar_to_accepted({0, 1}, [{1, 2}], min_jaccard_distance=0.7) is True


def test_too_similar_negative_distance_raises():
    with pytest.raises(ValueError, match="cannot be negative"):
        _too_similar_to_accepted({0, 1}, [], min_jaccard_distance=-0.1)


# --- _get_ranking_by_objective ---


@pytest.mark.parametrize(
    "objective,expected_column",
    [
        ("p_median", "weighted_average"),
        ("hybrid_p_median", "weighted_average"),
        ("simple_p_median", "unweighted_average"),
        ("hybrid_simple_p_median", "unweighted_average"),
        ("p_center", "max"),
        ("mclp", "proportion_within_coverage_threshold"),
    ],
)
def test_get_ranking_by_objective_maps_every_supported_objective(
    objective, expected_column
):
    assert _get_ranking_by_objective(objective) == expected_column
