"""
Tests pinning how solve()/evaluation handle equity data that does not cover
every demand location.

These pin the fix for a silent-corruption bug: the per-solution equity
merge in evaluate_single_solution_single_objective (site.py) had no `how=`
argument, so it defaulted to an INNER join and any demand location missing
from the equity data was silently dropped from the entire evaluation --
shrinking every metric (max, weighted averages, coverage) for every solve,
even when equity wasn't being weighted, and quietly voiding the hybrid
objectives' max-cutoff guarantee.

The fixed behaviour:

- The equity merge is a LEFT join: demand locations missing from the
  equity data keep their travel/cost rows (with NaN equity values), so all
  primary metrics are computed over the full demand set.
- solve() checks equity coverage once up front: incomplete coverage is a
  hard error when "equity" is in the weights dict (the row weights would
  be NaN), and a warning otherwise (the locations are merely excluded from
  equity-band breakdowns).
- EvaluatedCombination refuses to build compound row weights from a column
  containing NaN, instead of letting NaN propagate silently into every
  score.
"""

import warnings

import pandas as pd
import pytest

import lokigi


def _problem_with_worst_point_missing_from_equity(include_equity=True):
    """
    Four demand points, two sites. LSOA_MISSING is absent from the equity
    data and is also Site_A's worst-case (max) travel point:

    - Site_A travel times: [38, 18, 10, 28] -> true max 38 (at LSOA_MISSING)
    - Site_B travel times: [25, 29, 11, 28] -> true max 29 (at LSOA_2)

    Correct behaviour: Site_B wins p_center (29 < 38).
    The buggy inner join dropped LSOA_MISSING, making Site_A's max appear
    to be 28 and flipping the winner to Site_A.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_MISSING", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_MISSING", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [38.0, 18.0, 10.0, 28.0],
            "Site_B": [25.0, 29.0, 11.0, 28.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    if include_equity:
        equity_df = pd.DataFrame(
            {
                "location_id": ["LSOA_2", "LSOA_3", "LSOA_4"],  # no LSOA_MISSING
                "imd_decile": [1, 5, 10],
            }
        )
        problem.add_equity_data(
            equity_df,
            equity_col="imd_decile",
            common_col="location_id",
            label="equity",
            disadvantaged_end="low",
            verbose=False,
        )
    return problem


# --- demand points must never be dropped from the metrics ---


def test_demand_points_missing_from_equity_still_count_in_metrics():
    """The worst-case travel point being absent from the equity data must
    not erase it: Site_A's max stays 38, and Site_B (true max 29) wins
    p_center. The buggy inner join reported Site_A's max as 28 and flipped
    the winner."""
    problem = _problem_with_worst_point_missing_from_equity()
    with pytest.warns(UserWarning, match="no matching row in the equity data"):
        result = problem.solve(
            p=1,
            objectives="p_center",
            search_strategy="brute-force",
            show_progress=False,
        )

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_B"]
    assert best["max"] == 29.0
    assert sorted(result.solution_df["max"]) == [29.0, 38.0]


def test_metrics_match_an_identical_problem_without_equity_data():
    """Merely having (incomplete) equity data attached must not change any
    primary metric relative to the same problem with no equity data."""
    with_equity = _problem_with_worst_point_missing_from_equity(include_equity=True)
    without_equity = _problem_with_worst_point_missing_from_equity(
        include_equity=False
    )

    with pytest.warns(UserWarning, match="no matching row in the equity data"):
        with_result = with_equity.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )
    without_result = without_equity.solve(
        p=1, objectives="p_median", search_strategy="brute-force", show_progress=False
    )

    for metric in ["weighted_average", "unweighted_average", "max"]:
        with_scores = dict(
            zip(
                [names[0] for names in with_result.solution_df["site_names"]],
                with_result.solution_df[metric],
            )
        )
        without_scores = dict(
            zip(
                [names[0] for names in without_result.solution_df["site_names"]],
                without_result.solution_df[metric],
            )
        )
        assert with_scores == pytest.approx(without_scores)


def test_hybrid_cutoff_guarantee_covers_all_demand_points():
    """hybrid objectives promise every demand point sits within
    max_value_cutoff. With a cutoff of 30, Site_A (true max 38, at the
    LSOA missing from the equity data) must be filtered out -- the buggy
    inner join hid that point and let Site_A through."""
    problem = _problem_with_worst_point_missing_from_equity()
    with pytest.warns(UserWarning, match="no matching row in the equity data"):
        result = problem.solve(
            p=1,
            objectives="hybrid_p_median",
            search_strategy="brute-force",
            show_progress=False,
            max_value_cutoff=30,
        )

    assert len(result.solution_df) == 1
    assert result.solution_df.iloc[0]["site_names"] == ["Site_B"]


# --- solve()'s upfront coverage check ---


def test_solve_warns_when_equity_incomplete_but_not_weighted():
    problem = _problem_with_worst_point_missing_from_equity()
    with pytest.warns(UserWarning, match="1 of 4 demand"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
        )


def test_solve_raises_when_weighting_by_incomplete_equity():
    """NaN equity values make equity row weights impossible, so asking to
    weight by equity with incomplete coverage must fail fast and clearly,
    before any solving starts."""
    problem = _problem_with_worst_point_missing_from_equity()
    with pytest.raises(ValueError, match="Cannot weight by equity"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"equity": 1.0},
        )


def test_no_warning_when_equity_data_is_complete():
    problem = _problem_with_worst_point_missing_from_equity(include_equity=False)
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_MISSING", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [3, 1, 5, 10],
        }
    )
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
        verbose=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"equity": 1.0},
        )
    equity_warnings = [w for w in caught if "equity data" in str(w.message)]
    assert equity_warnings == []


# --- the NaN guard on compound row weights ---


def test_direct_evaluation_with_equity_weights_raises_on_missing_rows():
    """evaluate_single_solution_single_objective doesn't pass through
    solve()'s coverage check, so the NaN guard inside EvaluatedCombination
    must catch incomplete equity coverage there instead of silently
    producing NaN scores."""
    problem = _problem_with_worst_point_missing_from_equity()
    with pytest.raises(ValueError, match="missing values"):
        problem.evaluate_single_solution_single_objective(
            objective="p_median",
            site_names=["Site_A"],
            weights={"equity": 1.0},
        )


def test_nan_equity_value_with_equity_weight_raises():
    """Complete ID coverage but a NaN in the equity column itself passes
    solve()'s row-coverage check, so the NaN guard must catch it during
    the first evaluation rather than letting NaN poison every score."""
    problem = _problem_with_worst_point_missing_from_equity(include_equity=False)
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_MISSING", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [None, 1, 5, 10],
        }
    )
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
        verbose=False,
    )

    with pytest.raises(ValueError, match="missing values"):
        problem.solve(
            p=1,
            objectives="p_median",
            search_strategy="brute-force",
            show_progress=False,
            weights={"equity": 1.0},
        )
