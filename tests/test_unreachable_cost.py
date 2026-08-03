"""Tests for `solve(unreachable_cost=...)` -- the explicit modelling
policy for optimising over a primary travel matrix that contains a
genuinely missing (NaN) travel cost.

Chunk 1 (see test_missing_travel_values.py) made `weighted_average`/
`unweighted_average`/`90th_percentile`/`max` honest: computed over
reachable rows only, rather than one NaN poisoning the whole solution.
That's correct for *reporting*, but wrong for *ranking*: if solve() ranked
combinations on those honest, reachable-only figures, a combination that
strands more demand behind an unreachable pair would look BETTER, not
worse, since the stranded rows simply vanish from the average instead of
counting against it. `solve()` therefore refuses to optimise over a
matrix with missing values at all (`NotImplementedError`) unless the
caller supplies `unreachable_cost=<a number>` -- a finite cost substituted
for every unreachable pair, used only to rank/prune combinations, never in
the reported metrics.

`objective="mclp"` is exempt from the requirement entirely: its
coverage-proportion ranking already treats an unreachable pair as "not
covered" (correctly bad), so there's no equivalent silent-reward failure
mode to guard against.
"""

import numpy as np
import pandas as pd
import pytest

import lokigi
from lokigi.site_solutions import EvaluatedCombination
from lokigi.utils import _get_ranking_by_objective


# --- _get_ranking_by_objective -------------------------------------------


@pytest.mark.parametrize(
    "objective,base",
    [
        ("p_median", "weighted_average"),
        ("hybrid_p_median", "weighted_average"),
        ("simple_p_median", "unweighted_average"),
        ("hybrid_simple_p_median", "unweighted_average"),
        ("p_center", "max"),
    ],
)
def test_ranking_column_resolves_to_for_ranking_variant(objective, base):
    assert _get_ranking_by_objective(objective) == base
    assert (
        _get_ranking_by_objective(objective, unreachable_cost=1000)
        == f"{base}_for_ranking"
    )


def test_mclp_ranking_column_unaffected_by_unreachable_cost():
    assert _get_ranking_by_objective("mclp") == "proportion_within_coverage_threshold"
    assert (
        _get_ranking_by_objective("mclp", unreachable_cost=1000)
        == "proportion_within_coverage_threshold"
    )


# --- Direct EvaluatedCombination tests ------------------------------------


class _StubSiteProblem:
    def __init__(self, demand_col="demand"):
        self._demand_data_demand_col = demand_col


def _build(df, unreachable_cost=None):
    return EvaluatedCombination(
        solution_type="p_median",
        site_names=["A"],
        site_indices=[0],
        evaluated_combination_df=df,
        weights=None,
        site_problem=_StubSiteProblem(),
        coverage_threshold=None,
        unreachable_cost=unreachable_cost,
    )


@pytest.fixture
def partly_unreachable_df():
    return pd.DataFrame(
        {
            "min_cost": [10.0, np.nan, 30.0, 5.0],
            "demand": [100, 200, 150, 50],
            "within_threshold": [True, False, False, True],
        }
    )


def test_for_ranking_columns_equal_honest_ones_by_default(partly_unreachable_df):
    """unreachable_cost=None (never passed) is the overwhelming common
    case -- every existing caller must see zero behavioural change."""
    result = _build(partly_unreachable_df)
    assert result.weighted_average_for_ranking == result.weighted_average
    assert result.unweighted_average_for_ranking == result.unweighted_average
    assert result.max_for_ranking == result.max


def test_for_ranking_columns_substitute_unreachable_cost(partly_unreachable_df):
    result = _build(partly_unreachable_df, unreachable_cost=1000)
    # (10*100 + 1000*200 + 30*150 + 5*50) / 500
    expected_weighted = (10 * 100 + 1000 * 200 + 30 * 150 + 5 * 50) / 500
    assert result.weighted_average_for_ranking == pytest.approx(expected_weighted)
    assert result.unweighted_average_for_ranking == pytest.approx(
        (10 + 1000 + 30 + 5) / 4
    )
    assert result.max_for_ranking == 1000.0


def test_for_ranking_columns_do_not_affect_honest_reporting_metrics(
    partly_unreachable_df,
):
    """The whole point: unreachable_cost must never leak into the honest,
    reachable-only figures shown to users or plotted."""
    result = _build(partly_unreachable_df, unreachable_cost=1000)
    # Unchanged from the honest, reachable-only computation (see
    # test_missing_travel_values.py's equivalent assertions).
    assert result.weighted_average == pytest.approx((1000 + 4500 + 250) / 300)
    assert result.max == 30.0


def test_for_ranking_columns_when_fully_reachable_equal_unreachable_cost_is_a_noop():
    """No NaN present -- unreachable_cost has nothing to substitute, so
    the _for_ranking columns equal the honest ones regardless."""
    df = pd.DataFrame(
        {
            "min_cost": [10.0, 20.0],
            "demand": [100, 200],
            "within_threshold": [True, False],
        }
    )
    result = _build(df, unreachable_cost=1000)
    assert result.weighted_average_for_ranking == result.weighted_average
    assert result.max_for_ranking == result.max


def test_return_solution_metrics_always_includes_for_ranking_keys(
    partly_unreachable_df,
):
    metrics = _build(partly_unreachable_df).return_solution_metrics()
    assert "weighted_average_for_ranking" in metrics
    assert "unweighted_average_for_ranking" in metrics
    assert "max_for_ranking" in metrics


# --- solve()'s guard --------------------------------------------------


@pytest.fixture
def demand_df():
    return pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 100, 100]}
    )


@pytest.fixture
def candidate_df():
    return pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )


@pytest.fixture
def stranding_travel_df():
    """Site_A reaches everyone at a mediocre cost; Site_B is cheaper
    everywhere it reaches, but cannot reach LSOA_3 at all. A solve() that
    ranked on the honest (reachable-only) weighted_average would pick
    Site_B and silently strand LSOA_3's 100 people, since Site_B's
    honest weighted_average (5.0) beats Site_A's (10.0)."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 10.0, 10.0],
            "Site_B": [5.0, 5.0, np.nan],
        }
    )


@pytest.fixture
def stranding_problem(demand_df, candidate_df, stranding_travel_df):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(
        stranding_travel_df, source_col="source_id", allow_missing=True
    )
    return problem


@pytest.mark.parametrize(
    "objective", ["p_median", "simple_p_median", "p_center"]
)
def test_solve_requires_unreachable_cost_for_vulnerable_objectives(
    stranding_problem, objective
):
    with pytest.raises(NotImplementedError, match="unreachable_cost"):
        stranding_problem.solve(p=1, objectives=objective, show_progress=False)


def test_solve_mclp_does_not_require_unreachable_cost(stranding_problem):
    """mclp's coverage ranking is already safe with a missing value --
    solving must succeed with no unreachable_cost at all."""
    result = stranding_problem.solve(
        p=1, objectives="mclp", threshold_for_coverage=8, show_progress=False
    )
    assert "proportion_within_coverage_threshold" in result.solution_df.columns


def test_solve_hybrid_p_median_requires_unreachable_cost(stranding_problem):
    with pytest.raises(NotImplementedError, match="unreachable_cost"):
        stranding_problem.solve(
            p=1,
            objectives="hybrid_p_median",
            max_value_cutoff=100,
            show_progress=False,
        )


def test_solve_without_any_missing_values_ignores_unreachable_cost(
    demand_df, candidate_df
):
    """unreachable_cost is a no-op when there's nothing to substitute --
    passing it on a complete matrix must not raise or change the winner."""
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 10.0, 10.0],
            "Site_B": [5.0, 5.0, 5.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)

    result = problem.solve(p=1, unreachable_cost=1000, show_progress=False)
    assert result.solution_df.iloc[0]["site_names"] == ["Site_B"]


# --- The actual fix: correct ranking under unreachable_cost, per strategy -


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_solve_does_not_reward_stranding_demand(stranding_problem, search_strategy):
    """The core regression this feature exists to prevent: without
    unreachable_cost, Site_B's honest weighted_average (5.0) beats
    Site_A's (10.0) -- see stranding_travel_df's docstring. With
    unreachable_cost large enough to penalise the stranded 100 people,
    Site_A must win instead, for every search strategy."""
    result = stranding_problem.solve(
        p=1,
        unreachable_cost=1000,
        search_strategy=search_strategy,
        show_progress=False,
        grasp_num_solutions=1,
    )
    assert result.solution_df.iloc[0]["site_names"] == ["Site_A"]


def test_solve_reports_honest_metrics_not_unreachable_cost(stranding_problem):
    """solution_df's reported weighted_average must stay the honest,
    reachable-only figure -- unreachable_cost must never leak into it,
    even though it drove the ranking that picked this winner."""
    result = stranding_problem.solve(p=1, unreachable_cost=1000, show_progress=False)
    row = result.solution_df.iloc[0]
    assert row["site_names"] == ["Site_A"]
    # Site_A has no missing values at all, so honest == for_ranking here;
    # the meaningful check is that a *smaller* solve() (see below) shows
    # the two diverging for the loser.
    assert row["weighted_average"] == 10.0
    assert row["weighted_average_for_ranking"] == 10.0


def test_evaluate_single_solution_shows_honest_vs_ranking_divergence(
    stranding_problem,
):
    """Direct evaluation of the LOSING combination (Site_B) makes the
    divergence concrete: its honest weighted_average looks better than
    the winner's, but its ranking-only figure (what solve() actually used)
    correctly reflects the stranded demand."""
    combo = stranding_problem.evaluate_single_solution_single_objective(
        site_names=["Site_B"], unreachable_cost=1000
    )
    assert combo.weighted_average == pytest.approx(5.0)
    assert combo.weighted_average_for_ranking == pytest.approx(
        (5 * 100 + 5 * 100 + 1000 * 100) / 300
    )
    assert combo.weighted_average_for_ranking > combo.weighted_average


# --- max_value_cutoff uses the ranking-safe max ---------------------------


def test_max_value_cutoff_rejects_solution_via_ranking_max(
    demand_df, candidate_df
):
    """A combination whose only unreachable-substituted cost exceeds
    max_value_cutoff must be treated as infeasible -- using max_for_ranking,
    not the honest (reachable-only) max, which would misleadingly show the
    stranding solution as always satisfying the cutoff."""
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 10.0, 10.0],
            "Site_B": [5.0, 5.0, np.nan],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)

    # cutoff of 20 would accept Site_B on its honest max (5.0) but must
    # reject it once unreachable_cost=1000 makes its true worst case 1000.
    result = problem.solve(
        p=1,
        objectives="hybrid_p_median",
        max_value_cutoff=20,
        unreachable_cost=1000,
        show_progress=False,
    )
    assert result.solution_df.iloc[0]["site_names"] == ["Site_A"]
