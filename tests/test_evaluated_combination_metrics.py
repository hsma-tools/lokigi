"""
Direct unit tests for EvaluatedCombination's per-solution score formulas
(site_solutions.py), bypassing solve()/evaluate_single_solution_single_objective
so each metric can be checked against a small, hand-built dataframe with a
known answer.
"""

import pandas as pd
import pytest

import lokigi
from lokigi.site_solutions import EvaluatedCombination


class _StubSiteProblem:
    """Minimal duck-typed stand-in -- EvaluatedCombination only reads
    _demand_data_demand_col (always) and _equity_data_equity_col/
    _equity_data_disadvantaged_end (only if equity_data was actually
    configured)."""

    def __init__(
        self, demand_col="demand", equity_col=None, equity_disadvantaged_end=None
    ):
        self._demand_data_demand_col = demand_col
        if equity_col is not None:
            self._equity_data_equity_col = equity_col
            self._equity_data_disadvantaged_end = equity_disadvantaged_end


@pytest.fixture
def combination_df():
    return pd.DataFrame(
        {
            "min_cost": [10.0, 20.0, 30.0, 5.0],
            "demand": [100, 200, 150, 50],
            "within_threshold": [True, False, False, True],
        }
    )


def _build(df, coverage_threshold=15.0, site_problem=None):
    return EvaluatedCombination(
        solution_type="p_median",
        site_names=["A"],
        site_indices=[0],
        evaluated_combination_df=df,
        weights=None,
        site_problem=site_problem or _StubSiteProblem(),
        coverage_threshold=coverage_threshold,
    )


def test_weighted_average_is_demand_weighted_mean(combination_df):
    result = _build(combination_df)
    # (10*100 + 20*200 + 30*150 + 5*50) / (100+200+150+50)
    assert result.weighted_average == pytest.approx(9750 / 500)


def test_unweighted_average_is_plain_mean(combination_df):
    result = _build(combination_df)
    assert result.unweighted_average == pytest.approx((10 + 20 + 30 + 5) / 4)


def test_max_is_the_worst_min_cost(combination_df):
    result = _build(combination_df)
    assert result.max == 30.0


def test_percentile_90th_matches_numpy(combination_df):
    import numpy as np

    result = _build(combination_df)
    assert result.percentile_90th == pytest.approx(
        np.percentile(combination_df["min_cost"], q=90)
    )


def test_proportion_within_coverage_threshold_is_mean_of_flags(combination_df):
    result = _build(combination_df)
    assert result.proportion_within_coverage_threshold == pytest.approx(0.5)


def test_coverage_by_equity_group_is_a_dict_not_a_tuple():
    """
    BUG: site_solutions.py:313-315 has a trailing comma --
    `self.coverage_by_equity_group = (grouped_df[...].to_dict(),)` --
    which wraps the intended dict in a 1-tuple.
    """
    df = pd.DataFrame(
        {
            "min_cost": [10.0, 20.0, 30.0, 5.0],
            "demand": [100, 200, 150, 50],
            "within_threshold": [True, False, False, True],
            "equity_band": [1, 2, 3, 1],
        }
    )
    result = _build(
        df,
        site_problem=_StubSiteProblem(
            equity_col="equity_band", equity_disadvantaged_end="low"
        ),
    )

    assert isinstance(result.coverage_by_equity_group, dict)


def test_within_threshold_boundary_tie_is_not_covered():
    """A demand point whose min_cost exactly equals threshold_for_coverage
    should NOT count as covered -- site.py uses a strict `<` comparison
    (site.py:329), so exact ties fall on the "not covered" side."""
    demand_df = pd.DataFrame({"location_id": ["LSOA_1", "LSOA_2"], "demand": [100, 100]})
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {"source_id": ["LSOA_1", "LSOA_2"], "Site_A": [10.0, 15.0]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.evaluate_single_solution_single_objective(
        objective="mclp", site_indices=[0], threshold_for_coverage=15.0
    )
    df = result.show_result_df().set_index("location_id")

    assert df.loc["LSOA_1", "within_threshold"] == True
    assert df.loc["LSOA_2", "within_threshold"] == False
    assert result.proportion_within_coverage_threshold == pytest.approx(0.5)
