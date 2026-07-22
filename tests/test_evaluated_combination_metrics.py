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


def test_proportion_regions_within_coverage_threshold_is_mean_of_flags(combination_df):
    # 2 of the 4 regions are flagged covered, regardless of their demand.
    result = _build(combination_df)
    assert result.proportion_regions_within_coverage_threshold == pytest.approx(0.5)


def test_proportion_within_coverage_threshold_is_demand_weighted(combination_df):
    """The headline coverage metric weighs each region by its demand, so it
    diverges from the region count whenever demand is uneven.

    Here the two covered regions are the two *smallest* (100 and 50 of 500
    total demand), so demand-weighted coverage is well below the 0.5 share of
    regions -- exactly the case the region count was hiding.
    """
    result = _build(combination_df)
    assert result.proportion_within_coverage_threshold == pytest.approx(
        (100 + 50) / 500
    )
    assert result.proportion_regions_within_coverage_threshold == pytest.approx(0.5)


def test_coverage_metrics_are_nan_when_no_threshold_was_given(combination_df):
    """With no threshold_for_coverage, `within_threshold` is all-NaN and both
    metrics must stay NaN ("not measured").

    Guarding this explicitly matters because pandas' .sum() skips NaN: without
    the all-NaN check in _coverage_proportion, the demand-weighted branch
    returns a confident 0.0 and reports that none of the demand is covered for
    a problem where coverage was never assessed at all.
    """
    import numpy as np

    df = combination_df.assign(within_threshold=np.nan)
    result = _build(df, coverage_threshold=None)

    assert np.isnan(result.proportion_within_coverage_threshold)
    assert np.isnan(result.proportion_regions_within_coverage_threshold)


def test_coverage_by_equity_group_is_demand_weighted_and_regions_variant_is_not():
    """The per-band dicts follow the same naming rule as the headline pair:
    unqualified is demand-weighted, `regions` counts regions equally.

    Band 1 holds two regions (demand 100 covered, 50 covered) and band 2 holds
    two regions (demand 200 uncovered, 400 covered), so the two dicts disagree
    on band 2 -- a fixture with equal-sized, equal-demand bands would pass
    under either implementation.
    """
    df = pd.DataFrame(
        {
            "min_cost": [10.0, 5.0, 20.0, 8.0],
            "demand": [100, 50, 200, 400],
            "within_threshold": [True, True, False, True],
            "equity_band": [1, 1, 2, 2],
        }
    )
    result = _build(
        df,
        site_problem=_StubSiteProblem(
            equity_col="equity_band", equity_disadvantaged_end="low"
        ),
    )

    assert result.coverage_regions_by_equity_group == {1: 1.0, 2: 0.5}
    # Band 2: 400 of 600 demand covered.
    assert result.coverage_by_equity_group == {1: 1.0, 2: pytest.approx(0.67, abs=0.01)}


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
    # Demand is equal across the two regions here, so both metrics agree.
    assert result.proportion_within_coverage_threshold == pytest.approx(0.5)
    assert result.proportion_regions_within_coverage_threshold == pytest.approx(0.5)


def test_coverage_metrics_agree_when_add_demand_was_never_called():
    """solve() assumes equal demand when add_demand() was never called, which
    makes the demand-weighted and region-based metrics the same quantity -- so
    users who never supplied demand see no change from the v0.7.0 reweighting.
    """
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {"source_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "Site_A": [5.0, 10.0, 30.0]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.solve(
        p=1,
        objectives="mclp",
        show_progress=False,
        threshold_for_coverage=15.0,
    )
    row = result.solution_df.iloc[0]

    assert row["proportion_within_coverage_threshold"] == pytest.approx(2 / 3)
    assert row["proportion_regions_within_coverage_threshold"] == pytest.approx(2 / 3)
