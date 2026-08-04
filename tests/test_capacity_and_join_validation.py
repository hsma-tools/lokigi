"""
Tests for two supporting-validation code paths that sit outside the main
solve() flow:

- `_validate_capacity_constraint` (lokigi/utils.py) -- not currently wired
  into solve() since `capacitated=True` always raises NotImplementedError
  first, but capacity is a stated future feature, so it's worth locking down
  in isolation.
- `_create_joined_demand_travel_df`'s handling of a partial ID mismatch
  between demand data and the travel matrix.
- `_create_joined_demand_travel_df`'s guards for a missing travel matrix or
  missing demand data, which previously surfaced as a cryptic pandas merge
  TypeError ("a <class 'NoneType'> was passed").
"""

import pandas as pd
import pytest

import lokigi
from lokigi.utils import _validate_capacity_constraint


class _StubSiteProblem:
    def __init__(self, demand_df, demand_col, candidate_df, capacity_col):
        self.demand_data = demand_df
        self._demand_data_demand_col = demand_col
        self.candidate_sites = candidate_df
        self._candidate_sites_capacity_col = capacity_col


def test_validate_capacity_constraint_passes_when_capacity_is_sufficient():
    stub = _StubSiteProblem(
        demand_df=pd.DataFrame({"demand": [100, 200]}),
        demand_col="demand",
        candidate_df=pd.DataFrame({"capacity": [200, 150]}),
        capacity_col="capacity",
    )
    _validate_capacity_constraint(stub)  # should not raise


def test_validate_capacity_constraint_raises_when_capacity_is_insufficient():
    stub = _StubSiteProblem(
        demand_df=pd.DataFrame({"demand": [100, 200]}),
        demand_col="demand",
        candidate_df=pd.DataFrame({"capacity": [50, 100]}),
        capacity_col="capacity",
    )
    with pytest.raises(ValueError, match="Insufficient Capacity"):
        _validate_capacity_constraint(stub)


def test_capacitated_solving_raises_not_implemented(loaded_problem):
    with pytest.raises(NotImplementedError):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1], capacitated=True
        )


def test_partial_demand_travel_matrix_mismatch_silently_drops_rows():
    """
    NOTE: `_create_joined_demand_travel_df` only raises if the inner join
    produces zero rows entirely. A demand location with an ID absent from
    the travel matrix is silently dropped rather than raising or warning.
    This test documents that current behaviour explicitly, so a future
    change to make it warn/raise is a deliberate decision, not a surprise.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_MISSING"],
            "demand": [100, 200, 150, 999],
        }
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B", "Site_C"], "lat": [51.5, 51.6, 51.7], "long": [-0.1, -0.2, -0.3]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )
    joined_df = result.show_result_df()

    assert len(joined_df) == 3
    assert "LSOA_MISSING" not in joined_df["location_id"].tolist()


def _demand_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 200, 150],
        }
    )


def _candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
            "capacity": [500, 500, 500],
        }
    )


def _travel_df():
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )


def test_two_step_floating_catchment_without_travel_matrix_raises_helpful_error():
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(_demand_df(), demand_col="demand", location_id_col="location_id")
    problem.add_sites(_candidate_df(), candidate_id_col="site_id")

    with pytest.raises(ValueError, match=r"add_travel_matrix"):
        problem.two_step_floating_catchment(supply_col="capacity", catchment_size=30)


def test_evaluate_without_travel_matrix_raises_helpful_error():
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(_demand_df(), demand_col="demand", location_id_col="location_id")
    problem.add_sites(_candidate_df(), candidate_id_col="site_id")

    with pytest.raises(ValueError, match=r"add_travel_matrix"):
        problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0]
        )


def test_evaluate_without_demand_raises_helpful_error():
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_sites(_candidate_df(), candidate_id_col="site_id")
    problem.add_travel_matrix(_travel_df(), source_col="source_id")

    with pytest.raises(ValueError, match=r"add_demand"):
        problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0]
        )
