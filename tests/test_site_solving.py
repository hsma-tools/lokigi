import pytest


def test_solve_returns_solution_set(loaded_problem):
    result = loaded_problem.solve(p=2, objectives="p_median")
    assert result is not None


def test_solve_p1_returns_single_best_site(loaded_problem):
    result = loaded_problem.solve(p=1, objectives="p_median")
    assert result is not None


def test_solve_raises_on_unsupported_objective(loaded_problem):
    with pytest.raises(ValueError):
        loaded_problem.solve(p=2, objectives="unsupported")


def test_solve_raises_on_invalid_search_strategy(loaded_problem):
    with pytest.raises(ValueError, match="Unsupported search strategy"):
        loaded_problem.solve(p=2, search_strategy="invalid_strategy")


def test_solve_warns_on_multiple_objectives(loaded_problem):
    with pytest.warns(UserWarning):
        loaded_problem.solve(p=2, objectives=["p_median", "p_centre"])


def test_low_demand_not_registered_as_cost(loaded_problem_low_demand):
    result = loaded_problem_low_demand.solve(p=2, objectives="p_median")
    # print(result.show_solutions())
    # print(
    #     result.show_solutions()["problem_df"].iloc[0][
    #         ["min_cost", result.site_problem._demand_data_demand_col]
    #     ]
    # )

    for idx, row in result.show_solutions()["problem_df"].iloc[0].iterrows():
        assert row["min_cost"] != row["demand"], (
            "Minimum cost is equal to demand - check min cost identification logic"
        )
        site_values = row[[c for c in row.index if "Site_" in c]]
        assert row["min_cost"] == site_values.min(), (
            "Minimum cost is not equal to the travel time for any of the sites being evaluated"
        )
