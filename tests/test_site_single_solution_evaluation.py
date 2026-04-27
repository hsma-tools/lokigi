import pytest


class TestEvaluateSingleSolution:
    def test_evaluate_by_site_indices(self, loaded_problem):
        result = loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1]
        )
        assert result is not None

    def test_evaluate_by_site_names(self, loaded_problem):
        result = loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_names=["Site_A", "Site_B"]
        )
        assert result is not None

    def test_raises_when_neither_names_nor_indices_given(self, loaded_problem):
        with pytest.raises(ValueError, match="Please provide either"):
            loaded_problem.evaluate_single_solution_single_objective(
                objective="p_median"
            )

    def test_raises_when_both_names_and_indices_given(self, loaded_problem):
        with pytest.raises(ValueError, match="Please provide either"):
            loaded_problem.evaluate_single_solution_single_objective(
                objective="p_median",
                site_names=["Site_A"],
                site_indices=[0],
            )

    def test_raises_on_unsupported_objective(self, loaded_problem):
        with pytest.raises(ValueError):
            loaded_problem.evaluate_single_solution_single_objective(
                objective="unsupported_objective", site_indices=[0]
            )

    def test_raises_on_missing_site_name(self, loaded_problem):
        with pytest.raises((KeyError, ValueError)):
            loaded_problem.evaluate_single_solution_single_objective(
                objective="p_median", site_names=["Nonexistent_Site"]
            )

    def test_raises_on_out_of_bounds_index(self, loaded_problem):
        with pytest.raises(IndexError):
            loaded_problem.evaluate_single_solution_single_objective(
                objective="p_median", site_indices=[999]
            )
