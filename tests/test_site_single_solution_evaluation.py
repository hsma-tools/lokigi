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

    def test_uses_weights(self, loaded_problem):
        """weighted_average should be the demand-weighted mean of min_cost,
        not just the plain average -- with demand [100, 200, 150] and
        min_cost [10, 5, 15] for sites A/B, that's (1000+1000+2250)/450."""
        result = loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1], weights={"demand": 1.0}
        )

        assert result.weighted_average == pytest.approx(4250 / 450)
        assert result.weighted_average != pytest.approx(result.unweighted_average)

    def test_weights_dict_scale_is_ignored_for_demand_only_weighting(
        self, loaded_problem
    ):
        """A single-key {'demand': ...} dict falls back to weighting purely by
        the demand column -- the scalar itself shouldn't change the result,
        since it's just used to detect the legacy demand-only case."""
        default_result = loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1]
        )
        scaled_result = loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1], weights={"demand": 5.0}
        )

        assert scaled_result.weighted_average == pytest.approx(
            default_result.weighted_average
        )
