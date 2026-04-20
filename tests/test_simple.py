import lokigi


# Test that a simple problem runs successfully
def test_site_single_evaluation_runs(brighton_problem):
    result = brighton_problem.evaluate_single_solution_single_objective(
        site_indices=[1, 2], objective="p_median"
    )

    assert isinstance(result, lokigi.site.EvaluatedCombination)

    assert len(result.show_result_df()) > 0


# Test simple p-median brute force runs
def test_site_brute_force_runs(brighton_problem):
    result = brighton_problem.solve(p=3, objectives="p_median")

    assert isinstance(result, lokigi.site.SiteSolutionSet)
    # All possible combos of 3 sites for a total of 6 possible sites is 20 distinct combos
    assert len(result.solution_df) == 20
