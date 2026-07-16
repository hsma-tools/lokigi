import pytest


def test_p_median_and_simple_p_median_diverge_when_demand_is_skewed(
    demand_sensitivity_skewed_problem,
):
    """
    On `demand_sensitivity_skewed_problem` (demand [10, 10, 40]), the
    demand-weighted and plain averages of the three p=2 combinations rank
    differently:

      {Site_A, Site_B}: mins [8, 4, 6]  -> weighted 6.0, unweighted 6.0
      {Site_A, Site_C}: mins [20, 4, 3] -> weighted 6.0, unweighted 9.0
      {Site_B, Site_C}: mins [8, 10, 3] -> weighted 5.0, unweighted 7.0

    p_median should pick {Site_B, Site_C} (lowest weighted_average, 5.0),
    while simple_p_median should pick {Site_A, Site_B} (lowest
    unweighted_average, 6.0) -- a genuinely different combination, proving
    that demand actually influences which sites p_median chooses.
    """
    p_median_result = demand_sensitivity_skewed_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )
    simple_result = demand_sensitivity_skewed_problem.solve(
        p=2,
        objectives="simple_p_median",
        search_strategy="brute-force",
        show_progress=False,
    )

    p_median_best = p_median_result.solution_df.iloc[0]
    simple_best = simple_result.solution_df.iloc[0]

    assert frozenset(p_median_best["site_names"]) == frozenset({"Site_B", "Site_C"})
    assert p_median_best["weighted_average"] == pytest.approx(5.0)

    assert frozenset(simple_best["site_names"]) == frozenset({"Site_A", "Site_B"})
    assert simple_best["unweighted_average"] == pytest.approx(6.0)

    assert frozenset(p_median_best["site_names"]) != frozenset(
        simple_best["site_names"]
    )


def test_p_median_and_simple_p_median_agree_when_demand_is_uniform(
    demand_sensitivity_uniform_problem,
):
    """Same site/travel data as the skewed-demand test above, but with equal
    demand at every location. Uniform weights make weighted_average and
    unweighted_average mathematically identical for every combination, so
    p_median and simple_p_median must select the same best combination:
    {Site_A, Site_B} at 6.0."""
    p_median_result = demand_sensitivity_uniform_problem.solve(
        p=2, objectives="p_median", search_strategy="brute-force", show_progress=False
    )
    simple_result = demand_sensitivity_uniform_problem.solve(
        p=2,
        objectives="simple_p_median",
        search_strategy="brute-force",
        show_progress=False,
    )

    p_median_best = p_median_result.solution_df.iloc[0]
    simple_best = simple_result.solution_df.iloc[0]

    assert frozenset(p_median_best["site_names"]) == frozenset(
        simple_best["site_names"]
    )
    assert frozenset(p_median_best["site_names"]) == frozenset({"Site_A", "Site_B"})

    assert p_median_best["weighted_average"] == pytest.approx(6.0)
    assert simple_best["unweighted_average"] == pytest.approx(6.0)

    for _, row in p_median_result.solution_df.iterrows():
        assert row["weighted_average"] == pytest.approx(row["unweighted_average"])
