"""
Tests for the direction of equity weighting in solve()/EvaluatedCombination.

These pin the fix for an inversion bug in EvaluatedCombination's
compound-weight blending (site_solutions.py): the mapping from
`_equity_data_direction` to the row-weight direction was backwards, so
weights={"equity": ...} gave the MOST weight to the LEAST deprived regions
-- the exact opposite of the feature's purpose ("put more importance on
improving access in the most deprived areas").

Equity weighting must up-weight worse-off regions under BOTH encodings:

- direction="higher_is_better" (e.g. DLUHC IMD deciles, 10 = least
  deprived): LOW values are the deprived regions and must get the weight.
- direction="higher_is_worse" (e.g. raw IMD scores, higher = more
  deprived): HIGH values are the deprived regions and must get the weight.

The same fix also aligned add_equity_data()'s signature default with its
docstring ("higher_is_better") -- previously the signature said
"higher_is_worse" while the docstring said "higher_is_better".
"""

import pandas as pd
import pytest

import lokigi


def _two_site_problem():
    """
    A minimal, hand-checkable problem where the equity-optimal and
    equity-pessimal sites are cleanly separated:

    - LSOA_DEPRIVED / LSOA_MIDDLE / LSOA_AFFLUENT demand points.
    - Site_NearDeprived is close to the deprived LSOA and far from the
      affluent one; Site_NearAffluent is the mirror image.

    With deciles [1, 5, 10], min-max normalisation gives [0, 4/9, 1], so a
    correctly-inverted equity weighting yields row weights [1, 5/9, 0] and:

    - Site_NearDeprived weighted_average = (5*1 + 20*5/9) / (14/9) = 145/14
    - Site_NearAffluent weighted_average = (50*1 + 20*5/9) / (14/9) = 550/14

    The inverted (buggy) weighting instead picks Site_NearAffluent.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_DEPRIVED", "LSOA_MIDDLE", "LSOA_AFFLUENT"],
            "demand": [100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_NearDeprived", "Site_NearAffluent"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_DEPRIVED", "LSOA_MIDDLE", "LSOA_AFFLUENT"],
            "Site_NearDeprived": [5.0, 20.0, 50.0],
            "Site_NearAffluent": [50.0, 20.0, 5.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def _add_equity(problem, values, **kwargs):
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_DEPRIVED", "LSOA_MIDDLE", "LSOA_AFFLUENT"],
            "equity_value": values,
        }
    )
    problem.add_equity_data(
        equity_df,
        equity_col="equity_value",
        common_col="location_id",
        label="equity",
        verbose=False,
        **kwargs,
    )
    return problem


def _solve_equity_weighted(problem):
    return problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"equity": 1.0},
    )


# --- core direction behaviour, both encodings ---


def test_decile_data_prioritises_most_deprived_region():
    """IMD deciles (1 = most deprived) with the documented
    direction="higher_is_better": the deprived LSOA must dominate the
    weighting, so the best site is the one nearest to it."""
    problem = _add_equity(_two_site_problem(), [1, 5, 10], direction="higher_is_better")
    result = _solve_equity_weighted(problem)

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_NearDeprived"]
    assert best["weighted_average"] == pytest.approx(145 / 14)


def test_raw_score_data_prioritises_most_deprived_region():
    """Raw IMD-style scores (higher = more deprived) with
    direction="higher_is_worse": high-score (deprived) regions must get
    the weight, so the same site wins."""
    problem = _add_equity(_two_site_problem(), [90, 50, 0], direction="higher_is_worse")
    result = _solve_equity_weighted(problem)

    best = result.solution_df.iloc[0]
    assert best["site_names"] == ["Site_NearDeprived"]
    # scores [90, 50, 0] normalise to [1, 5/9, 0] directly (no inversion),
    # giving exactly the same row weights as the decile test above.
    assert best["weighted_average"] == pytest.approx(145 / 14)


def test_equivalent_decile_and_score_encodings_agree():
    """The two encodings of the same underlying deprivation ordering must
    produce identical weighted averages for every combination."""
    decile_problem = _add_equity(
        _two_site_problem(), [1, 5, 10], direction="higher_is_better"
    )
    score_problem = _add_equity(
        _two_site_problem(), [10, 6, 1], direction="higher_is_worse"
    )

    decile_result = _solve_equity_weighted(decile_problem)
    score_result = _solve_equity_weighted(score_problem)

    decile_scores = dict(
        zip(
            [names[0] for names in decile_result.solution_df["site_names"]],
            decile_result.solution_df["weighted_average"],
        )
    )
    score_scores = dict(
        zip(
            [names[0] for names in score_result.solution_df["site_names"]],
            score_result.solution_df["weighted_average"],
        )
    )
    assert decile_scores.keys() == score_scores.keys()
    for site, value in decile_scores.items():
        assert score_scores[site] == pytest.approx(value)


def test_equity_weighted_ranking_orders_all_combinations_correctly():
    """Not just the winner: the full solution_df must rank the
    deprived-serving site above the affluent-serving one."""
    problem = _add_equity(_two_site_problem(), [1, 5, 10], direction="higher_is_better")
    result = _solve_equity_weighted(problem)

    ranked_sites = [names[0] for names in result.solution_df["site_names"]]
    assert ranked_sites == ["Site_NearDeprived", "Site_NearAffluent"]
    assert result.solution_df.iloc[1]["weighted_average"] == pytest.approx(550 / 14)


# --- default direction ---


def test_default_direction_is_the_documented_decile_convention():
    """add_equity_data()'s docstring documents the default as
    "higher_is_better" (the DLUHC IMD decile convention); the signature
    must match, so decile data with no explicit direction still
    prioritises the most deprived region."""
    problem = _add_equity(_two_site_problem(), [1, 5, 10])  # direction omitted
    result = _solve_equity_weighted(problem)

    assert result.solution_df.iloc[0]["site_names"] == ["Site_NearDeprived"]
    assert problem._equity_data_direction == "higher_is_better"


# --- blending and degenerate cases ---


def test_blended_demand_and_equity_weights_use_corrected_direction():
    """A 50/50 demand+equity blend must apply the corrected equity
    direction inside the compound row weights. With demand
    [100, 400, 200] -> norm [0, 1, 1/3] and equity deciles [1, 5, 10] ->
    weights [1, 5/9, 0], the blend is 0.5*demand + 0.5*equity =
    [1/2, 14/18, 1/6], giving:

    - Site_NearDeprived = (5*1/2 + 20*14/18 + 50*1/6) / (26/18) = 475/26
    - Site_NearAffluent = (50*1/2 + 20*14/18 + 5*1/6) / (26/18) = 745/26

    (The inverted, buggy direction gives equity weights [0, 4/9, 1] and
    would rank Site_NearAffluent first instead.)"""
    problem = _two_site_problem()
    problem.demand_data.loc[
        problem.demand_data["location_id"] == "LSOA_MIDDLE", "demand"
    ] = 400
    problem.demand_data.loc[
        problem.demand_data["location_id"] == "LSOA_AFFLUENT", "demand"
    ] = 200
    _add_equity(problem, [1, 5, 10], direction="higher_is_better")

    result = problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 0.5, "equity": 0.5},
    )

    scores = dict(
        zip(
            [names[0] for names in result.solution_df["site_names"]],
            result.solution_df["weighted_average"],
        )
    )
    assert scores["Site_NearDeprived"] == pytest.approx(475 / 26)
    assert scores["Site_NearAffluent"] == pytest.approx(745 / 26)
    assert result.solution_df.iloc[0]["site_names"] == ["Site_NearDeprived"]


def test_uniform_equity_values_give_equal_weights():
    """When every region has the same equity value there is nothing to
    normalise against; each region gets an equal (full) weight and the
    equity-weighted average collapses to the plain mean."""
    problem = _add_equity(_two_site_problem(), [5, 5, 5], direction="higher_is_better")
    result = _solve_equity_weighted(problem)

    for _, row in result.solution_df.iterrows():
        assert row["weighted_average"] == pytest.approx(row["unweighted_average"])


def test_equity_weighting_beats_demand_weighting_at_serving_deprived_areas():
    """End-to-end sanity check of the feature's whole point: when demand
    pulls toward the affluent LSOA, switching from demand weights to
    equity weights must flip the chosen site toward the deprived LSOA."""
    problem = _two_site_problem()
    problem.demand_data.loc[
        problem.demand_data["location_id"] == "LSOA_AFFLUENT", "demand"
    ] = 1000
    _add_equity(problem, [1, 5, 10], direction="higher_is_better")

    demand_choice = problem.solve(
        p=1,
        objectives="p_median",
        search_strategy="brute-force",
        show_progress=False,
        weights={"demand": 1.0},
    ).solution_df.iloc[0]["site_names"]
    equity_choice = _solve_equity_weighted(problem).solution_df.iloc[0]["site_names"]

    assert demand_choice == ["Site_NearAffluent"]
    assert equity_choice == ["Site_NearDeprived"]
