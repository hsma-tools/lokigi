"""
End-to-end tests for the equity-band metrics computed inside
EvaluatedCombination (site_solutions.py:253-362) once equity data has been
registered via add_equity_data(). These were previously entirely untested --
no fixture exercised the equity code path at all.
"""

import warnings

import pandas as pd
import pytest

import lokigi
from lokigi.site_solutions import StrandedEquityBandWarning


def test_equity_band_gap_and_tertile_metrics(loaded_problem_with_equity):
    """With one demand location per equity band (deciles 1, 5, 10), each
    band's weighted average is just that location's own min_cost, making the
    gap/tertile arithmetic straightforward to hand-verify."""
    result = loaded_problem_with_equity.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group == {1: 10.0, 5: 5.0, 10: 15.0}

    assert result.gap_absolute_weighted == pytest.approx(10.0)
    assert "Spread of 10.0 units" in result.gap_absolute_desc

    assert result.gap_relative_weighted == pytest.approx(3.0)
    assert "Significant Disparity" in result.gap_relative_desc

    assert result.avg_lower_third_bins == pytest.approx(10.0)
    assert result.avg_middle_third_bins == pytest.approx(5.0)
    assert result.avg_upper_third_bins == pytest.approx(15.0)
    assert result.inter_tertile_ratio == pytest.approx(10.0 / 15.0)
    assert "Highly Progressive" in result.inter_tertile_desc


def test_gap_absolute_description_names_registered_unit():
    """gap_absolute_desc's "Spread of X units" previously always used the
    literal word "units", regardless of what unit the travel matrix was
    actually registered with -- meaningless when e.g. minutes or miles was
    the real unit. Falls back to the literal word "units" only when no
    unit was ever registered (see test_equity_band_gap_and_tertile_metrics,
    whose loaded_problem_with_equity fixture registers no unit)."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 100, 100]}
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 5.0, 15.0],
            "Site_B": [20.0, 15.0, 25.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="minutes")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD decile",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )
    assert "Spread of 10.0 minutes" in result.gap_absolute_desc


def test_gap_relative_weighted_zero_baseline_guard():
    """When the best-performing equity band has a weighted average of
    exactly 0, gap_relative_weighted must not raise ZeroDivisionError -- it
    should fall back to NaN with a descriptive message."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [0.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group[1] == pytest.approx(0.0)
    assert pd.isna(result.gap_relative_weighted)
    assert result.gap_relative_desc == "N/A (Zero baseline cost)"


def test_zero_weight_band_falls_back_to_unweighted_mean():
    """
    NOTE: when every demand point in an equity band has weight 0 (here,
    demand=0), `weighted_by_equity_group` for that band silently falls back
    to the plain (unweighted) mean rather than raising a ZeroDivisionError
    (site_solutions.py:273-279). This test pins that current fallback
    behaviour -- worth flagging as a known weighting-scheme inconsistency
    (a band computed this way isn't really "weighted" like its neighbours),
    even though it's the sensible choice over crashing.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [0, 0, 200, 150],
        }
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_A": [10.0, 20.0, 30.0, 12.0],
            "Site_B": [25.0, 5.0, 15.0, 9.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_decile": [1, 1, 5, 10],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    # band 1 = LSOA_1 (min_cost 10.0) + LSOA_2 (min_cost 5.0), both demand=0
    assert result.weighted_by_equity_group[1] == pytest.approx((10.0 + 5.0) / 2)
    assert result.weighted_by_equity_group[1] == pytest.approx(
        result.unweighted_by_equity_group[1]
    )


def test_tertile_ordering_respects_disadvantaged_end_high(loaded_problem):
    """Same three-band setup as test_equity_band_gap_and_tertile_metrics
    (deciles 1, 5, 10), but with disadvantaged_end="high" instead of "low"
    -- i.e. band 10 is now the MOST disadvantaged and band 1 the LEAST.
    The tertile ordering must flip accordingly: avg_lower_third_bins (most
    disadvantaged) becomes band 10's value (15.0, not band 1's 10.0), and
    avg_upper_third_bins (least disadvantaged) becomes band 1's value
    (10.0). Before the disadvantaged_end fix, this ordering was hardcoded
    to "lower raw bin value = more disadvantaged" regardless of this
    parameter, which is wrong whenever disadvantaged_end="high"."""
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 5, 10]}
    )
    loaded_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="high",
    )

    result = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.weighted_by_equity_group == {1: 10.0, 5: 5.0, 10: 15.0}
    assert result.avg_lower_third_bins == pytest.approx(15.0)
    assert result.avg_middle_third_bins == pytest.approx(5.0)
    assert result.avg_upper_third_bins == pytest.approx(10.0)
    assert result.inter_tertile_ratio == pytest.approx(15.0 / 10.0)
    assert "Highly Inequitable" in result.inter_tertile_desc


def test_tertile_split_uses_equal_sized_ends_not_array_split_remainder():
    """Regression test for the `_split_bins_into_tertiles()` fix: with 10
    IMD deciles (not divisible by 3), the old `np.array_split`-based split
    put the leftover band in the first chunk (1-4 / 5-7 / 8-10), comparing
    a 4-band "most deprived" group against a 3-band "least deprived" one.
    The fix keeps both ends at 3 bands each and puts the leftover decile
    (4) in the ignored middle instead (1-3 / 4-7 / 8-10).

    Decile 4's travel time (30.0) is deliberately far out of line with the
    rest, so which chunk it lands in isn't cosmetic: including it in the
    "most deprived" average (old behaviour) pushes inter_tertile_ratio from
    1.0 ("Balanced") to 1.5 ("Highly Inequitable") -- the fix changes not
    just a number but which `inter_tertile_desc` band the result falls
    into."""
    demand_df = pd.DataFrame(
        {
            "location_id": [f"LSOA_{i}" for i in range(1, 11)],
            "demand": [100] * 10,
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": [f"LSOA_{i}" for i in range(1, 11)],
            "Site_A": [10.0, 10.0, 10.0, 30.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": [f"LSOA_{i}" for i in range(1, 11)], "imd_decile": range(1, 11)}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD decile",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0]
    )

    assert result.avg_lower_third_bins == pytest.approx(10.0)
    assert result.avg_middle_third_bins == pytest.approx(7.5)
    assert result.avg_upper_third_bins == pytest.approx(10.0)

    # The invariant the discrepancy report caught: the ratio is exactly the
    # mean of the two chunks _split_bins_into_tertiles() actually returned,
    # not some other grouping.
    assert result.inter_tertile_ratio == pytest.approx(
        result.avg_lower_third_bins / result.avg_upper_third_bins
    )
    assert result.inter_tertile_ratio == pytest.approx(1.0)
    assert "Balanced" in result.inter_tertile_desc


def test_check_solution_equity_weighted_matches_weighted_by_equity_group():
    """check_solution_equity(weighted=True) must return the same figures as
    weighted_by_equity_group (and therefore inter_tertile_ratio) -- the
    whole point of adding weighted= was to remove the silent divergence
    between this method's default (plain per-region mean) and the
    demand-weighted metric plot_equity_tertiles() decomposes. Uses a band
    with two locations at very different demand so the weighted and
    unweighted means are provably different, not just non-identical by
    floating-point noise."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "demand": [990, 10, 100, 100],
        }
    )
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "Site_A": [0.0, 100.0, 20.0, 30.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1a", "LSOA_1b", "LSOA_2", "LSOA_3"],
            "band": [1, 1, 2, 3],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="band",
        common_col="location_id",
        label="Band",
        disadvantaged_end="low",
    )
    solution = problem.solve(p=1, search_strategy="brute-force", show_progress=False)
    row = solution.solution_df.iloc[0]

    unweighted_summary = solution.check_solution_equity(weighted=False).set_index("band")[
        "min_cost"
    ]
    weighted_summary = solution.check_solution_equity(weighted=True).set_index("band")[
        "min_cost"
    ]

    assert unweighted_summary.loc[1] == pytest.approx(50.0)
    assert weighted_summary.loc[1] == pytest.approx(row["weighted_by_equity_group"][1])
    assert weighted_summary.loc[1] != pytest.approx(unweighted_summary.loc[1])
    assert weighted_summary.loc[1] == pytest.approx(0.0, abs=2.0)


def test_fewer_than_three_equity_bins_skips_tertile_calculation():
    """Tertile metrics require >= 3 distinct equity bands (site_solutions.py:325);
    with only 2, avg_*_third_bins and inter_tertile_ratio should stay unset."""
    demand_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )
    candidate_df = pd.DataFrame(
        {"site_id": ["Site_A", "Site_B"], "lat": [51.5, 51.6], "long": [-0.1, -0.2]}
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
        }
    )
    equity_df = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [1, 1, 10]}
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )

    result = problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )

    assert result.avg_lower_third_bins is None
    assert result.inter_tertile_ratio is None
    # Names the real reason. Equity data IS registered here, so the old
    # "N/A (No equity data)" sent the reader looking for a missing dataset
    # instead of for a third band.
    assert result.inter_tertile_desc == ("N/A (needs at least 3 equity bands, found 2)")


def _stranded_band_problem(dead_band, bands=(1, 5, 10)):
    """A problem where every region in `dead_band` has no reachable site.

    The travel matrix carries NaN for that band's regions (registered with
    `allow_missing=True`), which is what "entirely unreachable" means here:
    the band contributes no travel time at all, so it cannot take part in a
    comparison of travel times between bands.
    """
    ids = [f"LSOA_{i}" for i in range(len(bands))]
    demand_df = pd.DataFrame({"location_id": ids, "demand": [100] * len(ids)})
    candidate_df = pd.DataFrame({"site_id": ["Site_A"], "lat": [51.5], "long": [-0.1]})
    travel_df = pd.DataFrame(
        {
            "source_id": ids,
            "Site_A": [
                float("nan") if band == dead_band else 10.0 + 5.0 * i
                for i, band in enumerate(bands)
            ],
        }
    )
    equity_df = pd.DataFrame({"location_id": ids, "imd_decile": list(bands)})

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="deprivation",
        disadvantaged_end="low",
    )
    return problem


def test_stranded_most_deprived_band_is_not_reported_as_progressive():
    """The regression this guards is an *inverted* conclusion, not a wrong
    number: a solution leaving the whole most-deprived band with no
    reachable site used to be reported as "Highly Progressive" -- the best
    equity verdict there is.

    `weighted_by_equity_group` correctly stored NaN for the stranded band,
    but `min()`/`max()` over a plain list short-circuit on comparison, so
    the NaN propagated into `inter_tertile_ratio`; every comparison in the
    descriptor chain is False against NaN, and it fell through to the bare
    `else` branch, which is the "Highly Progressive" one."""
    problem = _stranded_band_problem(dead_band=1)

    with pytest.warns(
        UserWarning, match="leaves equity band 1 with no reachable region"
    ):
        result = problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0]
        )

    assert "Progressive" not in result.inter_tertile_desc
    assert "entirely unreachable" in result.gap_absolute_desc


def test_stranded_band_position_does_not_decide_the_metrics():
    """The defect was position-dependent, which is why it stayed hidden:
    Python's `min`/`max` are short-circuit comparisons, so a NaN absorbed
    the result only when it happened to sort first and was invisible
    otherwise. Stranding the *middle* band produced numbers byte-identical
    to a solution that stranded nobody.

    Every stranded position must now warn, name the band, and compute the
    spread over the remaining two bands alone. Costs are 10/15/20 for
    bands 1/5/10, so the spread over the two survivors is known exactly."""
    costs = {1: 10.0, 5: 15.0, 10: 20.0}
    for dead in (1, 5, 10):
        problem = _stranded_band_problem(dead_band=dead)
        with pytest.warns(UserWarning, match=f"leaves equity band {dead} with no"):
            result = problem.evaluate_single_solution_single_objective(
                objective="p_median", site_indices=[0]
            )
        survivors = [cost for band, cost in costs.items() if band != dead]
        assert result.gap_absolute_weighted == pytest.approx(
            max(survivors) - min(survivors)
        )
        assert result.gap_relative_weighted == pytest.approx(
            max(survivors) / min(survivors)
        )
        assert f"excludes band {dead}, entirely unreachable" in result.gap_absolute_desc
        # The stranded band is still reported as unreachable -- excluded
        # from the disparity comparison, not dropped from the analysis.
        assert result.regions_unreachable_by_equity_group[dead] == 1


def test_healthy_problem_gains_no_warning_and_no_exclusion_note(
    loaded_problem_with_equity,
):
    """The exclusion path must be inert when every band has a reachable
    region -- otherwise the fix would change every existing result."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = loaded_problem_with_equity.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 1]
        )

    assert "entirely unreachable" not in result.gap_absolute_desc
    assert "entirely unreachable" not in result.inter_tertile_desc
    assert result.inter_tertile_ratio == pytest.approx(10.0 / 15.0)


def test_all_bands_unreachable_have_no_disparity_metrics():
    from lokigi.site_solutions import _equity_band_metrics

    df = pd.DataFrame({"band": [1, 5, 10], "cost": [float("nan")] * 3})
    demand = pd.Series([100.0, 200.0, 150.0])
    metrics = _equity_band_metrics(
        df, "band", "cost", "within", demand, demand, df.cost.notna()
    )
    for metric in (
        "gap_absolute_weighted",
        "gap_relative_weighted",
        "inter_tertile_ratio",
    ):
        assert metrics[metric] is None
    for descriptor in ("gap_absolute_desc", "gap_relative_desc", "inter_tertile_desc"):
        assert metrics[descriptor] == "N/A (no equity band has a reachable region)"
    assert metrics["demand_unreachable_by_equity_group"] == {1: 100, 5: 200, 10: 150}


def _three_band_problem(travel, imd=(1, 5, 10)):
    """Three regions in equity bands 1/5/10, one per region, with
    `travel` mapping site id -> that site's cost to each region (NaN for
    unreachable)."""
    ids = ["L0", "L1", "L2"]
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(
        pd.DataFrame({"location_id": ids, "demand": [100] * 3}),
        demand_col="demand",
        location_id_col="location_id",
    )
    problem.add_sites(
        pd.DataFrame(
            {
                "site_id": list(travel),
                "lat": [51.5] * len(travel),
                "long": [-0.1] * len(travel),
            }
        ),
        candidate_id_col="site_id",
    )
    problem.add_travel_matrix(
        pd.DataFrame({"source_id": ids, **travel}),
        source_col="source_id",
        allow_missing=True,
    )
    problem.add_equity_data(
        pd.DataFrame({"location_id": ids, "imd": list(imd)}),
        equity_col="imd",
        common_col="location_id",
        label="deprivation",
    )
    return problem


def test_solve_does_not_warn_about_combinations_it_discards():
    """Greedy scores {A} on the way to {B, C}. {A} strands band 1, but the
    returned solution reaches every band -- so there is nothing to warn
    about. The warning used to fire from inside the per-combination
    metrics, describing a solution the caller never sees."""
    problem = _three_band_problem(
        {"A": [float("nan"), 10, 15], "B": [5, 6, 7], "C": [4, 9, 9]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = problem.solve(
            p=2, search_strategy="greedy", show_progress=False, unreachable_cost=1000
        )
    assert list(result.solution_df.iloc[0]["site_names"]) == ["B", "C"]


def test_solve_warns_once_naming_the_returned_ranks_that_strand_a_band():
    problem = _three_band_problem({"A": [float("nan"), 15, 20], "B": [5, 6, 7]})
    with pytest.warns(UserWarning) as record:
        problem.solve(
            p=1,
            search_strategy="brute-force",
            show_progress=False,
            unreachable_cost=1000,
        )
    stranded = [w for w in record if "no reachable region" in str(w.message)]
    assert len(stranded) == 1
    assert "1 of 2 returned solutions" in str(stranded[0].message)
    assert "rank 2: band 1" in str(stranded[0].message)
    # Attributed to the caller, not to lokigi's internals.
    assert stranded[0].filename == __file__


def _stranded_non_primary_problem():
    """Region L0 is unreachable. The primary dimension puts L0 in band 1
    alongside the reachable L1, so it strands nothing; the non-primary one
    gives L0 a band of its own, so that band is entirely unreachable."""
    problem = _three_band_problem({"A": [float("nan"), 15, 20]}, imd=(1, 1, 5))
    problem.add_equity_data(
        pd.DataFrame({"location_id": ["L0", "L1", "L2"], "rurality": [1, 5, 10]}),
        equity_col="rurality",
        common_col="location_id",
        label="rurality",
        primary=False,
    )
    return problem


def test_stranded_band_in_a_non_primary_dimension_warns_and_names_it():
    problem = _stranded_non_primary_problem()
    with pytest.warns(StrandedEquityBandWarning, match="band 1 in 'rurality'") as record:
        problem.solve(
            p=1,
            search_strategy="brute-force",
            show_progress=False,
            unreachable_cost=1000,
        )
    assert len(record) == 1
    assert "in 'deprivation'" not in str(record[0].message)


def test_stranded_band_warning_can_be_silenced_by_category():
    problem = _stranded_non_primary_problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=StrandedEquityBandWarning)
        problem.solve(
            p=1,
            search_strategy="brute-force",
            show_progress=False,
            unreachable_cost=1000,
        )
        problem.evaluate_single_solution_single_objective(site_indices=[0])


def test_stranded_band_warning_category_is_public_and_a_user_warning():
    assert lokigi.StrandedEquityBandWarning is StrandedEquityBandWarning
    assert issubclass(StrandedEquityBandWarning, UserWarning)
