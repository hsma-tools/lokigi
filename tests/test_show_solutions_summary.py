"""Tests for `show_solutions_summary()` -- a stakeholder-facing view of
solution_df with plain-English columns (units in the header, whole people,
site names joined into a readable string) that only shows a
coverage/baseline-impact/equity section when the underlying data was
actually registered, rather than showing it full of placeholders.

Uses the shared `loaded_problem`/`loaded_problem_with_equity` fixtures from
conftest.py (Site_A/B/C, LSOA_1/2/3), also exercised by
`test_show_solutions_expand.py`.
"""


def test_core_columns_always_present_no_jargon_or_problem_df(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    assert list(df.columns) == [
        "Rank",
        "Sites in this option",
        "Sites not in this option",
        "Average travel time (mins)",
        "Longest journey (mins)",
    ]
    assert "problem_df" not in df.columns
    assert "weighted_average" not in df.columns


def test_unselected_site_names_is_a_solution_df_column(loaded_problem):
    """unselected_site_names is computed once in return_solution_metrics()
    and lives on solution_df, not just inside show_solutions_summary() --
    so it's available to any caller of show_solutions(), not only the
    stakeholder view."""
    result = loaded_problem.solve(p=1)
    raw = result.show_solutions(rounding=None).iloc[0]

    assert "unselected_site_names" in result.solution_df.columns
    assert raw["unselected_site_names"] == ["Site_A", "Site_C"]


def test_sites_not_in_this_option_lists_remaining_candidates_in_canonical_order(
    loaded_problem,
):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    assert df["Sites in this option"].iloc[0] == "Site_B"
    assert df["Sites not in this option"].iloc[0] == "Site_A, Site_C"


def test_sites_not_in_this_option_empty_string_when_all_sites_selected(loaded_problem):
    result = loaded_problem.solve(p=3)
    df = result.show_solutions_summary()
    assert df["Sites not in this option"].iloc[0] == ""


def test_site_names_joined_into_readable_string_not_a_list(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    value = df["Sites in this option"].iloc[0]
    assert isinstance(value, str)
    assert value == "Site_B"


def test_values_match_underlying_solution_df_rounded(loaded_problem):
    result = loaded_problem.solve(p=1)
    raw = result.show_solutions(rounding=None).iloc[0]
    summary = result.show_solutions_summary().iloc[0]

    assert summary["Average travel time (mins)"] == round(raw["weighted_average"], 1)
    assert summary["Longest journey (mins)"] == round(raw["max"], 1)


def test_evaluate_baseline_has_no_rank_column(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    df = baseline.show_solutions_summary()

    assert "Rank" not in df.columns
    assert "solution_rank" not in baseline.show_solutions().columns


def test_solve_has_rank_column(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()
    assert "Rank" in df.columns


def test_coverage_columns_absent_without_threshold(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    assert not any(col.startswith("People within") for col in df.columns)
    assert not any(col.startswith("% within") for col in df.columns)


def test_coverage_columns_present_and_correct_with_threshold(loaded_problem):
    result = loaded_problem.solve(p=1, threshold_for_coverage=20)
    df = result.show_solutions_summary()
    raw = result.show_solutions(rounding=None).iloc[0]

    assert "People within 20 mins" in df.columns
    assert "% within 20 mins" in df.columns
    assert df["People within 20 mins"].iloc[0] == round(
        raw["demand_within_coverage_threshold"]
    )
    assert df["% within 20 mins"].iloc[0] == round(
        raw["proportion_within_coverage_threshold"] * 100, 1
    )


def test_baseline_impact_columns_absent_without_baseline(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    assert "People with a longer journey" not in df.columns
    assert "People with a shorter journey" not in df.columns


def test_baseline_impact_columns_present_with_baseline(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    result = loaded_problem.solve(p=1, baseline=baseline)
    df = result.show_solutions_summary()
    raw = result.show_solutions(rounding=None).iloc[0]

    assert df["People with a longer journey"].iloc[0] == round(
        raw["demand_worsened"]
    )
    assert df["People with a shorter journey"].iloc[0] == round(
        raw["demand_improved"]
    )


def test_nan_mean_reduction_filled_with_zero_when_nobody_improves(loaded_problem):
    """Site_B (p=1) is strictly worse for every LSOA than the full 3-site
    baseline, so demand_improved == 0 and the raw
    mean_reduction_among_improved is NaN (mean of an empty set) -- the
    summary should read 0.0, not NaN."""
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    result = loaded_problem.solve(p=1, baseline=baseline)
    raw = result.show_solutions(rounding=None).iloc[0]
    assert raw["demand_improved"] == 0
    import math

    assert math.isnan(raw["mean_reduction_among_improved"])

    df = result.show_solutions_summary()
    assert df["Avg reduction for them (mins)"].iloc[0] == 0.0
    assert not df.isna().any().any()


def test_equity_columns_absent_without_equity_data(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary()

    assert not any(col.startswith("Equity") for col in df.columns)


def test_equity_columns_present_with_equity_data(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    df = result.show_solutions_summary()
    raw = result.show_solutions(rounding=None).iloc[0]

    assert df["Equity gap (mins, best vs worst group)"].iloc[0] == round(
        raw["gap_absolute_weighted"], 1
    )
    assert (
        df["Equity verdict (best vs worst group)"].iloc[0]
        == raw["gap_relative_description"]
    )
    assert (
        df["Equity verdict (most vs least deprived third)"].iloc[0]
        == raw["inter_tertile_description"]
    )


def test_n_best_limits_rows(loaded_problem):
    result = loaded_problem.solve(p=1)
    df = result.show_solutions_summary(n_best=2)
    assert len(df) == 2


def test_does_not_mutate_solution_df(loaded_problem):
    result = loaded_problem.solve(p=1)
    result.show_solutions_summary()
    assert "problem_df" in result.solution_df.columns
    assert "Sites in this option" not in result.solution_df.columns
