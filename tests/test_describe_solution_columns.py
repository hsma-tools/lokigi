"""Tests for `SiteSolutionSet.describe_solution_columns()` -- a grouped,
beginner-facing view of `solution_df`'s columns (as opposed to
`show_solutions_colnames()`'s flat list).

"Coverage" and "Equity" are special-cased: their columns are always part
of `solution_df`'s schema (holding placeholder NaN/None/"N/A ..." values
when not applicable), unlike genuinely conditional groups like "Change vs
a baseline", so they're gated on whether they're actually meaningful
rather than on column presence.

Uses the shared `loaded_problem`/`loaded_problem_with_equity` fixtures
from conftest.py, also used by `test_show_solutions_expand.py` and
`test_show_solutions_summary.py`.
"""


def test_minimal_problem_shows_only_which_sites_travel_cost_and_underlying_data(
    loaded_problem,
):
    result = loaded_problem.solve(p=1)
    groups = result.describe_solution_columns(return_dict=True)

    assert set(groups.keys()) == {
        "Which sites",
        "Travel cost",
        "Underlying per-region data",
    }
    assert groups["Which sites"] == [
        "solution_rank",
        "site_names",
        "site_indices",
        "unselected_site_names",
    ]
    assert "problem_df" in groups["Underlying per-region data"]


def test_coverage_group_absent_without_threshold(loaded_problem):
    result = loaded_problem.solve(p=1)
    groups = result.describe_solution_columns(return_dict=True)
    assert "Coverage" not in groups


def test_coverage_group_present_with_threshold(loaded_problem):
    result = loaded_problem.solve(p=1, threshold_for_coverage=20)
    groups = result.describe_solution_columns(return_dict=True)
    assert "Coverage" in groups
    assert "demand_within_coverage_threshold" in groups["Coverage"]


def test_equity_group_absent_without_equity_data(loaded_problem):
    result = loaded_problem.solve(p=1)
    groups = result.describe_solution_columns(return_dict=True)
    assert "Equity" not in groups


def test_equity_group_present_with_equity_data(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    groups = result.describe_solution_columns(return_dict=True)
    assert "Equity" in groups
    assert "gap_absolute_weighted" in groups["Equity"]
    assert "weighted_by_equity_group" in groups["Equity"]


def test_change_vs_baseline_group_present_only_with_baseline(loaded_problem):
    result_no_baseline = loaded_problem.solve(p=1)
    groups_no_baseline = result_no_baseline.describe_solution_columns(return_dict=True)
    assert "Change vs a baseline" not in groups_no_baseline

    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    result_with_baseline = loaded_problem.solve(p=1, baseline=baseline)
    groups_with_baseline = result_with_baseline.describe_solution_columns(
        return_dict=True
    )
    assert "Change vs a baseline" in groups_with_baseline
    assert "demand_worsened" in groups_with_baseline["Change vs a baseline"]


def test_secondary_matrix_columns_classified_under_travel_cost(
    loaded_problem_with_secondary_matrix,
):
    result = loaded_problem_with_secondary_matrix.solve(p=1)
    groups = result.describe_solution_columns(return_dict=True)
    assert "weighted_average__public_transport" in groups["Travel cost"]


def test_beyond_threshold_columns_grouped_separately(loaded_problem):
    result = loaded_problem.solve(p=1, beyond_thresholds=[20])
    groups = result.describe_solution_columns(return_dict=True)
    assert "demand_beyond_threshold_20" in groups["Left behind (beyond a threshold)"]


def test_every_column_is_classified_somewhere(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1, threshold_for_coverage=20)
    groups = result.describe_solution_columns(return_dict=True)

    grouped_cols = {col for cols in groups.values() for col in cols}
    assert grouped_cols == set(result.solution_df.columns)
    assert "Other" not in groups


def test_return_dict_false_prints_and_returns_none(loaded_problem, capsys):
    result = loaded_problem.solve(p=1)
    output = result.describe_solution_columns()
    assert output is None

    captured = capsys.readouterr()
    assert "Which sites" in captured.out
    assert "site_names" in captured.out
