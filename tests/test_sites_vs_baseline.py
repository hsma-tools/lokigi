"""Tests for `sites_closed_vs_baseline`/`sites_added_vs_baseline` on
`solution_df` -- which candidate sites differ between a solution and the
baseline passed to `solve(baseline=...)`, so a reader doesn't have to
compute the set difference between two `site_names` lists themselves.

Uses the shared `loaded_problem` fixture from conftest.py (Site_A/B/C,
LSOA_1/2/3), also exercised by `test_show_solutions_summary.py`.
"""

import pytest


def test_columns_present_and_correct_with_baseline(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    result = loaded_problem.solve(p=2, objectives="p_median", baseline=baseline)

    assert "sites_closed_vs_baseline" in result.solution_df.columns
    assert "sites_added_vs_baseline" in result.solution_df.columns

    baseline_names = {"Site_A", "Site_B"}
    for _, row in result.solution_df.iterrows():
        current_names = set(row["site_names"])
        assert set(row["sites_closed_vs_baseline"]) == baseline_names - current_names
        assert set(row["sites_added_vs_baseline"]) == current_names - baseline_names


def test_identical_solution_has_no_closures_or_additions(loaded_problem):
    """Evaluating exactly the baseline's own sites must show no change in
    either direction -- both diff columns empty."""
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    combo = loaded_problem.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"],
        baseline_costs=loaded_problem._resolve_baseline_costs(
            baseline, "p_median", {"demand": 1.0}, None
        ),
    )
    metrics = combo.return_solution_metrics()
    assert metrics["sites_closed_vs_baseline"] == []
    assert metrics["sites_added_vs_baseline"] == []


def test_completely_different_solution_closes_and_adds_everything(loaded_problem):
    """Baseline = {Site_A}, forced solution = {Site_B, Site_C} (via
    site_indices) -- entirely disjoint, so every baseline site is closed
    and every solution site is added."""
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A"])
    combo = loaded_problem.evaluate_single_solution_single_objective(
        site_names=["Site_B", "Site_C"],
        baseline_costs=loaded_problem._resolve_baseline_costs(
            baseline, "p_median", {"demand": 1.0}, None
        ),
    )

    assert combo.return_solution_metrics()["sites_closed_vs_baseline"] == ["Site_A"]
    assert set(combo.return_solution_metrics()["sites_added_vs_baseline"]) == {
        "Site_B",
        "Site_C",
    }


def test_columns_absent_without_baseline(loaded_problem):
    result = loaded_problem.solve(p=1, objectives="p_median")
    assert "sites_closed_vs_baseline" not in result.solution_df.columns
    assert "sites_added_vs_baseline" not in result.solution_df.columns


def test_show_solutions_summary_columns(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    result = loaded_problem.solve(
        p=2,
        objectives="p_median",
        baseline=baseline,
        search_strategy="brute-force",
        show_progress=False,
    )
    df = result.show_solutions_summary()
    raw = result.show_solutions(rounding=None)

    assert "Sites closed" in df.columns
    assert "Sites added" in df.columns
    for i in range(len(df)):
        assert df["Sites closed"].iloc[i] == ", ".join(
            raw["sites_closed_vs_baseline"].iloc[i]
        )
        assert df["Sites added"].iloc[i] == ", ".join(
            raw["sites_added_vs_baseline"].iloc[i]
        )


def test_show_solutions_summary_columns_absent_without_baseline(loaded_problem):
    result = loaded_problem.solve(p=1, objectives="p_median")
    df = result.show_solutions_summary()
    assert "Sites closed" not in df.columns
    assert "Sites added" not in df.columns


def test_describe_solution_columns_groups_under_change_vs_baseline(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    result = loaded_problem.solve(p=2, objectives="p_median", baseline=baseline)
    groups = result.describe_solution_columns(return_dict=True)

    assert "sites_closed_vs_baseline" in groups["Change vs a baseline"]
    assert "sites_added_vs_baseline" in groups["Change vs a baseline"]


def test_reserved_key_never_leaks_into_show_solutions(loaded_problem):
    """The sentinel key baseline_costs carries the baseline's site_names
    under must never survive into solution_df -- it's popped inside
    EvaluatedCombination.__init__ before the cost-Series validation loop,
    and must not appear as a column or inside any dict-valued column."""
    baseline = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    result = loaded_problem.solve(p=2, objectives="p_median", baseline=baseline)

    assert "__baseline_site_names__" not in result.solution_df.columns
