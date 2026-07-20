"""Tests for `show_solutions(expand_dict_columns=..., inplace=...)`.

Dict-valued columns (`weighted_by_equity_group`, `coverage_by_equity_group`,
etc., and their `__<label>` secondary-matrix equivalents when
`full_secondary_metrics=True`) are awkward to consume directly -- e.g. they
can't be plotted or fed to `rank_on`/`ParetoMetric`. `expand_dict_columns`
flattens each one into `<column>__<key>` columns; `inplace` controls whether
that flattening is written back to `solution_df` or only returned.

Uses the shared `equity_df` fixture (imd_decile 1/5/10, one per demand
location) via `loaded_problem_with_equity`, and
`loaded_problem_with_equity_and_secondary_matrix` for the secondary-matrix
case, both already exercised by `test_secondary_travel_matrices.py`.
"""

import pytest


def _row_for_site(solution_df, site_name):
    mask = solution_df["site_names"].apply(lambda names: list(names) == [site_name])
    matches = solution_df[mask]
    assert len(matches) == 1, f"Expected exactly one row for {site_name}"
    return matches.iloc[0]


def test_default_leaves_dict_columns_as_dicts(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    df = result.show_solutions()
    assert isinstance(df["weighted_by_equity_group"].iloc[0], dict)


def test_expand_dict_columns_splits_into_one_column_per_key(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    df = result.show_solutions(expand_dict_columns=True)

    assert "weighted_by_equity_group" not in df.columns
    for key in (1, 5, 10):
        assert f"weighted_by_equity_group__{key}" in df.columns
    for base in (
        "unweighted_by_equity_group",
        "coverage_by_equity_group",
        "max_cost_by_equity_group",
    ):
        assert f"{base}__1" in df.columns
        assert f"{base}__5" in df.columns
        assert f"{base}__10" in df.columns


def test_expand_dict_columns_values_match_direct_dict_lookup(loaded_problem_with_equity):
    """Site_B's public_transport equivalent here is the primary matrix, so
    just cross-check the expanded columns against the un-expanded dict
    rather than hand-deriving new numbers -- both come from the same
    `_compute_travel_metrics` call, so any divergence is an expansion bug."""
    result = loaded_problem_with_equity.solve(p=1)
    raw_row = result.show_solutions().iloc[0]
    expanded_row = result.show_solutions(expand_dict_columns=True, rounding=None).iloc[0]

    for key, value in raw_row["weighted_by_equity_group"].items():
        assert expanded_row[f"weighted_by_equity_group__{key}"] == value


def test_expand_dict_columns_leaves_non_dict_columns_untouched(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    df = result.show_solutions(expand_dict_columns=True)

    assert "gap_absolute_weighted" in df.columns
    assert "site_names" in df.columns
    assert "problem_df" in df.columns


def test_expand_dict_columns_with_no_equity_data_drops_empty_columns(loaded_problem):
    """With no equity data registered, the dict-valued columns are all `{}`
    -- expanding them should make them disappear cleanly rather than error
    or leave a column of empty dicts."""
    result = loaded_problem.solve(p=1)
    df = result.show_solutions(expand_dict_columns=True)

    assert "weighted_by_equity_group" not in df.columns
    assert not any(col.startswith("weighted_by_equity_group__") for col in df.columns)
    # Untouched columns still present.
    assert "weighted_average" in df.columns


def test_expand_dict_columns_covers_secondary_matrix_dict_columns(
    loaded_problem_with_equity_and_secondary_matrix,
):
    result = loaded_problem_with_equity_and_secondary_matrix.solve(
        p=1, full_secondary_metrics=True
    )
    df = result.show_solutions(expand_dict_columns=True)

    assert "weighted_by_equity_group__public_transport" not in df.columns
    site_b_row = _row_for_site(df, "Site_B")
    assert site_b_row["weighted_by_equity_group__public_transport__1"] == 50.0
    assert site_b_row["weighted_by_equity_group__public_transport__5"] == 50.0
    assert site_b_row["weighted_by_equity_group__public_transport__10"] == 50.0


def test_expand_dict_columns_default_does_not_mutate_solution_df(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    result.show_solutions(expand_dict_columns=True)
    assert "weighted_by_equity_group" in result.solution_df.columns
    assert "weighted_by_equity_group__1" not in result.solution_df.columns


def test_inplace_persists_expansion_to_solution_df(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    result.show_solutions(expand_dict_columns=True, inplace=True)

    assert "weighted_by_equity_group" not in result.solution_df.columns
    assert "weighted_by_equity_group__1" in result.solution_df.columns

    # A later plain call sees the persisted expansion without re-requesting it.
    df = result.show_solutions()
    assert "weighted_by_equity_group__1" in df.columns


def test_inplace_without_expand_dict_columns_is_a_no_op(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    result.show_solutions(inplace=True)
    assert "weighted_by_equity_group" in result.solution_df.columns


def test_expand_dict_columns_respects_rounding_and_n_best(loaded_problem_with_equity):
    result = loaded_problem_with_equity.solve(p=1)
    df = result.show_solutions(expand_dict_columns=True, rounding=1, n_best=2)

    assert len(df) == 2
    value = df["weighted_by_equity_group__1"].iloc[0]
    assert value == pytest.approx(round(value, 1))
