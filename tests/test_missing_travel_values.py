"""Tests for treating a missing (NaN) travel cost as "no feasible journey"
rather than a crash or a silently-corrupting placeholder.

Before this, `add_travel_matrix()` had no NaN check at all, and a NaN
travel cost reaching `evaluate_single_solution_single_objective()` crashed
with `ValueError: Encountered all NA values` (from `DataFrame.idxmin`) for
any demand location where every selected site was unreachable -- or, if it
survived that, silently poisoned `weighted_average`/`unweighted_average`/
`90th_percentile`/`max` via plain `np.average`/`np.percentile`/`np.max` NaN
propagation, since a single unreachable row turned the whole solution's
averages to NaN.

Two-part API:
- `add_travel_matrix(allow_missing=True)` / `add_secondary_travel_matrix(
  allow_missing=True)`: opt in to registering a matrix that contains NaN.
  Default is still to reject it loudly (`ValueError`/`KeyError`), since an
  unnoticed NaN usually means an ID mismatch or a botched generation run.
- `treat_as_missing=<value or callable>`: convert an existing sentinel
  (e.g. `9999`) to a proper NaN before anything else runs.

`solve()` (search/optimisation) does not yet support a primary matrix that
actually contains NaN -- see `test_solve_rejects_nan_primary_matrix` --
deferred to a follow-up that adds an explicit unreachable-cost policy for
the objective. Secondary travel matrices are unaffected, since they never
drive optimisation.
"""

import numpy as np
import pandas as pd
import pytest

import lokigi
from lokigi.site_solutions import EvaluatedCombination


# --- Direct EvaluatedCombination tests (no travel matrix, no solve()) -----
# Mirrors test_evaluated_combination_metrics.py's pattern: a hand-built
# dataframe with a known answer, isolating _compute_travel_metrics from
# ingestion and search.


class _StubSiteProblem:
    def __init__(
        self, demand_col="demand", equity_col=None, equity_disadvantaged_end=None
    ):
        self._demand_data_demand_col = demand_col
        if equity_col is not None:
            self._equity_data_equity_col = equity_col
            self._equity_data_disadvantaged_end = equity_disadvantaged_end


def _build(df, coverage_threshold=None, site_problem=None):
    return EvaluatedCombination(
        solution_type="p_median",
        site_names=["A"],
        site_indices=[0],
        evaluated_combination_df=df,
        weights=None,
        site_problem=site_problem or _StubSiteProblem(),
        coverage_threshold=coverage_threshold,
    )


@pytest.fixture
def partly_unreachable_df():
    """Demand location 2 (demand=200) has no feasible journey (NaN
    min_cost) to the selected site."""
    return pd.DataFrame(
        {
            "min_cost": [10.0, np.nan, 30.0, 5.0],
            "demand": [100, 200, 150, 50],
            "within_threshold": [True, False, False, True],
        }
    )


def test_weighted_average_excludes_unreachable_rows(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    # (10*100 + 30*150 + 5*50) / (100+150+50), the 200-demand NaN row excluded
    assert result.weighted_average == pytest.approx((1000 + 4500 + 250) / 300)


def test_unweighted_average_excludes_unreachable_rows(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    assert result.unweighted_average == pytest.approx((10 + 30 + 5) / 3)


def test_max_excludes_unreachable_rows(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    assert result.max == 30.0


def test_percentile_90th_excludes_unreachable_rows(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    reachable = [10.0, 30.0, 5.0]
    assert result.percentile_90th == pytest.approx(np.percentile(reachable, q=90))


def test_regions_and_demand_unreachable_counts(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    assert result.regions_unreachable == 1
    assert result.demand_unreachable == pytest.approx(200.0)
    assert result.proportion_demand_unreachable == pytest.approx(200 / 500)


def test_unreachable_metrics_are_zero_when_nothing_missing():
    """A problem that never opts into allow_missing sees regions_
    unreachable=0 / demand_unreachable=0.0, not NaN or absent -- so the
    columns are always present and safe to compare against."""
    df = pd.DataFrame(
        {
            "min_cost": [10.0, 20.0],
            "demand": [100, 200],
            "within_threshold": [True, False],
        }
    )
    result = _build(df)
    assert result.regions_unreachable == 0
    assert result.demand_unreachable == 0.0
    assert result.proportion_demand_unreachable == 0.0


def test_all_rows_unreachable_gives_nan_averages_not_a_crash():
    df = pd.DataFrame(
        {
            "min_cost": [np.nan, np.nan],
            "demand": [100, 200],
            "within_threshold": [False, False],
        }
    )
    result = _build(df)
    assert np.isnan(result.weighted_average)
    assert np.isnan(result.unweighted_average)
    assert np.isnan(result.percentile_90th)
    assert np.isnan(result.max)
    assert result.regions_unreachable == 2
    assert result.demand_unreachable == pytest.approx(300.0)
    assert result.proportion_demand_unreachable == pytest.approx(1.0)


def test_return_solution_metrics_includes_unreachable_keys(partly_unreachable_df):
    result = _build(partly_unreachable_df)
    metrics = result.return_solution_metrics()
    assert metrics["regions_unreachable"] == 1
    assert metrics["demand_unreachable"] == pytest.approx(200.0)
    assert metrics["proportion_demand_unreachable"] == pytest.approx(200 / 500)


def test_unreachable_by_equity_group_and_band_isolation():
    """Band 2 is entirely unreachable; band 1 is entirely reachable. The
    unreachable band's own weighted average is NaN, but it must not poison
    band 1's -- the exact failure mode plain np.average propagation caused
    before _compute_travel_metrics restricted itself to reachable rows per
    band."""
    df = pd.DataFrame(
        {
            "min_cost": [10.0, 20.0, np.nan, np.nan],
            "demand": [100, 50, 200, 400],
            "within_threshold": [True, False, False, False],
            "equity_band": [1, 1, 2, 2],
        }
    )
    result = _build(
        df,
        site_problem=_StubSiteProblem(
            equity_col="equity_band", equity_disadvantaged_end="low"
        ),
    )

    assert result.regions_unreachable_by_equity_group == {1: 0, 2: 2}
    assert result.demand_unreachable_by_equity_group == {1: 0.0, 2: 600.0}

    assert result.weighted_by_equity_group[1] == pytest.approx(
        round((10 * 100 + 20 * 50) / 150, 2)
    )
    assert np.isnan(result.weighted_by_equity_group[2])

    # The global weighted_average is unaffected by band 2's NaNs.
    assert result.weighted_average == pytest.approx((10 * 100 + 20 * 50) / 150)


# --- add_travel_matrix() ingestion ----------------------------------------


@pytest.fixture
def demand_df():
    return pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "demand": [100, 200, 150]}
    )


@pytest.fixture
def candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
        }
    )


@pytest.fixture
def travel_df_with_nan():
    """Site_A and Site_B are both unreachable from LSOA_2; Site_C always
    reachable."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, np.nan, 30.0],
            "Site_B": [25.0, np.nan, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )


@pytest.fixture
def problem_with_demand_and_sites(demand_df, candidate_df):
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    return problem


def test_add_travel_matrix_rejects_nan_by_default(
    problem_with_demand_and_sites, travel_df_with_nan
):
    with pytest.raises(ValueError, match="allow_missing"):
        problem_with_demand_and_sites.add_travel_matrix(
            travel_df_with_nan, source_col="source_id"
        )


def test_add_travel_matrix_accepts_nan_with_allow_missing(
    problem_with_demand_and_sites, travel_df_with_nan
):
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df_with_nan, source_col="source_id", allow_missing=True
    )
    assert problem_with_demand_and_sites.travel_matrix["Site_A"].isna().sum() == 1


def test_treat_as_missing_scalar_converts_sentinel(
    problem_with_demand_and_sites, travel_df_with_nan
):
    sentinel_travel = travel_df_with_nan.fillna(9999)
    problem_with_demand_and_sites.add_travel_matrix(
        sentinel_travel,
        source_col="source_id",
        allow_missing=True,
        treat_as_missing=9999,
    )
    result = problem_with_demand_and_sites.travel_matrix
    assert result.loc[1, "Site_A"] is None or np.isnan(result.loc[1, "Site_A"])
    assert result.loc[1, "Site_B"] is None or np.isnan(result.loc[1, "Site_B"])
    # Untouched values survive unchanged.
    assert result.loc[0, "Site_A"] == 10.0


def test_treat_as_missing_callable_converts_sentinel(
    problem_with_demand_and_sites, travel_df_with_nan
):
    sentinel_travel = travel_df_with_nan.fillna(9999)
    problem_with_demand_and_sites.add_travel_matrix(
        sentinel_travel,
        source_col="source_id",
        allow_missing=True,
        treat_as_missing=lambda v: v >= 9000,
    )
    result = problem_with_demand_and_sites.travel_matrix
    assert np.isnan(result.loc[1, "Site_A"])
    assert np.isnan(result.loc[1, "Site_B"])


def test_treat_as_missing_without_allow_missing_still_raises(
    problem_with_demand_and_sites, travel_df_with_nan
):
    """treat_as_missing only normalises a sentinel to NaN -- it doesn't
    itself waive the allow_missing requirement."""
    sentinel_travel = travel_df_with_nan.fillna(9999)
    with pytest.raises(ValueError, match="allow_missing"):
        problem_with_demand_and_sites.add_travel_matrix(
            sentinel_travel, source_col="source_id", treat_as_missing=9999
        )


def test_treat_as_missing_applied_before_unit_conversion(
    problem_with_demand_and_sites,
):
    """The sentinel is matched in the caller's original units, not
    post-conversion -- otherwise a from_unit/to_unit conversion would
    silently change the sentinel's value out from under treat_as_missing."""
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [600.0, 9999.0, 1800.0],
            "Site_B": [1500.0, 300.0, 900.0],
            "Site_C": [1800.0, 600.0, 480.0],
        }
    )
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df,
        source_col="source_id",
        allow_missing=True,
        treat_as_missing=9999,
        from_unit="seconds",
        to_unit="minutes",
    )
    result = problem_with_demand_and_sites.travel_matrix
    assert np.isnan(result.loc[1, "Site_A"])
    # Genuine values still converted (seconds -> minutes, /60).
    assert result.loc[0, "Site_A"] == pytest.approx(10.0)
    assert result.loc[1, "Site_B"] == pytest.approx(5.0)


# --- evaluate_single_solution_single_objective(): the crash fix ----------


def test_evaluate_does_not_crash_when_every_selected_site_unreachable(
    problem_with_demand_and_sites, travel_df_with_nan
):
    """Regression test for `ValueError: Encountered all NA values` (from
    `DataFrame.idxmin`) -- LSOA_2 has no feasible journey to either
    selected site, so its row is entirely NaN. Prior to the fix in
    site.py's evaluate_single_solution_single_objective, this crashed
    rather than reporting an unreachable region."""
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df_with_nan, source_col="source_id", allow_missing=True
    )
    combo = problem_with_demand_and_sites.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"]
    )
    df = combo.evaluated_combination_df
    # LSOA_2 is unreachable for both selected sites.
    assert df["min_cost"].isna().sum() == 1
    assert combo.regions_unreachable == 1
    assert combo.demand_unreachable == pytest.approx(200.0)


def test_selected_site_is_missing_for_unreachable_row(
    problem_with_demand_and_sites, travel_df_with_nan
):
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df_with_nan, source_col="source_id", allow_missing=True
    )
    combo = problem_with_demand_and_sites.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"]
    )
    df = combo.evaluated_combination_df
    unreachable_row = df[df["min_cost"].isna()]
    assert len(unreachable_row) == 1
    assert pd.isna(unreachable_row["selected_site"].iloc[0])


def test_within_threshold_is_false_for_unreachable_row(
    problem_with_demand_and_sites, travel_df_with_nan
):
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df_with_nan, source_col="source_id", allow_missing=True
    )
    combo = problem_with_demand_and_sites.evaluate_single_solution_single_objective(
        site_names=["Site_A", "Site_B"], threshold_for_coverage=20
    )
    df = combo.evaluated_combination_df
    unreachable_row = df[df["min_cost"].isna()]
    assert unreachable_row["within_threshold"].iloc[0] == False  # noqa: E712


# --- solve() guard for a primary matrix with missing values ---------------


def test_solve_rejects_nan_primary_matrix(
    problem_with_demand_and_sites, travel_df_with_nan
):
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df_with_nan, source_col="source_id", allow_missing=True
    )
    with pytest.raises(NotImplementedError, match="solve"):
        problem_with_demand_and_sites.solve(p=1)


def test_solve_allows_allow_missing_matrix_with_no_actual_nan(
    problem_with_demand_and_sites,
):
    """allow_missing=True only widens what's *permitted* -- a matrix
    registered with it that happens to contain no NaN must solve()
    normally."""
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )
    problem_with_demand_and_sites.add_travel_matrix(
        travel_df, source_col="source_id", allow_missing=True
    )
    result = problem_with_demand_and_sites.solve(p=1)
    assert result.solution_df.iloc[0]["site_names"] == ["Site_B"]


# --- add_secondary_travel_matrix(allow_missing=True) ----------------------


@pytest.fixture
def secondary_travel_df_with_nan():
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_C": [5.0, np.nan, 5.0],
            "Site_B": [50.0, np.nan, 50.0],
            "Site_A": [90.0, np.nan, 90.0],
        }
    )


@pytest.fixture
def travel_df():
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )


def test_secondary_matrix_allow_missing_solves_without_raising(
    problem_with_demand_and_sites, travel_df, secondary_travel_df_with_nan
):
    problem_with_demand_and_sites.add_travel_matrix(travel_df, source_col="source_id")
    problem_with_demand_and_sites.add_secondary_travel_matrix(
        secondary_travel_df_with_nan,
        source_col="source_id",
        label="public_transport",
        allow_missing=True,
    )
    result = problem_with_demand_and_sites.solve(p=1)
    assert "weighted_average__public_transport" in result.solution_df.columns


def test_secondary_matrix_allow_missing_reports_unreachable_demand(
    problem_with_demand_and_sites, travel_df, secondary_travel_df_with_nan
):
    """LSOA_2 (demand=200) has no public_transport route to any site."""
    problem_with_demand_and_sites.add_travel_matrix(travel_df, source_col="source_id")
    problem_with_demand_and_sites.add_secondary_travel_matrix(
        secondary_travel_df_with_nan,
        source_col="source_id",
        label="public_transport",
        allow_missing=True,
    )
    result = problem_with_demand_and_sites.solve(p=1)
    row = result.solution_df.iloc[0]
    assert row["demand_unreachable__public_transport"] == pytest.approx(200.0)
    # Its own weighted_average (uniform 5/50/90 per site) is unaffected by
    # the unreachable LSOA_2 row, matching the primary matrix's treatment.
    assert row["weighted_average__public_transport"] in (5.0, 50.0, 90.0)
