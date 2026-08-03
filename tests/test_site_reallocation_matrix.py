"""Tests for `SolutionComparator.site_reallocation_matrix()` -- a cross-tab
of each demand location's closest site under `set_a` against its closest
site under `set_b`, answering "where did a closed site's demand go?" and
"where did an added site's demand come from?" in one table.

Uses `loaded_problem` (Site_A/B/C, LSOA_1/2/3, demand=[100, 200, 150]; see
`tests/conftest.py`). Nearest-site assignment is hand-derived from its
`travel_df`:
    LSOA_1: A=10, B=25, C=30 -> A
    LSOA_2: A=20, B=5,  C=10 -> B
    LSOA_3: A=30, B=15, C=8  -> C
so with all three sites open, A serves LSOA_1 (100), B serves LSOA_2 (200),
and C serves LSOA_3 (150). Closing C (only A/B remain) re-derives to:
    LSOA_3: A=30, B=15 -> B
i.e. C's 150 moves to B; A and B's own regions are unaffected.
"""

import pytest

from lokigi.site_solutions import SolutionComparator


def test_closed_site_row_shows_where_its_demand_went(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])

    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix()

    assert list(result.index) == ["Site_A", "Site_B", "Site_C"]
    assert list(result.columns) == ["Site_A", "Site_B"]

    # Diagonal: A and B's own regions are unaffected by C's closure.
    assert result.loc["Site_A", "Site_A"] == pytest.approx(100.0)
    assert result.loc["Site_B", "Site_B"] == pytest.approx(200.0)
    # C's entire demand (LSOA_3, 150) moved to B; nothing went to A.
    assert result.loc["Site_C", "Site_B"] == pytest.approx(150.0)
    assert result.loc["Site_C", "Site_A"] == pytest.approx(0.0)
    # No cross-contamination between A and B's own regions.
    assert result.loc["Site_A", "Site_B"] == pytest.approx(0.0)
    assert result.loc["Site_B", "Site_A"] == pytest.approx(0.0)


def test_added_site_column_shows_where_its_demand_came_from(loaded_problem):
    """The exact reverse of the closure scenario above: opening Site_C
    back up takes its 150 back from Site_B, and Site_A is untouched."""
    two_site = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    three_site = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )

    comparator = SolutionComparator(two_site, three_site, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix()

    assert list(result.index) == ["Site_A", "Site_B"]
    assert list(result.columns) == ["Site_A", "Site_B", "Site_C"]

    assert result.loc["Site_A", "Site_A"] == pytest.approx(100.0)
    assert result.loc["Site_B", "Site_B"] == pytest.approx(200.0)
    # Site_C's column: its demand came entirely from Site_B, none from A.
    assert result.loc["Site_B", "Site_C"] == pytest.approx(150.0)
    assert result.loc["Site_A", "Site_C"] == pytest.approx(0.0)


def test_columns_sum_to_each_site_s_total_demand_in_set_b(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])

    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix()

    # Site_B now serves its own 200 plus Site_C's reallocated 150.
    assert result["Site_B"].sum() == pytest.approx(350.0)
    assert result["Site_A"].sum() == pytest.approx(100.0)


def test_by_regions_counts_locations_not_demand(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])

    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix(by="regions")

    # One demand location (LSOA_3) moved from C to B, regardless of its
    # demand weight of 150.
    assert result.loc["Site_C", "Site_B"] == 1
    assert result.loc["Site_A", "Site_A"] == 1
    assert result.loc["Site_B", "Site_B"] == 1


def test_index_and_column_names_are_the_comparator_labels(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])

    comparator = SolutionComparator(
        baseline, closed, labels=("Current network", "Proposed network")
    )
    result = comparator.site_reallocation_matrix()

    assert result.index.name == "Current network"
    assert result.columns.name == "Proposed network"


def test_invalid_by_raises(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))

    with pytest.raises(ValueError, match="by must be"):
        comparator.site_reallocation_matrix(by="places")


def test_by_regions_works_without_add_demand(basic_problem, candidate_df, travel_df):
    """`evaluate_baseline()` always auto-synthesizes equal demand even when
    `add_demand()` was never called (see `SiteProblem.evaluate_baseline`),
    so by="demand" requires demand data" can't actually be triggered via
    the public API -- that guard mirrors `site_allocation_summary()`'s own
    equally-unreachable one, kept for defensive symmetry. Only the
    reachable half is tested here: by="regions" still works with no
    add_demand() call at all."""
    basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
    basic_problem.add_travel_matrix(travel_df, source_col="source_id")

    baseline = basic_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = basic_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))

    result = comparator.site_reallocation_matrix(by="regions")
    assert result.loc["Site_C", "Site_B"] == 1


def test_demand_kwarg_passthrough(loaded_problem_with_secondary_demand):
    result_p1 = loaded_problem_with_secondary_demand.solve(p=1)
    result_p2 = loaded_problem_with_secondary_demand.solve(
        p=2, search_strategy="brute-force"
    )

    comparator = SolutionComparator(result_p1, result_p2, labels=("1-site", "2-site"))
    result = comparator.site_reallocation_matrix(demand="future_demand")
    assert result is not None
    assert result.shape[0] >= 1


def test_matrix_kwarg_passthrough(loaded_problem_with_secondary_matrix):
    result_p1 = loaded_problem_with_secondary_matrix.solve(p=1)
    result_p2 = loaded_problem_with_secondary_matrix.solve(
        p=2, search_strategy="brute-force"
    )

    comparator = SolutionComparator(result_p1, result_p2, labels=("1-site", "2-site"))
    result = comparator.site_reallocation_matrix(matrix="public_transport")
    assert result is not None
    assert result.shape[0] >= 1


def test_mismatched_demand_locations_raises(loaded_problem, five_site_problem):
    sols_a = loaded_problem.solve(p=1)
    sols_b = five_site_problem.solve(p=1)
    comparator = SolutionComparator(sols_a, sols_b, labels=("A", "B"))

    with pytest.raises(ValueError, match="different demand locations"):
        comparator.site_reallocation_matrix()


# --- Axis ordering: closed/opened sites at the end, not by candidate index -


def test_closed_site_sorts_last_even_though_it_is_canonically_first(
    five_site_problem,
):
    """Site_1 is canonically the FIRST candidate site, so a plain canonical-
    order listing would put it at the TOP of the row axis despite being the
    one that closed -- exactly the "Dawlish sorts to the end and looks like
    the interesting row" confusion in reverse. Persisting sites (2, 3) must
    come first, with the closed Site_1 pushed to the bottom regardless of
    its candidate index.

    Hand-derived from `five_site_problem`'s travel_df, nearest of {1,2,3}:
        LSOA_1: S1=38,S2=25,S3=24 -> S3 (24)
        LSOA_2: S1=18,S2=40,S3=13 -> S3 (13)
        LSOA_3: S1=10,S2=11,S3=29 -> S1 (10)
        LSOA_4: S1=28,S2=31,S3=13 -> S3 (13)
    so with {1,2,3} open: Site_1 serves LSOA_3 (100), Site_2 serves nothing,
    Site_3 serves LSOA_1/2/4 (300). Closing Site_1 (only {2,3} remain),
    nearest of {2,3}:
        LSOA_1: S2=25,S3=24 -> S3 (24)
        LSOA_2: S2=40,S3=13 -> S3 (13)
        LSOA_3: S2=11,S3=29 -> S2 (11)
        LSOA_4: S2=31,S3=13 -> S3 (13)
    so LSOA_3 (Site_1's only region) moves to Site_2; Site_3's regions are
    unaffected.
    """
    three_site = five_site_problem.evaluate_baseline(
        site_names=["Site_1", "Site_2", "Site_3"]
    )
    two_site = five_site_problem.evaluate_baseline(site_names=["Site_2", "Site_3"])

    comparator = SolutionComparator(three_site, two_site, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix()

    assert list(result.index) == ["Site_2", "Site_3", "Site_1"]
    assert list(result.columns) == ["Site_2", "Site_3"]
    assert result.loc["Site_1", "Site_2"] == pytest.approx(100.0)
    assert result.loc["Site_1", "Site_3"] == pytest.approx(0.0)
    assert result.loc["Site_3", "Site_3"] == pytest.approx(300.0)
    assert result.loc["Site_2", "Site_2"] == pytest.approx(0.0)
    assert result.loc["Site_2", "Site_3"] == pytest.approx(0.0)


def test_opened_site_column_sorts_last_even_though_it_is_canonically_first(
    five_site_problem,
):
    """The exact reverse of the closure scenario above: re-opening Site_1
    (canonically first) must still put its column LAST, after the
    persisting Site_2/Site_3 columns."""
    two_site = five_site_problem.evaluate_baseline(site_names=["Site_2", "Site_3"])
    three_site = five_site_problem.evaluate_baseline(
        site_names=["Site_1", "Site_2", "Site_3"]
    )

    comparator = SolutionComparator(two_site, three_site, labels=("Before", "After"))
    result = comparator.site_reallocation_matrix()

    assert list(result.index) == ["Site_2", "Site_3"]
    assert list(result.columns) == ["Site_2", "Site_3", "Site_1"]
    assert result.loc["Site_2", "Site_1"] == pytest.approx(100.0)
    assert result.loc["Site_3", "Site_3"] == pytest.approx(300.0)
    assert result.loc["Site_2", "Site_2"] == pytest.approx(0.0)
    assert result.loc["Site_3", "Site_1"] == pytest.approx(0.0)


# --- changed_only -----------------------------------------------------------


def test_changed_only_drops_unchanged_rows_and_columns_on_closure(loaded_problem):
    """Closing Site_C: Site_A and Site_B each keep 100% of their own
    patients (unchanged as a row), and Site_A's column receives nothing
    extra (unchanged as a column). Site_B's column DOES receive Site_C's
    reallocated demand, so it's kept even though Site_B's own row is
    dropped -- rows and columns are independent decisions, not "drop this
    site everywhere"."""
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))

    full = comparator.site_reallocation_matrix()
    assert list(full.index) == ["Site_A", "Site_B", "Site_C"]
    assert list(full.columns) == ["Site_A", "Site_B"]

    result = comparator.site_reallocation_matrix(changed_only=True)
    assert list(result.index) == ["Site_C"]
    assert list(result.columns) == ["Site_B"]
    assert result.loc["Site_C", "Site_B"] == pytest.approx(150.0)


def test_changed_only_drops_unchanged_rows_and_columns_on_addition(loaded_problem):
    """The exact reverse: re-opening Site_C, Site_A is fully unaffected
    (dropped both as row and column). Site_B's row is kept (some of its
    own patients moved to the new Site_C) even though Site_B's column is
    dropped (it received nothing new)."""
    two_site = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    three_site = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    comparator = SolutionComparator(two_site, three_site, labels=("Before", "After"))

    result = comparator.site_reallocation_matrix(changed_only=True)
    assert list(result.index) == ["Site_B"]
    assert list(result.columns) == ["Site_C"]
    assert result.loc["Site_B", "Site_C"] == pytest.approx(150.0)


def test_changed_only_defaults_to_false(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, closed, labels=("Before", "After"))

    default_result = comparator.site_reallocation_matrix()
    explicit_false = comparator.site_reallocation_matrix(changed_only=False)
    assert list(default_result.index) == list(explicit_false.index)
    assert list(default_result.columns) == list(explicit_false.columns)


def test_changed_only_empty_when_nothing_changed(loaded_problem):
    """Comparing a network against an identical copy of itself: every
    site is unchanged both as a row and a column, so changed_only=True
    returns an empty (but not erroring) DataFrame."""
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    same_again = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    comparator = SolutionComparator(baseline, same_again, labels=("Before", "After"))

    result = comparator.site_reallocation_matrix(changed_only=True)
    assert result.shape == (0, 0)
