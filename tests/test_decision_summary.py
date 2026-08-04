"""Tests for `SolutionComparator.decision_summary()` -- a single
stakeholder-facing paragraph bundling site changes, the population-impact
phrase, the hardest-hit places, and the candidate network's equity verdict,
so a decision-maker doesn't have to assemble those pieces by hand.

Uses `loaded_problem` (Site_A/B/C, LSOA_1/2/3, demand 100/200/150; see
`tests/conftest.py`). Nearest-site assignment, hand-derived from its
`travel_df`:
    LSOA_1: A=10, B=25, C=30 -> A
    LSOA_2: A=20, B=5,  C=10 -> B
    LSOA_3: A=30, B=15, C=8  -> C
so with all three sites open, A serves LSOA_1 (100), B serves LSOA_2 (200),
C serves LSOA_3 (150). Closing C (only A/B remain), LSOA_3 re-derives to B
(A=30, B=15 -> B): C's 150 people get a longer journey.
"""

import pytest

from lokigi.site_solutions import SolutionComparator


@pytest.fixture
def closure_comparator(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    return SolutionComparator(baseline, closed, labels=("Current", "Proposed"))


def test_names_the_closed_site(closure_comparator):
    summary = closure_comparator.decision_summary()
    assert "1 site would close: Site_C." in summary


def test_names_opened_sites_in_reverse_direction(loaded_problem):
    two_site = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    three_site = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    comparator = SolutionComparator(two_site, three_site, labels=("Current", "Proposed"))

    summary = comparator.decision_summary()
    assert "1 site would open: Site_C." in summary


def test_no_site_change_notes_it(loaded_problem):
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    same_again = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    comparator = SolutionComparator(baseline, same_again, labels=("Current", "Proposed"))

    summary = comparator.decision_summary()
    assert "No sites differ between the two networks compared here." in summary


def test_includes_population_impact_phrase(closure_comparator):
    summary = closure_comparator.decision_summary()
    assert closure_comparator.population_impact_phrase() in summary


def test_includes_worst_affected_place_by_name(closure_comparator):
    summary = closure_comparator.decision_summary()
    assert "Hit hardest:" in summary
    assert "LSOA_3" in summary
    # LSOA_3's nearest site is C (cost 8) while all three are open, and B
    # (cost 15) once C closes.
    assert "(8.0 -> 15.0)" in summary


def test_worst_affected_n_controls_how_many_places_are_named(loaded_problem):
    """Only one location genuinely worsens in this fixture, so
    worst_affected_n has nothing further to trim -- this just confirms the
    kwarg reaches through to population_impact_worst_affected() without
    erroring for a smaller n than the default."""
    baseline = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    comparator = SolutionComparator(baseline, closed, labels=("Current", "Proposed"))

    summary = comparator.decision_summary(worst_affected_n=1)
    assert "LSOA_3" in summary


def test_no_worsened_locations_omits_hit_hardest_clause(loaded_problem):
    """Adding a site back (the reverse of closure_comparator) leaves
    nobody worse off, so there's nothing to name -- the "Hit hardest"
    clause should be omitted entirely rather than printed empty."""
    two_site = loaded_problem.evaluate_baseline(site_names=["Site_A", "Site_B"])
    three_site = loaded_problem.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    comparator = SolutionComparator(two_site, three_site, labels=("Current", "Proposed"))

    summary = comparator.decision_summary()
    assert "Hit hardest" not in summary


def test_equity_verdicts_absent_without_equity_data(closure_comparator):
    summary = closure_comparator.decision_summary()
    assert "Equity in the proposed network" not in summary


def test_equity_verdicts_present_with_equity_data(loaded_problem_with_equity):
    baseline = loaded_problem_with_equity.evaluate_baseline(
        site_names=["Site_A", "Site_B", "Site_C"]
    )
    closed = loaded_problem_with_equity.evaluate_baseline(
        site_names=["Site_A", "Site_B"]
    )
    comparator = SolutionComparator(baseline, closed, labels=("Current", "Proposed"))

    summary = comparator.decision_summary()
    candidate = closed.solution_df.iloc[0]
    assert "Equity in the proposed network:" in summary
    assert candidate["gap_relative_description"] in summary
    assert candidate["inter_tertile_description"] in summary


def test_paragraphs_are_separated_by_blank_lines(closure_comparator):
    summary = closure_comparator.decision_summary()
    assert "\n\n" in summary
