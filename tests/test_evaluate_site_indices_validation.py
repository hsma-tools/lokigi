"""
Tests pinning stricter validation of `site_indices`/`site_names` in
evaluate_single_solution_single_objective (site.py).

Previously, resolving `site_indices` used `.isin()` against
candidate_sites, which:

- Silently DROPPED any index that doesn't exist -- the only check was
  "is the resolved names list non-empty", which caught the all-invalid
  case but let a PARTIALLY-invalid list (e.g. one real index plus one
  nonexistent one) through, silently evaluating a smaller solution than
  the caller asked for (no error, no warning).
- Silently COLLAPSED duplicate indices to a single entry, for the same
  reason -- each candidate site appears once in candidate_sites no
  matter how many times its index is repeated in site_indices.

Separately, the "exactly one of site_names/site_indices" mutual-
exclusivity check used truthiness (`site_names and site_indices`)
instead of `is not None`, so an explicitly-passed empty list ([]) was
treated as "not provided" -- meaning site_names=[] silently bypassed
both the "neither provided" and "both provided" checks.

The fixed behaviour: any index in site_indices that doesn't exist in
candidate_sites raises IndexError naming exactly which ones; duplicate
indices raise ValueError; and an empty list for either parameter raises
ValueError rather than silently slipping through.
"""

import pytest

import lokigi


def test_partially_invalid_site_indices_raises_instead_of_silently_shrinking(
    loaded_problem,
):
    """The core bug: site_indices=[0, 999] (one real site, one that
    doesn't exist) used to silently evaluate just site 0 -- a smaller
    solution than requested, with no error."""
    with pytest.raises(IndexError, match=r"\[999\]"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 999]
        )


def test_entirely_invalid_site_indices_still_raises(loaded_problem):
    """Regression: the case that already worked (every index invalid)
    must keep working."""
    with pytest.raises(IndexError, match=r"\[999\]"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[999]
        )


def test_duplicate_site_indices_raises_instead_of_silently_collapsing(loaded_problem):
    """site_indices=[0, 0] used to silently resolve to a single site,
    treating a 2-site request as if p=1."""
    with pytest.raises(ValueError, match="duplicate"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[0, 0]
        )


def test_empty_site_indices_list_raises(loaded_problem):
    with pytest.raises(ValueError, match="empty"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[]
        )


def test_empty_site_names_list_raises(loaded_problem):
    """The empty-list gap applied to site_names too -- site_names=[]
    previously bypassed the mutual-exclusivity check entirely (an empty
    list is falsy, so `site_names and site_indices` never saw it as
    "provided")."""
    with pytest.raises(ValueError, match="empty"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_names=[]
        )


def test_valid_site_indices_still_work(loaded_problem):
    result = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_indices=[0, 1]
    )
    assert sorted(result.site_names) == ["Site_A", "Site_B"]


def test_valid_site_names_still_work(loaded_problem):
    result = loaded_problem.evaluate_single_solution_single_objective(
        objective="p_median", site_names=["Site_A"]
    )
    assert result.site_names == ["Site_A"]


def test_neither_site_names_nor_site_indices_raises(loaded_problem):
    with pytest.raises(ValueError, match="but not both"):
        loaded_problem.evaluate_single_solution_single_objective(objective="p_median")


def test_both_site_names_and_site_indices_raises(loaded_problem):
    with pytest.raises(ValueError, match="but not both"):
        loaded_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_names=["Site_A"], site_indices=[0]
        )
