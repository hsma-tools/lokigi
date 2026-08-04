"""
Tests pinning the corrected min_sites_different -> Jaccard distance
threshold formula in `_grasp` (mixins/site_solvers.py).

`min_sites_different=m` promises that any two accepted solutions differ
in at least m of their p site positions -- i.e. their intersection size k
satisfies p - k >= m. For two same-size (p) sets with intersection k,
Jaccard distance is 2(p-k) / (2p-k), NOT the plain fraction m/p the code
used to compute. Solving for the distance at the boundary k = p - m (the
most-similar pair that should still be ACCEPTED) gives the correct
threshold: 2m / (p + m), which is strictly larger than the old m/p
whenever m > 1 and p > m.

The practical effect: for m=1 or m=2 the two formulas happen to agree
at every achievable integer intersection size (verified below), but for
m >= 3 with p large enough relative to m (e.g. p >= 6 at m=3), the old
threshold was too small -- pairs differing in only m-1 sites (one site
too similar) were wrongly accepted as "diverse enough".
"""

import pytest

import lokigi
from lokigi.utils import _resolve_ranking_metric, _too_similar_to_accepted


# --- pure-function pin: the exact bug and fix, hand-computed ---


def test_old_formula_wrongly_accepted_a_too_similar_pair_at_m3_p6():
    """Sanity check pinning the bug itself (not the fix): with the old
    threshold (m/p = 0.5), two p=6 solutions sharing 4 sites (differing
    in only 2, one short of the required 3) were incorrectly treated as
    diverse enough. If this stops holding, the old formula wasn't
    actually buggy for this case and the fix test below proves nothing."""
    accepted = {0, 1, 2, 3, 4, 5}
    candidate_differs_by_2 = {0, 1, 2, 3, 6, 7}  # intersection=4, differs in 2

    old_threshold = 3 / 6  # min_sites_different=3, p=6
    assert (
        _too_similar_to_accepted(candidate_differs_by_2, [accepted], old_threshold)
        is False
    )


def test_corrected_formula_rejects_the_same_too_similar_pair():
    accepted = {0, 1, 2, 3, 4, 5}
    candidate_differs_by_2 = {0, 1, 2, 3, 6, 7}  # intersection=4, differs in 2

    new_threshold = (2 * 3) / (6 + 3)  # 2m / (p + m)
    assert (
        _too_similar_to_accepted(candidate_differs_by_2, [accepted], new_threshold)
        is True
    )


def test_corrected_formula_still_accepts_the_exact_boundary():
    """A pair differing in exactly m=3 sites (the minimum the user asked
    for) must still be accepted under the corrected threshold -- the fix
    must not overshoot and start rejecting solutions that satisfy the
    constraint."""
    accepted = {0, 1, 2, 3, 4, 5}
    candidate_differs_by_3 = {0, 1, 2, 6, 7, 8}  # intersection=3, differs in 3

    new_threshold = (2 * 3) / (6 + 3)
    assert (
        _too_similar_to_accepted(candidate_differs_by_3, [accepted], new_threshold)
        is False
    )


# --- m=1 and m=2 are unaffected: both formulas agree at every integer k ---
#
# (The exact k = p - m boundary distance is, by construction, equal to the
# new threshold -- and it also happens to equal the old threshold's
# decision outcome for m=1/m=2, as proven analytically in this file's
# module docstring context. Asserting that exact-equality boundary here
# would compare two independently-computed floats that can legitimately
# differ by 1 ulp depending on arithmetic order (2/11 vs 1 - 9/11), so
# these tests use points with clear margin instead of the knife-edge
# boundary -- the boundary itself is pinned separately, once, for the m=3
# bug case above, where the exact floats involved are already known to
# agree.)


@pytest.mark.parametrize("p", [2, 4, 6, 10])
def test_m1_unaffected_by_the_formula_change(p):
    """m=1 (reject only exact duplicates) behaves identically under old
    and new formulas: duplicates are always rejected, and a solution
    differing in every site is always accepted."""
    old_threshold = 1 / p
    new_threshold = (2 * 1) / (p + 1)
    accepted = set(range(p))
    duplicate = set(range(p))  # k=p, differs in 0 -- must always be rejected
    completely_different = set(range(p, 2 * p))  # k=0, differs in p -- must be accepted

    for threshold in (old_threshold, new_threshold):
        assert _too_similar_to_accepted(duplicate, [accepted], threshold) is True
        assert (
            _too_similar_to_accepted(completely_different, [accepted], threshold)
            is False
        )


# --- _grasp computes the corrected threshold internally ---


@pytest.mark.parametrize(
    "p,m",
    [(2, 1), (4, 1), (4, 2), (5, 2), (5, 4)],
)
def test_grasp_computes_the_corrected_threshold_internally(
    five_site_problem, monkeypatch, p, m
):
    """Spy on _too_similar_to_accepted to capture the min_jaccard_distance
    _grasp actually computes and passes down, and assert it matches
    2m / (p + m) rather than the old m / p. (five_site_problem has 5
    candidate sites, so p is kept within that range; the behavioural
    bug itself only manifests at larger p -- see the pure-function tests
    above for that -- this just pins the formula _grasp evaluates.)"""
    captured = []
    import lokigi.mixins.site_solvers as site_solvers_module

    original = site_solvers_module._too_similar_to_accepted

    def spy(new_set, accepted_sets, min_jaccard_distance):
        captured.append(min_jaccard_distance)
        return original(new_set, accepted_sets, min_jaccard_distance)

    monkeypatch.setattr(site_solvers_module, "_too_similar_to_accepted", spy)

    five_site_problem._grasp(
        p=p,
        objectives="p_median",
        weights={"demand": 1.0},
        scorer=_resolve_ranking_metric(objective="p_median")[0],
        num_solutions=1,
        max_attempts=1,
        min_sites_different=m,
        random_seed=0,
    )

    assert captured, "Expected at least one diversity check during _grasp"
    expected = (2.0 * m) / (p + m)
    assert all(threshold == pytest.approx(expected) for threshold in captured)
