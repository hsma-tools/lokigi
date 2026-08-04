from lokigi.utils import (
    _RANK_SCORE_COL,
    _add_rank_score_column,
    _generate_all_combinations,
    _get_required_site_indices,
    _too_similar_to_accepted,
    _apply_cost_weighting,
)

from lokigi.site_solutions import SiteSolutionSet

# Data manipulation imports
import pandas as pd
import random
import math

# Other imports
from warnings import warn
import heapq
from tqdm.auto import tqdm
from functools import lru_cache
from joblib import Parallel, delayed, effective_n_jobs

# Warn if brute force will be slow
BRUTE_FORCE_WARN_THRESHOLD = 75_000
BRUTE_FORCE_LIMIT = 500_000


def _push_capped(heap, key, tie, metrics, n):
    """Fill-then-replace a size-capped heap, keeping the n largest keys.
    `tie` only breaks comparisons between exactly-equal-key heap entries
    during internal heap sifting (metrics dicts aren't orderable) -- it is
    NOT part of the admission decision, matching the pre-parallelisation
    behaviour exactly: an equal-key newcomer is never admitted once the
    heap is full, regardless of tie. n_jobs=1 always evaluates the whole
    combination list as a single chunk/heap (see _brute_force), so this
    reproduces that prior behaviour bit-for-bit. n_jobs>1 evaluates
    several of these heaps independently (one per chunk) and merges them
    the same way; the merged result is always a valid, correctly-ranked
    top n by score, but on an exact score tie that spans more combinations
    than n allows, which specific combination survives can differ from a
    single-process run -- an edge case only reachable with genuinely
    tied floating-point scores in excess of keep_best_n/keep_worst_n.
    """
    if len(heap) < n:
        heapq.heappush(heap, (key, tie, metrics))
    elif key > heap[0][0]:
        heapq.heapreplace(heap, (key, tie, metrics))


def _evaluate_chunk(
    site_problem,
    indexed_chunk,
    objectives,
    weights,
    threshold_for_coverage,
    max_value_cutoff,
    scorer,
    keep_best_n,
    keep_worst_n,
    stream_top_n,
    full_secondary_metrics=False,
    baseline_costs=None,
    meaningful_change_threshold=0.0,
    beyond_thresholds=None,
    unreachable_cost=None,
):
    """Evaluate one chunk of combinations. Returns either the list of
    metrics dicts (materialising path) or (local_best, local_worst) lists
    of (key, global_index, metrics) tuples (streaming path). Runs in a
    worker process under joblib, so it must not depend on any state that
    changes across chunks (it doesn't -- the heaps below are local).
    `baseline_costs` (a dict of small pd.Series, see `solve()`'s `baseline`
    parameter) pickles cheaply across the worker-process boundary -- it is
    resolved once per `solve()` call, not once per combination.

    The `max_value_cutoff` feasibility check reads `max_for_ranking`, not
    `max` -- identical to `max` unless `unreachable_cost` is set, in which
    case a combination that strands demand behind an unreachable pair is
    correctly judged against the assumed cost of that failure, not judged
    as if the stranded rows were never there at all."""
    if not stream_top_n:
        chunk_outputs = []
        for _global_index, possible_solution in indexed_chunk:
            metrics = site_problem.evaluate_single_solution_single_objective(
                site_indices=possible_solution,
                objective=objectives,
                threshold_for_coverage=threshold_for_coverage,
                weights=weights,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

            if (
                max_value_cutoff is None
                or metrics["max_for_ranking"] <= max_value_cutoff
            ):
                chunk_outputs.append(metrics)
        return chunk_outputs

    local_best = []
    local_worst = []
    for global_index, possible_solution in indexed_chunk:
        metrics = site_problem.evaluate_single_solution_single_objective(
            site_indices=possible_solution,
            objective=objectives,
            threshold_for_coverage=threshold_for_coverage,
            weights=weights,
            baseline_costs=baseline_costs,
            meaningful_change_threshold=meaningful_change_threshold,
            beyond_thresholds=beyond_thresholds,
            unreachable_cost=unreachable_cost,
        ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

        # `scorer.normalise` returns a value where lower is always better,
        # whatever the ranking column's own direction -- so everything below
        # this point is a plain minimisation. It's a per-element transform
        # (negate, or distance-from-target), needing no view of the other
        # combinations, which is what lets the streaming heap survive a
        # custom `rank_on` where batch-relative cost weighting cannot.
        score = float(scorer.normalise(metrics[scorer.column]))
        max_value = metrics["max_for_ranking"]

        if max_value_cutoff is not None and max_value > max_value_cutoff:
            continue

        if keep_best_n is not None:
            _push_capped(local_best, -score, global_index, metrics, keep_best_n)
        if keep_worst_n is not None:
            _push_capped(local_worst, score, global_index, metrics, keep_worst_n)

    return local_best, local_worst


class BruteForceMixin:
    def _brute_force(
        self,
        p: int,
        objectives,
        weights,
        brute_force_ignore_limit: bool = False,
        show_progress: bool = False,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        scorer=None,
        max_value_cutoff=None,
        threshold_for_coverage=None,
        n_jobs=1,
        full_secondary_metrics=False,
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
    ):

        # Greedy and GRASP already fail fast with a clear message when
        # required sites can't fit in p; brute-force had no equivalent
        # check and instead silently filtered every combination out in
        # _generate_all_combinations below, falling through to the
        # generic "No feasible solutions" guard in site.py -- whose
        # wording ("checking that p does not exceed the number of
        # candidate sites") is wrong for this case.
        required_site_indices = _get_required_site_indices(self)
        if len(required_site_indices) > p:
            raise ValueError(
                f"{len(required_site_indices)} sites are marked as required "
                f"in '{self._candidate_sites_required_sites_col}', but p={p}. "
                "Increase p to at least the number of required sites."
            )

        keep_n_active = (
            brute_force_keep_best_n is not None or brute_force_keep_worst_n is not None
        )

        # _apply_cost_weighting does batch-relative min-max normalization: it
        # needs to see every combination's ranking value and total_cost at
        # once to normalise either meaningfully. That's incompatible with
        # keep_best_n/keep_worst_n's streaming heap below, which prunes to N
        # using the raw, pre-cost ranking value one combination at a time --
        # a combination with a mediocre raw ranking but a low cost could be
        # the true cost-weighted best, yet never survive the heap because
        # cost is never considered until after _brute_force returns. So
        # whenever cost weighting is actually requested, keep_best_n/
        # keep_worst_n fall back to materialising every combination (the
        # same way the "keep everything" path already does) and only prune
        # AFTER cost has been blended in over the full batch. This trades
        # away keep_best_n/keep_worst_n's memory-saving benefit for
        # correctness whenever cost weighting is active; BRUTE_FORCE_WARN_
        # THRESHOLD/BRUTE_FORCE_LIMIT above still guard against unbounded
        # memory use either way.
        use_cost_weighting = bool(weights) and weights.get("cost", 0) > 0
        stream_top_n = keep_n_active and not use_cost_weighting

        if keep_n_active and use_cost_weighting:
            warn(
                "brute_force_keep_best_n/brute_force_keep_worst_n was "
                "requested together with a cost weight (weights['cost']). "
                "Pruning to the top/bottom N using the raw ranking value, "
                "before cost is blended in, could discard the true cost-"
                "weighted best/worst combination. To stay correct, every "
                "combination is being evaluated and held in memory so cost "
                "weighting can be applied over the full set before pruning "
                "-- this forgoes keep_best_n/keep_worst_n's usual memory-"
                "saving benefit for this run.",
                UserWarning,
                stacklevel=2,
            )

        if stream_top_n and brute_force_keep_best_n is not None:
            top_n_heap = []  # To store the smallest scores (best)
            # print(f"Keeping top {brute_force_keep_best_n}")
        if stream_top_n and brute_force_keep_worst_n is not None:
            bottom_n_heap = []  # To store the largest scores (worst)
            # print(f"Keeping worst {brute_force_keep_worst_n}")

        possible_combinations = _generate_all_combinations(
            n_facilities=self.total_n_sites, p=p, site_problem=self
        )

        if len(possible_combinations) > BRUTE_FORCE_LIMIT:
            if not brute_force_ignore_limit:
                raise MemoryError(
                    f"You are trying to evaluate {len(possible_combinations):,d} combinations via brute force. The limit is {BRUTE_FORCE_LIMIT:,d}. Please try a different solver."
                )
            else:
                warn(
                    f"You are trying to evaluate {len(possible_combinations):,d} combinations via brute force and have opted to ignore the advised limit of {BRUTE_FORCE_LIMIT:,d} combinations. This could take a while!"
                )
        elif len(possible_combinations) > BRUTE_FORCE_WARN_THRESHOLD:
            warn(
                f"You are trying to evaluate {len(possible_combinations):,d} combinations via brute force. The recommended maximum is {BRUTE_FORCE_WARN_THRESHOLD:,d}. This could take a while! You may wish to try a different solver."
            )

        outputs = []

        # Chunk the (already fully materialised) combination list and hand
        # chunks to worker processes. Each combination's position in
        # `possible_combinations` doubles as its global_index, used only
        # to keep heap comparisons from falling through to comparing two
        # (unorderable) metrics dicts on an exact score tie.
        #
        # n_jobs=1 always evaluates as a single chunk, so it goes through
        # exactly the same single-heap computation the pre-parallelisation
        # code did -- bit-for-bit identical output, including which
        # combination wins an exact tie. n_jobs>1 splits into several
        # chunks/heaps merged afterwards; see _push_capped for what that
        # does and does not guarantee under exact ties.
        n_combinations = len(possible_combinations)
        if n_combinations == 0:
            chunks = []
        elif n_jobs == 1:
            chunks = [list(enumerate(possible_combinations))]
        else:
            n_workers = max(1, effective_n_jobs(n_jobs))
            n_chunks = min(n_combinations, max(1, n_workers * 4))
            chunk_size = math.ceil(n_combinations / n_chunks)
            indexed_combinations = list(enumerate(possible_combinations))
            chunks = [
                indexed_combinations[i : i + chunk_size]
                for i in range(0, n_combinations, chunk_size)
            ]

        progress = tqdm(total=n_combinations) if show_progress else None

        parallel_results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_evaluate_chunk)(
                self,
                chunk,
                objectives,
                weights,
                threshold_for_coverage,
                max_value_cutoff,
                scorer,
                brute_force_keep_best_n if stream_top_n else None,
                brute_force_keep_worst_n if stream_top_n else None,
                stream_top_n,
                full_secondary_metrics,
                baseline_costs,
                meaningful_change_threshold,
                beyond_thresholds,
                unreachable_cost,
            )
            for chunk in chunks
        )

        for chunk, result in zip(chunks, parallel_results):
            if progress is not None:
                progress.update(len(chunk))

            if not stream_top_n:
                # Keep all results -- either because no pruning was
                # requested, or because cost weighting is active and every
                # combination must be materialised before pruning (see
                # `use_cost_weighting` above).
                outputs.extend(result)
            else:
                local_best, local_worst = result
                if brute_force_keep_best_n is not None:
                    for key, tie, metrics in local_best:
                        _push_capped(
                            top_n_heap,
                            key,
                            tie,
                            metrics,
                            brute_force_keep_best_n,
                        )
                if brute_force_keep_worst_n is not None:
                    for key, tie, metrics in local_worst:
                        _push_capped(
                            bottom_n_heap,
                            key,
                            tie,
                            metrics,
                            brute_force_keep_worst_n,
                        )

        if progress is not None:
            progress.close()

        if not keep_n_active:
            return outputs

        if use_cost_weighting:
            # `outputs` was fully materialised above (the `not stream_top_n`
            # branch), so cost weighting can be applied once, correctly,
            # over the whole batch, and only then pruned to N.
            if not outputs:
                return []

            outputs_df = pd.DataFrame(outputs)
            outputs_df, score_col = _apply_cost_weighting(
                outputs_df,
                ranking_col=scorer.column,
                weights=weights,
                scorer=scorer,
            )

            # After this, "best" is unambiguously "smallest" either way:
            # _apply_cost_weighting's blended "composite_score" is already on
            # a lower-is-better scale, and when it no-ops (no usable cost
            # data, so score_col comes back unchanged) `_RANK_SCORE_COL`
            # puts the raw ranking column onto that same scale via the
            # scorer. Deciding direction from the objective instead, as this
            # used to, could not express a `closest_to_target` rank_on.
            if score_col == scorer.column:
                outputs_df, score_col = _add_rank_score_column(outputs_df, scorer)

            result_frames = []
            if brute_force_keep_best_n is not None:
                result_frames.append(
                    outputs_df.nsmallest(brute_force_keep_best_n, score_col)
                )
            if brute_force_keep_worst_n is not None:
                result_frames.append(
                    outputs_df.nlargest(brute_force_keep_worst_n, score_col)
                )

            combined = (
                pd.concat(result_frames) if len(result_frames) > 1 else result_frames[0]
            )
            return combined.drop(columns=_RANK_SCORE_COL, errors="ignore").to_dict(
                "records"
            )
        else:
            # Reconstruct the 'outputs' list
            # Extract dictionaries from heaps and sort them
            if brute_force_keep_best_n is not None:
                best_list = [item[2] for item in sorted(top_n_heap, key=lambda x: x[0])]

            if brute_force_keep_worst_n is not None:
                worst_list = [
                    item[2] for item in sorted(bottom_n_heap, key=lambda x: x[0])
                ]

            if brute_force_keep_best_n is not None and brute_force_keep_worst_n is None:
                return best_list
            elif (
                brute_force_keep_worst_n is not None and brute_force_keep_best_n is None
            ):
                return worst_list
            else:
                return best_list + worst_list


class GreedyMixin:
    def _greedy(
        self,
        p: int,
        objectives,
        weights,
        scorer,
        show_progress: bool = False,
        threshold_for_coverage=None,
        max_value_cutoff=None,
        full_secondary_metrics=False,
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
    ):
        # Greedy grows the solution one site at a time, but every size-i
        # combination is also filtered to contain ALL required sites -- so
        # with two or more required sites, the early steps (i < n_required)
        # had no valid combinations at all and crashed on the empty result.
        # Seed the build with the required sites instead, and start the
        # loop at that size (its first step then just evaluates the
        # required set itself before free choices begin).
        required_site_indices = _get_required_site_indices(self)
        if len(required_site_indices) > p:
            raise ValueError(
                f"{len(required_site_indices)} sites are marked as required "
                f"in '{self._candidate_sites_required_sites_col}', but p={p}. "
                "Increase p to at least the number of required sites."
            )

        best_indices = list(required_site_indices)

        loop_iterations = range(max(1, len(required_site_indices)), p + 1)
        if show_progress:
            loop_iterations = tqdm(loop_iterations)

        for i in loop_iterations:
            # print(f"Loop {i}")
            possible_combinations = _generate_all_combinations(
                n_facilities=self.total_n_sites,
                p=i,
                site_problem=self,
                force_include_indices=list(best_indices) if best_indices else None,
            )

            # print(f"Possible combinations: {possible_combinations}")

            outputs = []

            for possible_solution in possible_combinations:
                # print(f"Evaluating possible solution: {possible_solution}")
                outputs.append(
                    self.evaluate_single_solution_single_objective(
                        site_indices=possible_solution,
                        objective=objectives,
                        threshold_for_coverage=threshold_for_coverage,
                        weights=weights,
                        baseline_costs=baseline_costs,
                        meaningful_change_threshold=meaningful_change_threshold,
                        beyond_thresholds=beyond_thresholds,
                        unreachable_cost=unreachable_cost,
                    ).return_solution_metrics(
                        full_secondary_metrics=full_secondary_metrics
                    )
                )

            outputs_df = pd.DataFrame(outputs)

            # The max-value cutoff (hybrid objectives) is a constraint on
            # the FINAL solution only: with fewer than p sites the
            # worst-case travel is usually still shrinking, so filtering
            # intermediate steps would wrongly rule everything out. At the
            # final step, keep only combinations meeting the cutoff so the
            # guarantee the hybrid objectives promise actually holds for
            # whatever is returned.
            if max_value_cutoff is not None and i == p:
                outputs_df = outputs_df[
                    outputs_df["max_for_ranking"] <= max_value_cutoff
                ]
                if len(outputs_df) == 0:
                    raise ValueError(
                        f"Greedy search found no combination of {p} sites "
                        f"meeting max_value_cutoff={max_value_cutoff}, given "
                        "the sites fixed at earlier steps "
                        f"({sorted(int(s) for s in best_indices)}). Greedy "
                        "never revisits earlier choices, so a feasible "
                        "solution may still exist -- try search_strategy="
                        "'brute-force' or 'grasp', or relax the cutoff."
                    )

            score_col = scorer.column
            if weights and weights.get("cost", 0) > 0:
                outputs_df, score_col = _apply_cost_weighting(
                    outputs_df,
                    ranking_col=scorer.column,
                    weights=weights,
                    scorer=scorer,
                )

            # _apply_cost_weighting only blends onto its lower-is-better
            # "composite_score" scale when it actually has usable cost data
            # to blend (a positive cost weight is not enough on its own --
            # it also no-ops, returning score_col unchanged, when e.g. every
            # candidate's total_cost is NaN). Whenever it hasn't blended, put
            # the raw ranking column onto that same lower-is-better scale via
            # the scorer, so the sort below is unconditionally ascending. The
            # old `not higher_is_better` flag could only express a two-way
            # split, so it had no way to rank on a `closest_to_target` metric.
            if score_col == scorer.column:
                outputs_df, score_col = _add_rank_score_column(outputs_df, scorer)

            evaluated_solutions = outputs_df.sort_values(
                [score_col, "weighted_average"],
                ascending=[True, True],
                kind="mergesort",
            ).drop(columns=_RANK_SCORE_COL, errors="ignore")

            # print("==Evaluated solution dataframe==")
            # print(evaluated_solutions)

            single_solution_metrics = SiteSolutionSet(
                solution_df=evaluated_solutions,
                site_problem=self,
                objectives=objectives,
                n_sites=i,
            )

            # print("Single Solution Set object created")
            # print(single_solution_metrics)
            # print(single_solution_metrics.show_solutions())

            best_indices = single_solution_metrics.show_solutions().head(1)[
                "site_indices"
            ][0]

            if show_progress:
                print(
                    f"Best combination for {i} sites: {[int(i) for i in best_indices]}"
                )

        best_solution_metrics = self.evaluate_single_solution_single_objective(
            site_indices=best_indices,
            objective=objectives,
            threshold_for_coverage=threshold_for_coverage,
            weights=weights,
            baseline_costs=baseline_costs,
            meaningful_change_threshold=meaningful_change_threshold,
            beyond_thresholds=beyond_thresholds,
            unreachable_cost=unreachable_cost,
        ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

        return [best_solution_metrics]


class GraspMixin:
    def _grasp(
        self,
        p: int,
        objectives,
        weights,
        scorer,
        num_solutions: int = 5,
        show_progress: bool = False,
        threshold_for_coverage=None,
        alpha: float = 0.2,
        random_seed: int = 42,
        max_attempts: int | str = "default",
        min_sites_different: int = 1,
        local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        max_swap_count_local_search=10,
        max_value_cutoff=None,
        full_secondary_metrics=False,
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
    ):
        """
        GRASP (Greedy Randomised Adaptive Search Procedure) for finding multiple
        near-optimal facility location solutions.

        Every comparison below runs on `scorer.normalise()`'d values, where
        lower is always better -- so this is unconditionally a minimisation.
        The `is_minimization` flag this replaced could only express a
        two-way split, and so could not represent a `closest_to_target`
        `rank_on` (an inter-tertile ratio is best at 1.0, with both larger
        and smaller values worse).
        """
        rng = random.Random(random_seed)
        ranking = scorer.column
        all_site_indices = list(range(self.total_n_sites))

        # Brute force and greedy enforce required_sites_col through
        # _generate_all_combinations, but GRASP builds solutions
        # incrementally and never calls it -- so required sites must be
        # pinned here: seeded into every construction, and protected from
        # being swapped out during local search.
        required_site_indices = _get_required_site_indices(self)
        required_site_set = set(required_site_indices)
        if len(required_site_indices) > p:
            raise ValueError(
                f"{len(required_site_indices)} sites are marked as required "
                f"in '{self._candidate_sites_required_sites_col}', but p={p}. "
                "Increase p to at least the number of required sites."
            )

        # min_sites_different (m) means "any two accepted solutions must
        # differ in at least m of their p site positions", i.e. their
        # intersection size k must satisfy p - k >= m, i.e. k <= p - m.
        # For two same-size (p) sets with intersection k, Jaccard distance
        # is 2(p-k) / (2p-k) -- NOT the plain fraction m/p this used to use.
        # Solving for the distance at the exact boundary k = p - m (the
        # most-similar pair that should still be ACCEPTED) gives
        # 2m / (p + m), which is strictly larger than m/p whenever m > 1
        # and p > m. The old, smaller m/p threshold rejected fewer
        # candidates than intended: for m >= 3 (and p large enough
        # relative to m, e.g. p >= 6 at m=3), pairs differing in only m-1
        # sites -- one site too similar -- were wrongly accepted as
        # "diverse enough".
        min_jaccard_distance = (2.0 * float(min_sites_different)) / (
            float(p) + float(min_sites_different)
        )

        # Only the non-required slots are free to vary, so that's the true
        # size of the search space the attempt budget is drawn from.
        total_combinations = math.comb(
            self.total_n_sites - len(required_site_indices),
            p - len(required_site_indices),
        )
        if max_attempts == "default":
            max_attempts = min(num_solutions * 20, total_combinations)

        final_solutions_metrics = []
        accepted_solution_sets: list[set] = []
        attempts = 0

        # -------------------------------------------------------------------
        # CACHING: Memoize evaluations to prevent redundant compute.
        # Uses a tuple of sorted indices as a canonical, hashable key.
        # -------------------------------------------------------------------
        @lru_cache(maxsize=10000)
        def _get_cached_metrics(indices_tuple: tuple):
            return self.evaluate_single_solution_single_objective(
                site_indices=list(indices_tuple),
                objective=objectives,
                threshold_for_coverage=threshold_for_coverage,
                weights=weights,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

        pbar = None
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(
                total=num_solutions,
                desc=f"Finding {num_solutions} diverse solutions (max {max_attempts} attempts)",
            )

        while len(final_solutions_metrics) < num_solutions and attempts < max_attempts:
            attempt_rng = random.Random(rng.randint(0, 2**32 - 1))
            attempts += 1

            # ---------------------------------------------------------------
            # CONSTRUCTION PHASE
            # ---------------------------------------------------------------
            current_solution: list[int] = list(required_site_indices)
            current_solution_set: set[int] = set(current_solution)
            construction_failed = False

            for step in range(p - len(required_site_indices)):
                remaining_sites = [
                    s for s in all_site_indices if s not in current_solution_set
                ]

                if not remaining_sites:
                    construction_failed = True
                    break

                candidate_rows = []
                for site in remaining_sites:
                    candidate_indices = current_solution + [site]
                    metrics = _get_cached_metrics(tuple(sorted(candidate_indices)))
                    candidate_rows.append(
                        {
                            "site": site,
                            ranking: metrics[ranking],
                            "weighted_average": metrics["weighted_average"],
                            "total_cost": metrics["total_cost"],
                        }
                    )

                use_cost = bool(weights) and weights.get("cost", 0) > 0
                if use_cost:
                    # Cost weighting needs batch-relative normalization, which
                    # only pandas' vectorised operations give us cheaply, so
                    # only pay for the DataFrame here.
                    candidates_df, score_col = _apply_cost_weighting(
                        pd.DataFrame(candidate_rows),
                        ranking_col=ranking,
                        weights=weights,
                        scorer=scorer,
                    )
                    # _apply_cost_weighting only blends onto its
                    # lower-is-better "composite_score" scale when it has
                    # usable cost data to blend -- a positive cost weight
                    # alone isn't enough, it also no-ops (score_col ==
                    # ranking unchanged) when e.g. every candidate's
                    # total_cost is NaN. In that case put the raw ranking
                    # column onto the same scale via the scorer, so either
                    # way the RCL below is built on lower-is-better values.
                    if score_col == ranking:
                        candidates_df, score_col = _add_rank_score_column(
                            candidates_df, scorer
                        )
                    candidate_scores: list[tuple[float, float, int]] = list(
                        zip(
                            candidates_df[score_col],
                            candidates_df["weighted_average"],
                            candidates_df["site"],
                        )
                    )
                else:
                    candidate_scores: list[tuple[float, float, int]] = [
                        (
                            float(scorer.normalise(row[ranking])),
                            row["weighted_average"],
                            row["site"],
                        )
                        for row in candidate_rows
                    ]

                # Scores are lower-is-better by construction above, so this
                # is always a minimisation -- no direction branch needed.
                candidate_scores.sort(key=lambda x: (x[0], x[1]))

                f_best = candidate_scores[0][0]
                f_worst = candidate_scores[-1][0]
                value_range = abs(f_best - f_worst)

                if value_range < 1e-9:
                    # All candidates are tied; picking any of them is equally greedy.
                    rcl = [s for _, _, s in candidate_scores]
                else:
                    threshold = f_best + alpha * value_range
                    rcl = [s for score, _, s in candidate_scores if score <= threshold]

                if not rcl:
                    rcl = [candidate_scores[0][2]]

                chosen_site = attempt_rng.choice(rcl)
                current_solution.append(chosen_site)
                current_solution_set.add(chosen_site)

            if construction_failed:
                continue

            # ---------------------------------------------------------------
            # LOCAL SEARCH PHASE (1-opt swap)
            # [UPDATED] Shifted to First-Improvement for massive speed gains.
            # ---------------------------------------------------------------
            # 20% of the time, keep the raw GRASP construction to ensure pool diversity
            use_cost = bool(weights) and weights.get("cost", 0) > 0

            if rng.random() > (1 - local_search_chance):
                improved = True
                max_swaps = max_swap_count_local_search
                swaps = 0
                while improved and swaps < max_swaps:
                    swaps += 1
                    improved = False

                    current_metrics = _get_cached_metrics(
                        tuple(sorted(current_solution))
                    )
                    current_primary = float(scorer.normalise(current_metrics[ranking]))
                    current_secondary = current_metrics["weighted_average"]

                    outside_sites = [
                        s for s in all_site_indices if s not in current_solution_set
                    ]

                    if not use_cost:
                        # Lazy pairwise first-improvement scan (unchanged from
                        # before cost weighting was introduced).
                        for old_site in current_solution:
                            if old_site in required_site_set:
                                continue
                            for new_site in outside_sites:
                                candidate = [
                                    s for s in current_solution if s != old_site
                                ] + [new_site]

                                swap_metrics = _get_cached_metrics(
                                    tuple(sorted(candidate))
                                )
                                swap_primary = float(
                                    scorer.normalise(swap_metrics[ranking])
                                )
                                swap_secondary = swap_metrics["weighted_average"]

                                # Both sides are normalised, so lower is
                                # always better and one comparison covers
                                # every ranking direction.
                                if (swap_primary, swap_secondary) < (
                                    current_primary,
                                    current_secondary,
                                ):
                                    # First-Improvement: Apply immediately, break loops, restart neighborhood
                                    current_solution = candidate
                                    current_solution_set = set(current_solution)
                                    improved = True
                                    break

                            if improved:
                                break  # Break outer loop to restart the `while improved` check
                    else:
                        # Cost weighting is active: total_cost is a per-combination
                        # scalar, so it can only be normalized fairly against a
                        # batch of alternatives, not compared pairwise/lazily.
                        # Precompute the whole 1-opt neighborhood up front, blend
                        # cost in via the same batch-relative normalization used
                        # elsewhere, then scan for the first improving swap.
                        rows = [
                            {
                                "total_cost": current_metrics["total_cost"],
                                ranking: current_metrics[ranking],
                                "weighted_average": current_metrics[
                                    "weighted_average"
                                ],
                            }
                        ]
                        swap_candidates = []
                        for old_site in current_solution:
                            if old_site in required_site_set:
                                continue
                            for new_site in outside_sites:
                                candidate = [
                                    s for s in current_solution if s != old_site
                                ] + [new_site]
                                swap_metrics = _get_cached_metrics(
                                    tuple(sorted(candidate))
                                )
                                swap_candidates.append(candidate)
                                rows.append(
                                    {
                                        "total_cost": swap_metrics["total_cost"],
                                        ranking: swap_metrics[ranking],
                                        "weighted_average": swap_metrics[
                                            "weighted_average"
                                        ],
                                    }
                                )

                        if swap_candidates:
                            # `scorer` is consumed *inside*
                            # _apply_cost_weighting: it puts the raw ranking
                            # column onto a "badness" (0=best) scale before
                            # blending in cost, so the returned score_col
                            # ("composite_score") is lower-is-better whatever
                            # the ranking column's own direction -- BUT ONLY
                            # when it actually blends. It also no-ops
                            # (returns score_col == ranking_col unchanged)
                            # when it lacks usable cost data to blend (e.g.
                            # every candidate's total_cost is NaN), in which
                            # case score_col is back on the ranking column's
                            # natural scale and a blind "<" would pick the
                            # worst swap for a maximising metric. Normalising
                            # here in that case keeps the comparison below
                            # unconditional.
                            batch_df, score_col = _apply_cost_weighting(
                                pd.DataFrame(rows),
                                ranking_col=ranking,
                                weights=weights,
                                scorer=scorer,
                            )
                            if score_col == ranking:
                                batch_df, score_col = _add_rank_score_column(
                                    batch_df, scorer
                                )
                            current_score = batch_df.iloc[0][score_col]
                            current_secondary_score = batch_df.iloc[0][
                                "weighted_average"
                            ]

                            # rows[0] / batch_df.iloc[0] is the current
                            # solution's own metrics (appended first, above);
                            # rows[1:] are the swap candidates in the same
                            # order they were appended to swap_candidates.
                            # reset_index(drop=True) re-bases that slice to
                            # 0..n-1 so the loop index `i` lines up exactly
                            # with swap_candidates[i] -- without it, `i`
                            # would instead be the original batch_df index
                            # (1..n), which would misalign by one and pick
                            # the wrong candidate solution.
                            for i, row in batch_df.iloc[1:].reset_index(
                                drop=True
                            ).iterrows():
                                candidate_score = (
                                    row[score_col],
                                    row["weighted_average"],
                                )
                                current = (current_score, current_secondary_score)
                                # score_col is lower-is-better either way
                                # (blended composite, or normalised raw).
                                if candidate_score < current:
                                    current_solution = swap_candidates[i]
                                    current_solution_set = set(current_solution)
                                    improved = True
                                    break

            # ---------------------------------------------------------------
            # FEASIBILITY CHECK (hybrid objectives' max-value cutoff)
            # ---------------------------------------------------------------
            # Judged on the finished (post-local-search) solution: with
            # fewer than p sites mid-construction, the worst-case travel is
            # usually still shrinking, so only the final form is checked.
            # A rejected solution costs an attempt, like a diversity reject.
            if max_value_cutoff is not None:
                candidate_metrics = _get_cached_metrics(
                    tuple(sorted(current_solution))
                )
                if candidate_metrics["max_for_ranking"] > max_value_cutoff:
                    continue

            # ---------------------------------------------------------------
            # DIVERSITY CHECK
            # ---------------------------------------------------------------
            if _too_similar_to_accepted(
                current_solution_set, accepted_solution_sets, min_jaccard_distance
            ):
                continue

            current_solution.sort()

            # Accept the solution.
            #
            # `unreachable_cost` must be passed here, not just to the cached
            # in-loop evaluations above: without it the returned rows'
            # `*_for_ranking` columns fall back to the reachable-only
            # figures, so GRASP would *search* on substituted values but
            # *return* unsubstituted ones -- and solve()'s final sort ranks
            # the returned pool on exactly those columns. Brute-force and
            # greedy have always passed it; this call was the odd one out.
            final_metrics = self.evaluate_single_solution_single_objective(
                site_indices=current_solution,
                objective=objectives,
                threshold_for_coverage=threshold_for_coverage,  # Applied only at the end
                weights=weights,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

            accepted_solution_sets.append(current_solution_set)
            final_solutions_metrics.append(final_metrics)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        if len(final_solutions_metrics) < num_solutions:
            from warnings import warn

            warn(
                f"GRASP exhausted attempt budget ({max_attempts} attempts) before finding "
                f"{num_solutions} sufficiently diverse solutions. "
                f"Returning {len(final_solutions_metrics)} solutions.",
                UserWarning,
                stacklevel=2,
            )

        return final_solutions_metrics
