from lokigi.utils import (
    _generate_all_combinations,
    _get_ranking_by_objective,
    _get_required_site_indices,
    _too_similar_to_accepted,
    _apply_cost_weighting,
)

from lokigi.site_solutions import SiteSolutionSet

# Data manipulation imports
import pandas as pd
import random
import math
import itertools

# Other imports
from warnings import warn
import heapq
from tqdm.auto import tqdm
from functools import lru_cache

# Warn if brute force will be slow
BRUTE_FORCE_WARN_THRESHOLD = 75_000
BRUTE_FORCE_LIMIT = 500_000


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
        rank_best_n_on="weighted_average",
        max_value_cutoff=None,
        threshold_for_coverage=None,
    ):

        if brute_force_keep_best_n is not None:
            top_n_heap = []  # To store the smallest scores (best)
            # print(f"Keeping top {brute_force_keep_best_n}")
        if brute_force_keep_worst_n is not None:
            bottom_n_heap = []  # To store the largest scores (worst)
            # print(f"Keeping worst {brute_force_keep_worst_n}")

        # A unique, monotonically increasing tie-breaker sits between the
        # score and the metrics dict in every heap entry, so that exact
        # score ties never fall through to comparing two dicts (which
        # aren't orderable and would raise TypeError).
        tie_breaker = itertools.count()

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

        # mclp's ranking column (coverage proportion) is higher-is-better,
        # while every other objective's ranking column is lower-is-better.
        higher_is_better = objectives == "mclp"

        if show_progress:
            possible_combinations = tqdm(possible_combinations)

        for possible_solution in possible_combinations:
            if brute_force_keep_best_n is None and brute_force_keep_worst_n is None:
                # Keep all results
                single_solution_metrics = (
                    self.evaluate_single_solution_single_objective(
                        site_indices=possible_solution,
                        objective=objectives,
                        threshold_for_coverage=threshold_for_coverage,
                        weights=weights,
                    ).return_solution_metrics()
                )

                if max_value_cutoff is None or (
                    max_value_cutoff is not None
                    and single_solution_metrics["max"] <= max_value_cutoff
                ):
                    outputs.append(single_solution_metrics)

            # --- Logic for Top N (Smallest Scores) ---
            # We store -score to simulate a Max-Heap using heapq
            else:
                metrics = self.evaluate_single_solution_single_objective(
                    site_indices=possible_solution,
                    objective=objectives,
                    threshold_for_coverage=threshold_for_coverage,
                    weights=weights,
                ).return_solution_metrics()

                raw_score = metrics[rank_best_n_on]
                # Normalise so lower always means better: the heaps below
                # keep the N smallest scores as "best" and the N largest as
                # "worst", which only holds for minimising objectives.
                # Negating mclp's coverage proportion lets the same heap
                # logic serve both directions -- without this, keep_best_n
                # for mclp retained the WORST-coverage combinations.
                score = -raw_score if higher_is_better else raw_score
                max_value = metrics["max"]

                if max_value_cutoff is None or (
                    max_value_cutoff is not None and max_value <= max_value_cutoff
                ):
                    if brute_force_keep_best_n is not None:
                        if len(top_n_heap) < brute_force_keep_best_n:
                            heapq.heappush(top_n_heap, (-score, next(tie_breaker), metrics))
                        elif -score > top_n_heap[0][0]:
                            heapq.heapreplace(
                                top_n_heap, (-score, next(tie_breaker), metrics)
                            )

                    # --- Logic for Bottom N (Largest Scores) ---
                    # Standard Min-Heap to keep the largest values
                    if brute_force_keep_worst_n is not None:
                        if len(bottom_n_heap) < brute_force_keep_worst_n:
                            heapq.heappush(bottom_n_heap, (score, next(tie_breaker), metrics))
                        elif score > bottom_n_heap[0][0]:
                            heapq.heapreplace(
                                bottom_n_heap, (score, next(tie_breaker), metrics)
                            )

        if brute_force_keep_best_n is None and brute_force_keep_worst_n is None:
            return outputs
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
        show_progress: bool = False,
        threshold_for_coverage=None,
        max_value_cutoff=None,
    ):
        ranking = _get_ranking_by_objective(objective=objectives)

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
                    ).return_solution_metrics()
                )

            # mclp's ranking column (coverage proportion) is higher-is-better,
            # while every other objective's ranking column is lower-is-better
            outputs_df = pd.DataFrame(outputs)
            higher_is_better = objectives == "mclp"

            # The max-value cutoff (hybrid objectives) is a constraint on
            # the FINAL solution only: with fewer than p sites the
            # worst-case travel is usually still shrinking, so filtering
            # intermediate steps would wrongly rule everything out. At the
            # final step, keep only combinations meeting the cutoff so the
            # guarantee the hybrid objectives promise actually holds for
            # whatever is returned.
            if max_value_cutoff is not None and i == p:
                outputs_df = outputs_df[outputs_df["max"] <= max_value_cutoff]
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

            if weights and weights.get("cost", 0) > 0:
                outputs_df, score_col = _apply_cost_weighting(
                    outputs_df,
                    ranking_col=ranking,
                    weights=weights,
                    higher_is_better=higher_is_better,
                )
                sort_ascending = [True, True]
            else:
                score_col = ranking
                sort_ascending = [not higher_is_better, True]

            evaluated_solutions = outputs_df.sort_values(
                [score_col, "weighted_average"], ascending=sort_ascending
            )

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
        ).return_solution_metrics()

        return [best_solution_metrics]


class GraspMixin:
    def _grasp(
        self,
        p: int,
        objectives,
        weights,
        num_solutions: int = 5,
        show_progress: bool = False,
        threshold_for_coverage=None,
        alpha: float = 0.2,
        random_seed: int = 42,
        max_attempts: int | str = "default",
        min_sites_different: int = 1,
        is_minimization: bool = True,  # Flag for sort order & thresholding
        local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        max_swap_count_local_search=10,
        max_value_cutoff=None,
    ):
        """
        GRASP (Greedy Randomised Adaptive Search Procedure) for finding multiple
        near-optimal facility location solutions.
        """
        rng = random.Random(random_seed)
        ranking = _get_ranking_by_objective(objective=objectives)
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

        min_jaccard_distance = float(min_sites_different) / float(p)

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
            ).return_solution_metrics()

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
                        higher_is_better=not is_minimization,
                    )
                    scores_minimized = True
                    candidate_scores: list[tuple[float, float, int]] = list(
                        zip(
                            candidates_df[score_col],
                            candidates_df["weighted_average"],
                            candidates_df["site"],
                        )
                    )
                else:
                    scores_minimized = is_minimization
                    candidate_scores: list[tuple[float, float, int]] = [
                        (row[ranking], row["weighted_average"], row["site"])
                        for row in candidate_rows
                    ]

                # [UPDATED] Sort and construct RCL based on minimization vs maximization
                candidate_scores.sort(
                    key=lambda x: (x[0], x[1]), reverse=not scores_minimized
                )

                f_best = candidate_scores[0][0]
                f_worst = candidate_scores[-1][0]
                value_range = abs(f_best - f_worst)

                if value_range < 1e-9:
                    # All candidates are tied; picking any of them is equally greedy.
                    rcl = [s for _, _, s in candidate_scores]
                else:
                    if scores_minimized:
                        threshold = f_best + alpha * value_range
                        rcl = [
                            s for score, _, s in candidate_scores if score <= threshold
                        ]
                    else:
                        threshold = f_best - alpha * value_range
                        rcl = [
                            s for score, _, s in candidate_scores if score >= threshold
                        ]

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
                    current_primary = current_metrics[ranking]
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
                                swap_primary = swap_metrics[ranking]
                                swap_secondary = swap_metrics["weighted_average"]

                                if is_minimization:
                                    is_better = (swap_primary, swap_secondary) < (
                                        current_primary,
                                        current_secondary,
                                    )
                                else:
                                    is_better = (swap_primary, swap_secondary) > (
                                        current_primary,
                                        current_secondary,
                                    )

                                if is_better:
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
                            # `higher_is_better` is consumed *inside*
                            # _apply_cost_weighting: it inverts the raw
                            # ranking column into "badness" (0=best) before
                            # blending in cost, and the returned score_col
                            # ("composite_score") is unconditionally on that
                            # same lower-is-better, 0-is-best scale --
                            # regardless of whether the underlying objective
                            # is minimized (e.g. weighted_average) or
                            # maximized (e.g. mclp's coverage proportion).
                            #
                            # That's why the comparison below is a plain "<"
                            # with no is_minimization/higher_is_better branch,
                            # unlike the raw-metric comparison in the
                            # non-cost branch above. Do NOT wrap it in an
                            # `if is_minimization: ... else: ...` to mirror
                            # that branch -- the direction has already been
                            # normalized once by _apply_cost_weighting, and
                            # inverting it a second time here would make this
                            # code pick the worst swap instead of the best
                            # one for every maximizing objective (mclp).
                            batch_df, score_col = _apply_cost_weighting(
                                pd.DataFrame(rows),
                                ranking_col=ranking,
                                weights=weights,
                                higher_is_better=not is_minimization,
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
                                if (row[score_col], row["weighted_average"]) < (
                                    current_score,
                                    current_secondary_score,
                                ):
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
                if candidate_metrics["max"] > max_value_cutoff:
                    continue

            # ---------------------------------------------------------------
            # DIVERSITY CHECK
            # ---------------------------------------------------------------
            if _too_similar_to_accepted(
                current_solution_set, accepted_solution_sets, min_jaccard_distance
            ):
                continue

            current_solution.sort()

            # Accept the solution
            final_metrics = self.evaluate_single_solution_single_objective(
                site_indices=current_solution,
                objective=objectives,
                threshold_for_coverage=threshold_for_coverage,  # Applied only at the end
                weights=weights,
            ).return_solution_metrics()

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
