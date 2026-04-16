from lokigi.utils import (
    _generate_all_combinations,
    _get_ranking_by_objective,
    _too_similar_to_accepted,
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

# Warn if brute force will be slow
BRUTE_FORCE_WARN_THRESHOLD = 75_000
BRUTE_FORCE_LIMIT = 500_000


class BruteForceMixin:
    def _brute_force(
        self,
        p: int,
        objectives,
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
                    site_indices=possible_solution, objective=objectives
                ).return_solution_metrics()

                score = metrics[rank_best_n_on]
                max_value = metrics["max"]

                if max_value_cutoff is None or (
                    max_value_cutoff is not None and max_value <= max_value_cutoff
                ):
                    if brute_force_keep_best_n is not None:
                        if len(top_n_heap) < brute_force_keep_best_n:
                            heapq.heappush(top_n_heap, (-score, metrics))
                        elif -score > top_n_heap[0][0]:
                            heapq.heapreplace(top_n_heap, (-score, metrics))

                    # --- Logic for Bottom N (Largest Scores) ---
                    # Standard Min-Heap to keep the largest values
                    if brute_force_keep_worst_n is not None:
                        if len(bottom_n_heap) < brute_force_keep_worst_n:
                            heapq.heappush(bottom_n_heap, (score, metrics))
                        elif score > bottom_n_heap[0][0]:
                            heapq.heapreplace(bottom_n_heap, (score, metrics))

        if brute_force_keep_best_n is None and brute_force_keep_worst_n is None:
            return outputs
        else:
            # Reconstruct the 'outputs' list
            # Extract dictionaries from heaps and sort them
            if brute_force_keep_best_n is not None:
                best_list = [item[1] for item in sorted(top_n_heap, key=lambda x: x[0])]

            if brute_force_keep_worst_n is not None:
                worst_list = [
                    item[1] for item in sorted(bottom_n_heap, key=lambda x: x[0])
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
        show_progress: bool = False,
        threshold_for_coverage=None,
    ):
        ranking = _get_ranking_by_objective(objective=objectives)

        # Loop through
        best_indices = []

        loop_iterations = range(1, p + 1)
        if show_progress:
            loop_iterations = tqdm(loop_iterations)

        for i in loop_iterations:
            # print(f"Loop {i}")
            possible_combinations = _generate_all_combinations(
                n_facilities=self.total_n_sites,
                p=i,
                site_problem=self,
                force_include_indices=None if i == 1 else list(best_indices),
            )

            # print(f"Possible combinations: {possible_combinations}")

            outputs = []

            for possible_solution in possible_combinations:
                # print(f"Evaluating possible solution: {possible_solution}")
                outputs.append(
                    self.evaluate_single_solution_single_objective(
                        site_indices=possible_solution,
                        objective=objectives,
                    ).return_solution_metrics()
                )

            evaluated_solutions = pd.DataFrame(outputs).sort_values(
                [ranking, "weighted_average"]
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
                print(f"Best combination for {i} sites: {best_indices}")

        best_solution_metrics = self.evaluate_single_solution_single_objective(
            site_indices=best_indices,
            objective=objectives,
            threshold_for_coverage=threshold_for_coverage,
        ).return_solution_metrics()

        return [best_solution_metrics]


class GraspMixin:
    def _grasp(
        self,
        p: int,
        objectives,
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
    ):
        """
        GRASP (Greedy Randomised Adaptive Search Procedure) for finding multiple
        near-optimal facility location solutions.
        """
        rng = random.Random(random_seed)
        ranking = _get_ranking_by_objective(objective=objectives)
        all_site_indices = list(range(self.total_n_sites))

        min_jaccard_distance = float(min_sites_different) / float(p)

        total_combinations = math.comb(self.total_n_sites, p)
        if max_attempts == "default":
            max_attempts = min(num_solutions * 20, total_combinations)

        final_solutions_metrics = []
        accepted_solution_sets: list[set] = []
        attempts = 0

        # -------------------------------------------------------------------
        # [NEW] CACHING: Memoize evaluations to prevent redundant compute.
        # Uses a tuple of sorted indices as a canonical, hashable key.
        # -------------------------------------------------------------------
        evaluation_cache = {}

        # def _get_cached_metrics(indices: list[int]):
        #     sig = tuple(sorted(indices))
        #     if sig not in evaluation_cache:
        #         evaluation_cache[sig] = self.evaluate_single_solution_single_objective(
        #             site_indices=list(indices),
        #             objective=objectives,
        #         ).return_solution_metrics()
        #     return evaluation_cache[sig]

        @lru_cache(maxsize=10000)
        def _get_cached_metrics(indices_tuple: tuple):
            return self.evaluate_single_solution_single_objective(
                site_indices=list(indices_tuple),
                objective=objectives,
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
            current_solution: list[int] = []
            current_solution_set: set[int] = set()
            construction_failed = False

            for step in range(p):
                remaining_sites = [
                    s for s in all_site_indices if s not in current_solution_set
                ]

                if not remaining_sites:
                    construction_failed = True
                    break

                candidate_scores: list[tuple[float, float, int]] = []
                for site in remaining_sites:
                    candidate_indices = current_solution + [site]
                    metrics = _get_cached_metrics(tuple(sorted(candidate_indices)))

                    primary_score = metrics[ranking]
                    secondary_score = metrics["weighted_average"]
                    candidate_scores.append((primary_score, secondary_score, site))

                # [UPDATED] Sort and construct RCL based on minimization vs maximization
                candidate_scores.sort(
                    key=lambda x: (x[0], x[1]), reverse=not is_minimization
                )

                f_best = candidate_scores[0][0]
                f_worst = candidate_scores[-1][0]
                value_range = abs(f_best - f_worst)

                if value_range < 1e-9:
                    # All candidates are tied; picking any of them is equally greedy.
                    rcl = [s for _, _, s in candidate_scores]
                else:
                    if is_minimization:
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

                    for old_site in current_solution:
                        for new_site in outside_sites:
                            candidate = [
                                s for s in current_solution if s != old_site
                            ] + [new_site]

                            swap_metrics = _get_cached_metrics(tuple(sorted(candidate)))
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
