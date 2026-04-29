import pandas as pd
from lokigi.utils import _add_rank_column


class SolutionComparatorMethodsMixin:
    def compare_top_results(self, n=1):
        """
        Compares the top N solutions from both sets.
        """
        top_a = self.set_a.solution_df.head(n).copy()
        top_b = self.set_b.solution_df.head(n).copy()

        top_a["origin"] = self.labels[0]
        top_b["origin"] = self.labels[1]

        return pd.concat([top_a, top_b]).reset_index(drop=True)

    def get_metric_summary(self, objective="weighted_average"):
        """
        Returns a comparison of descriptive statistics for a specific objective.
        """

        stats_a = self.set_a.solution_df[objective].describe()
        stats_b = self.set_b.solution_df[objective].describe()

        summary = pd.DataFrame({self.labels[0]: stats_a, self.labels[1]: stats_b})
        summary["difference"] = summary[self.labels[0]] - summary[self.labels[1]]
        return summary

    def site_overlap(self, top_n=1):
        """
        Analyzes how many sites are common between the top N solutions
        of both sets.
        """

        def get_all_sites(solution_df, n):
            # Flattens the list of site_indices from the top N rows
            all_indices = solution_df.head(n)["site_indices"].explode()
            return set(all_indices)

        sites_a = get_all_sites(self.set_a.solution_df, top_n)
        sites_b = get_all_sites(self.set_b.solution_df, top_n)

        common = sites_a.intersection(sites_b)
        only_a = sites_a - sites_b
        only_b = sites_b - sites_a

        return {
            "common_sites_count": len(common),
            "common_sites_indices": list(common),
            "unique_to_a": list(only_a),
            "unique_to_b": list(only_b),
            "jaccard_similarity": len(common) / len(sites_a.union(sites_b))
            if sites_a.union(sites_b)
            else 0,
        }

    def find_balanced_solution(
        self,
        top_n=25,
        method="rank_balanced",
        objective="weighted_average",
        secondary_objective="max",
        rank_weight=0.5,  # Controls how harshly to penalise high (poor) ranks
        return_details=False,
    ):
        """
        Finds the most similar top solutions from each set to approximate
        a balanced compromise without deep optimization.

        Parameters:
        -----------
        top_n : int
            Number of top solutions from each set to consider
        method : str
            Similarity metric to use:
            - 'jaccard': Jaccard similarity of site sets
            - 'overlap': Raw count of overlapping sites
            - 'combined': Weighted combination of Jaccard and normalized objectives
            - 'rank_balanced': Jaccard similarity penalised by the sum of their ranks (prioritises shared balance)
        rank_weight : float
            Only used if method='rank_balanced'. Determines the strength of the rank penalty.
        return_details : bool
            If True, returns full comparison details; if False, returns just the best pair

        Returns:
        --------
        dict or tuple
            If return_details=True: dict with best pair info and all comparisons
            If return_details=False: tuple of (solution_from_a, solution_from_b)
        """

        ascending_primary = (
            False if objective == "proportion_within_coverage_threshold" else True
        )
        ascending_secondary = (
            False
            if secondary_objective == "proportion_within_coverage_threshold"
            else True
        )

        sols_a_copy = self.set_a.solution_df.copy()
        sols_a_copy = _add_rank_column(
            sols_a_copy,
            score_col=objective,
            tiebreaker_col=secondary_objective,
            ascending=[ascending_primary, ascending_secondary],
        )
        sols_b_copy = self.set_b.solution_df.copy()
        sols_b_copy = _add_rank_column(
            sols_b_copy,
            score_col=objective,
            tiebreaker_col=secondary_objective,
            ascending=[ascending_primary, ascending_secondary],
        )

        # Get top N from each set
        top_a = sols_a_copy.head(top_n)
        top_b = sols_b_copy.head(top_n)

        # Get the objective column name
        obj = objective

        # Changed to negative infinity to allow for negative scores in rank_balanced
        best_similarity = -float("inf")
        best_pair = None
        all_comparisons = []

        # Use enumerate (start=1) to capture the rank of the solution in the top_n slice
        for rank_a, (idx_a, row_a) in enumerate(top_a.iterrows(), start=1):
            sites_a = set(row_a["site_indices"])

            for rank_b, (idx_b, row_b) in enumerate(top_b.iterrows(), start=1):
                sites_b = set(row_b["site_indices"])

                # Calculate similarity metrics
                intersection = sites_a.intersection(sites_b)
                union = sites_a.union(sites_b)

                jaccard = len(intersection) / len(union) if union else 0
                overlap_count = len(intersection)

                if method == "combined":
                    # Normalize objectives to [0, 1] range within the top_n
                    obj_range_a = top_a[obj].max() - top_a[obj].min()
                    obj_range_b = top_b[obj].max() - top_b[obj].min()

                    norm_a = (
                        (row_a[obj] - top_a[obj].min()) / obj_range_a
                        if obj_range_a > 0
                        else 1
                    )
                    norm_b = (
                        (row_b[obj] - top_b[obj].min()) / obj_range_b
                        if obj_range_b > 0
                        else 1
                    )

                    # Penalize large differences in objective quality
                    obj_similarity = 1 - abs(norm_a - norm_b)

                    # Combined score: 70% site overlap, 30% objective similarity
                    similarity = 0.7 * jaccard + 0.3 * obj_similarity

                elif method == "rank_balanced":
                    # Calculate rank penalty based on sum of ranks.
                    # Normalised so the maximum penalty is 1.0 (when both are ranked at top_n)
                    rank_penalty = (rank_a + rank_b) / (2 * top_n)

                    # Score is Jaccard overlap minus the weighted penalty.
                    # This ensures identical sets (Jaccard=1) are differentiated purely by rank sum.
                    similarity = jaccard - (rank_weight * rank_penalty)

                else:
                    similarity = jaccard if method == "jaccard" else overlap_count

                comparison = {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "rank_a": rank_a,  # Track rank A
                    "rank_b": rank_b,  # Track rank B
                    "similarity": similarity,
                    "jaccard": jaccard,
                    "overlap_count": overlap_count,
                    "obj_value_a": row_a[obj],
                    "obj_value_b": row_b[obj],
                    "sites_a": sites_a,
                    "sites_b": sites_b,
                    "common_sites": intersection,
                }

                all_comparisons.append(comparison)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (row_a, row_b, comparison)

        if best_pair is None:
            return None

        solution_a, solution_b, best_comparison = best_pair

        if return_details:
            # Sort all comparisons by similarity
            all_comparisons.sort(key=lambda x: x["similarity"], reverse=True)

            return {
                "best_from_a": solution_a.to_dict(),
                "best_from_b": solution_b.to_dict(),
                "best_rank_a": best_comparison[
                    "rank_a"
                ],  # Include winning ranks in output
                "best_rank_b": best_comparison["rank_b"],
                "similarity_score": best_similarity,
                "jaccard": best_comparison["jaccard"],
                "overlap_count": best_comparison["overlap_count"],
                "common_sites": list(best_comparison["common_sites"]),
                "unique_to_a": list(
                    best_comparison["sites_a"] - best_comparison["sites_b"]
                ),
                "unique_to_b": list(
                    best_comparison["sites_b"] - best_comparison["sites_a"]
                ),
                "top_5_matches": all_comparisons[:5],
                "method_used": method,
            }
        else:
            return (solution_a, solution_b)
