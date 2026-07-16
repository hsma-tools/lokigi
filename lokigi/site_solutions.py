import numpy as np
import pandas as pd

from lokigi.utils import _min_max_normalize

from lokigi.mixins.site_solution_plots import (
    MapsMixin,
    NonMapPlotsMixin,
    DistributionPlotsMixin,
    EquityPlotsMixin,
)

from lokigi.mixins.site_solution_pareto import ParetoMixin

import copy

from lokigi.mixins.solution_comparator_plots import SolutionComparatorPlotsMixin
from lokigi.mixins.solution_comparator_methods import SolutionComparatorMethodsMixin
from lokigi.mixins.site_eda import (
    SiteSolutionHotspotCalculationMixin,
    HotspotPlotMixin,
    SiteProblemEDAMixin,
)


# MARK: CLASS EvaluatedCombination
class EvaluatedCombination:
    """
    Container for results and summary metrics of an evaluated site combination.

    This class stores the outcome of evaluating a candidate solution (i.e.,
    a set of selected sites) against a demand dataset, and computes a range
    of summary statistics based on the minimum cost (e.g., travel time or
    distance) from demand locations to the selected sites.

    Parameters
    ----------
    solution_type : str
        Label describing the type of solution (e.g., optimisation method or scenario).
    site_names : list of str
        Names of the selected sites in the solution.
    site_indices : list of int
        Indices of the selected sites corresponding to the original site list.
    evaluated_combination_df : pandas.DataFrame
        DataFrame containing evaluation results for each demand point. Must include:
        - "min_cost": Minimum cost from each demand point to the selected sites.
        - "within_threshold": Boolean indicator of whether the demand point is
          within the specified coverage threshold.
        - A demand column specified by ``site_problem._demand_data_demand_col``.
    site_problem : object
        Object containing problem configuration and metadata, including the name
        of the demand column via ``_demand_data_demand_col``.
    coverage_threshold : float, optional
        Threshold used to determine whether a demand point is considered covered.
        If provided, used to compute the proportion of demand points within coverage.

    Attributes
    ----------
    solution_type : str
        Type or label of the solution.
    site_names : list of str
        Names of the selected sites.
    site_indices : list of int
        Indices of the selected sites.
    evaluated_combination_df : pandas.DataFrame
        DataFrame containing per-demand-point evaluation results.
    site_problem : object
        Problem definition object.
    coverage_threshold : float or None
        Coverage threshold used in evaluation.

    weighted_average : float
        Demand-weighted average of the minimum cost.
    unweighted_average : float
        Simple (unweighted) average of the minimum cost.
    percentile_90th : float
        90th percentile of the minimum cost distribution.
    max : float
        Maximum minimum cost across all demand points.
    total_cost : float
        Total fixed cost of the selected sites (sum of the `cost_col`
        values configured via `add_sites()`). `NaN` if no `cost_col` was
        configured. Always calculated; only influences which solution is
        selected if explicitly passed as a weight (`weights={"cost": ...}`).
    proportion_within_coverage_threshold : float
        Proportion of demand points that fall within the coverage threshold.



    Notes
    -----
    The weighted average is computed using demand values as weights.
    """

    def __init__(
        self,
        solution_type,
        site_names,
        site_indices,
        evaluated_combination_df,
        weights,
        site_problem,
        coverage_threshold=None,
    ):
        self.solution_type = solution_type
        self.site_names = site_names
        self.site_indices = site_indices
        self.evaluated_combination_df = evaluated_combination_df
        self.site_problem = site_problem

        self.weighted_by_equity_group = {}
        self.unweighted_by_equity_group = {}
        self.gap_absolute_weighted = None
        self.gap_absolute_desc = "N/A (No equity data)"

        self.gap_relative_weighted = None
        self.gap_relative_desc = "N/A (No equity data)"

        self.inter_tertile_ratio = None
        self.inter_tertile_desc = "N/A (No equity data)"

        self.coverage_by_equity_group = {}
        self.max_cost_by_equity_group = {}
        self.avg_lower_third_bins = None
        self.avg_middle_third_bins = None
        self.avg_upper_third_bins = None

        # Weighted average code modified from
        # https://github.com/health-data-science-OR/healthcare-logistics/blob/8d03b890a8ce861b64f6f834710dc50f2d85f68e/optimisation/metapy/evolutionary/evolutionary.py#L722
        # Credit for original code to Dr Tom Monks
        # Licence reproduced below in line with reuse conditions
        # MIT License
        #
        # Copyright (c) 2020 health-data-science-OR
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.

        # "cost" is a combination-level weight (total cost of the selected
        # sites), not a per-demand-point value like demand/equity/additional
        # data, so it cannot be blended via the row-level mechanism below
        # (a constant column would normalize to all-ones and have no
        # effect). It is excluded here and handled separately by the
        # solver's combination-comparison logic (see _apply_cost_weighting).
        row_level_weights = (
            None
            if weights is None
            else {k: v for k, v in weights.items() if k.lower() != "cost"}
        )

        # If weights is purely the demand data, as was the default behaviour prior to 0.3.0,
        # then calculate the weighted average metric in the existing way
        if row_level_weights is None or (
            isinstance(row_level_weights, dict)
            and (
                len(row_level_weights) == 0
                or (len(row_level_weights) == 1 and "demand" in row_level_weights)
            )
        ):
            active_weights = self.evaluated_combination_df[
                self.site_problem._demand_data_demand_col
            ]

            # print("Falling back to default: weighting p-median by demand only")
            # print(f"Weights: {weights}")
            self.weighted_average = np.average(
                self.evaluated_combination_df["min_cost"],
                weights=active_weights,
            )
        else:
            # print("Weighting p-median by custom options")
            # print(f"Weights: {weights}")

            # Initialize an array of zeros to build our blended row-level weights
            compound_weights = np.zeros(len(self.evaluated_combination_df))
            self.extra_metrics = {}

            for label, weight in row_level_weights.items():
                # Map the user's label to the correct DataFrame column name
                if label == "demand":
                    col_name = self.site_problem._demand_data_demand_col
                elif label == "equity":
                    col_name = self.site_problem._equity_data_equity_col
                else:
                    col_name = label

                if col_name in self.evaluated_combination_df.columns:
                    # Extract raw data
                    column_data = self.evaluated_combination_df[col_name].astype(float)

                    # if demand, assume higher_better
                    # if equity, get direction from equity data
                    direction = None

                    if label.lower() == "demand":
                        direction = (
                            "higher_better"  # 'demand' defaults to higher_better
                        )

                    elif label.lower() == "equity":
                        # Equity weighting exists to prioritise worse-off
                        # regions, so whichever end of the equity scale is
                        # disadvantaged is the end that gets the weight.
                        direction = (
                            "lower_better"
                            if self.site_problem._equity_data_disadvantaged_end
                            == "low"
                            else "higher_better"
                        )

                    # Look up directionality from the problem configuration
                    elif self.site_problem.additional_data is not None:
                        # Find the metadata dict matching this label
                        meta = next(
                            (
                                item
                                for item in self.site_problem.additional_data
                                if item["label"] == label
                            ),
                            None,
                        )
                        if meta:
                            direction = meta.get("direction", "higher_better")

                    if direction is None:
                        raise ValueError(
                            f"Weight key '{label}' does not correspond to demand, "
                            "equity, or any registered additional dataset. Register "
                            "it via add_additional_data() first, or remove it from "
                            "the weights dict."
                        )

                    # Handle directionality by negating BEFORE normalising:
                    # for lower_better this maps the smallest raw value to
                    # 1.0 and the largest to 0.0 (identical to inverting
                    # afterwards), but keeps the identical-values edge case
                    # on the equal (full) baseline weight from constant_fill.
                    # Inverting after normalising would turn that neutral
                    # 1.0 into an all-zero weight vector and crash
                    # np.average with "Weights sum to zero".
                    if direction == "lower_better":
                        column_data = -column_data

                    # Min-Max Normalization to a 0.0 - 1.0 scale. Edge case:
                    # if all values are identical, give them equal (full)
                    # baseline weight rather than 0.
                    norm_data = _min_max_normalize(column_data, constant_fill=1.0)

                    # Accumulate the normalized, directional weight
                    compound_weights += norm_data * weight

            # Calculate the final composite weighted average across all objectives
            self.weighted_average = np.average(
                self.evaluated_combination_df["min_cost"], weights=compound_weights
            )
            active_weights = pd.Series(
                compound_weights, index=self.evaluated_combination_df.index
            )

        # Calculate the unweighted travel/cost statistics
        self.unweighted_average = np.average(self.evaluated_combination_df["min_cost"])

        self.percentile_90th = np.percentile(
            self.evaluated_combination_df["min_cost"], q=90
        )

        self.max = np.max(self.evaluated_combination_df["min_cost"])

        # Total fixed cost of the selected sites (e.g. build/operating cost).
        # Always calculated when a cost column has been configured via
        # add_sites(cost_col=...), regardless of whether "cost" is used as a
        # weight -- it is purely a reporting metric unless explicitly opted
        # into via weights={"cost": ...}.
        cost_col = getattr(self.site_problem, "_candidate_sites_cost_col", None)
        if cost_col is None or self.site_problem.candidate_sites is None:
            self.total_cost = np.nan
        else:
            cost_lookup = self.site_problem.candidate_sites.set_index(
                "canonical_site_index"
            )[cost_col]
            # skipna=False: if any selected site has a missing (NaN) cost,
            # the combination's total cost is genuinely unknown, not zero.
            # pandas' default skipna=True would otherwise silently treat a
            # missing cost as $0, making that site look free.
            self.total_cost = cost_lookup.loc[sorted(set(self.site_indices))].sum(
                skipna=False
            )

        self.coverage_threshold = coverage_threshold

        self.proportion_within_coverage_threshold = np.sum(
            self.evaluated_combination_df["within_threshold"]
        ) / len(self.evaluated_combination_df)

        # Calculate the weighted and unweighted cost per equity band
        # if equity data present
        self.weighted_by_equity_group = {}
        self.unweighted_by_equity_group = {}

        equity_col = getattr(self.site_problem, "_equity_data_equity_col", None)

        if equity_col and equity_col in self.evaluated_combination_df.columns:
            grouped_df = self.evaluated_combination_df.groupby(equity_col)

            # 1. Unweighted average by equity group
            self.unweighted_by_equity_group = (
                grouped_df["min_cost"].mean().round(2).to_dict()
            )

            # 2. Weighted average by equity group (matching global composite weights logic)
            for band, group in grouped_df:
                # Extract matching row weights for this specific group
                group_weights = active_weights.loc[group.index]

                # Avoid ZeroDivisionError if the combined weight for this band is 0
                if group_weights.sum() > 0:
                    self.weighted_by_equity_group[band] = np.average(
                        group["min_cost"], weights=group_weights
                    ).round(2)
                else:
                    self.weighted_by_equity_group[band] = group["min_cost"].mean()

            # 3. Disparity Metrics & Verbal Descriptors
            if self.weighted_by_equity_group:
                weighted_vals = list(self.weighted_by_equity_group.values())
                min_cost = min(weighted_vals)
                max_cost = max(weighted_vals)

                self.gap_absolute_weighted = max_cost - min_cost
                self.gap_absolute_desc = f"Spread of {self.gap_absolute_weighted:.1f} units between best and worst groups"

                if min_cost > 0:
                    self.gap_relative_weighted = max_cost / min_cost

                    # Generate Relative Gap Descriptor
                    if self.gap_relative_weighted <= 1.005:
                        self.gap_relative_desc = "Perfect Parity"
                    elif self.gap_relative_weighted <= 1.10:
                        self.gap_relative_desc = (
                            "Minimal Disparity (Worst group travels <10% longer)"
                        )
                    elif self.gap_relative_weighted <= 1.30:
                        self.gap_relative_desc = (
                            "Moderate Disparity (Worst group travels 10-30% longer)"
                        )
                    else:
                        pct_longer = (self.gap_relative_weighted - 1.0) * 100
                        self.gap_relative_desc = f"Significant Disparity (Worst group travels {pct_longer:.0f}% longer)"
                else:
                    self.gap_relative_weighted = np.nan
                    self.gap_relative_desc = "N/A (Zero baseline cost)"

            # 4. Coverage Equity (Thresholds by Group)
            if "within_threshold" in self.evaluated_combination_df.columns:
                self.coverage_by_equity_group = (
                    grouped_df["within_threshold"].mean().round(2).to_dict()
                )

            # 5. Worst-Case Scenarios by Group
            self.max_cost_by_equity_group = (
                grouped_df["min_cost"].max().round(2).to_dict()
            )

            # 6. Tertile Groupings (Averaging the bin results into thirds)
            # Sorts the bins (e.g., 1-10) and splits them into 3 roughly equal chunks
            unique_bins = sorted(list(self.weighted_by_equity_group.keys()))
            if len(unique_bins) >= 3:
                chunks = np.array_split(unique_bins, 3)
                self.avg_lower_third_bins = np.mean(
                    [self.weighted_by_equity_group[b] for b in chunks[0]]
                )
                self.avg_middle_third_bins = np.mean(
                    [self.weighted_by_equity_group[b] for b in chunks[1]]
                )
                self.avg_upper_third_bins = np.mean(
                    [self.weighted_by_equity_group[b] for b in chunks[2]]
                )

                if self.avg_upper_third_bins and self.avg_upper_third_bins > 0:
                    self.inter_tertile_ratio = (
                        self.avg_lower_third_bins / self.avg_upper_third_bins
                    )

                    # Generate Inter-Tertile Ratio Descriptor
                    # (Assuming lower bins = higher deprivation, e.g., IMD Deciles 1-3)
                    if 0.95 <= self.inter_tertile_ratio <= 1.05:
                        self.inter_tertile_desc = (
                            "Balanced (Macro travel times are broadly equal)"
                        )
                    elif 1.05 < self.inter_tertile_ratio <= 1.25:
                        pct = (self.inter_tertile_ratio - 1.0) * 100
                        self.inter_tertile_desc = f"Slightly Inequitable (Most deprived travel {pct:.0f}% longer)"
                    elif self.inter_tertile_ratio > 1.25:
                        pct = (self.inter_tertile_ratio - 1.0) * 100
                        self.inter_tertile_desc = f"Highly Inequitable (Most deprived travel {pct:.0f}% longer)"
                    elif 0.75 <= self.inter_tertile_ratio < 0.95:
                        pct = (1.0 - self.inter_tertile_ratio) * 100
                        self.inter_tertile_desc = f"Slightly Progressive (Most deprived travel {pct:.0f}% shorter)"
                    else:
                        pct = (1.0 - self.inter_tertile_ratio) * 100
                        self.inter_tertile_desc = f"Highly Progressive (Most deprived travel {pct:.0f}% shorter)"
                else:
                    self.inter_tertile_ratio = np.nan
                    self.inter_tertile_desc = "N/A (Zero upper-third travel time)"

    def show_result_df(self):
        return self.evaluated_combination_df

    def return_solution_metrics(self):
        """
        INTERPRETATION GUIDE FOR SUMMARY TABLES & SORTING:

        1a. 'weighted_average'
            - LOWER is better. Represents travel time adjusted for specified weighting factors.

        1b. Travel Costs ('unweighted_average', '90th_percentile', 'max'):
           - LOWER is better. Represents travel time or distance.

        1c. 'total_cost'
           - LOWER is better. Total fixed cost of the selected sites (sum of
             the `cost_col` configured via `add_sites()`). `NaN` if no
             `cost_col` was configured. Only influences which solution is
             selected if explicitly passed as a weight (weights={"cost": ...}).

        2. Absolute Equity Gap ('gap_absolute_weighted'):
           - CLOSER TO 0 is better. Measures the flat minute/distance difference
             between the best-served and worst-served equity bands. High numbers
             mean severe geographical disparity.

        3. Relative Equity Gap ('gap_relative_weighted'):
           - CLOSER TO 1.0 is better. If it's 1.5, the worst-served group travels
             1.5x longer than the best-served group.

        4. Inter-Tertile Ratio ('inter_tertile_ratio'):
           - Measures macro-equity assuming lower bins = higher deprivation (e.g., IMD 1-3).
           - SORTING CRITERIA:
             * ITR > 1.0: Inequity. The most deprived third faces longer travel times
                          than the least deprived third (e.g., 1.25 = 25% longer travel).
             * ITR = 1.0: Perfect equality in macro travel times.
             * ITR < 1.0: Progressive equity. Travel times are shorter for the most
                          deprived communities.

        5. Coverage Metrics ('proportion_within_coverage_threshold', 'coverage_by_equity_group'):
           - HIGHER is better (Scale: 0.0 to 1.0). Represents accessibility. Look for
             solutions where coverage is both globally high and uniformly distributed
             across groups.
        """

        # Return weighted average
        return {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "coverage_threshold": self.coverage_threshold,
            "weighted_average": self.weighted_average,
            "unweighted_average": self.unweighted_average,
            "90th_percentile": self.percentile_90th,
            "max": self.max,
            "total_cost": self.total_cost,
            "proportion_within_coverage_threshold": self.proportion_within_coverage_threshold,
            # Granular Equity Collections
            "weighted_by_equity_group": self.weighted_by_equity_group,
            "unweighted_by_equity_group": self.unweighted_by_equity_group,
            "coverage_by_equity_group": self.coverage_by_equity_group,
            "max_cost_by_equity_group": self.max_cost_by_equity_group,
            # Numeric Aggregations
            "gap_absolute_weighted": self.gap_absolute_weighted,
            "gap_relative_weighted": self.gap_relative_weighted,
            "avg_lower_third_bins": self.avg_lower_third_bins,
            "avg_middle_third_bins": self.avg_middle_third_bins,
            "avg_upper_third_bins": self.avg_upper_third_bins,
            "inter_tertile_ratio": self.inter_tertile_ratio,
            # Verbal Interpretation Columns
            "gap_absolute_description": self.gap_absolute_desc,
            "gap_relative_description": self.gap_relative_desc,
            "inter_tertile_description": self.inter_tertile_desc,
            # Underlying per-region df
            "problem_df": self.evaluated_combination_df,
        }


# MARK: CLASS SiteSolutionSet
class SiteSolutionSet(
    MapsMixin,
    NonMapPlotsMixin,
    ParetoMixin,
    DistributionPlotsMixin,
    EquityPlotsMixin,
    SiteSolutionHotspotCalculationMixin,
    HotspotPlotMixin,
    SiteProblemEDAMixin,
):
    """
    Container for a set of evaluated site selection solutions.

    This class stores and provides convenient access to a collection of
    candidate solutions from a brute-force or heuristic search,
    along with their associated evaluation metrics. It supports returning
    and plotting details of the best-performing solutions.

    Parameters
    ----------
    solution_df : pandas.DataFrame
        DataFrame containing one row per evaluated solution. Typically includes:
        - "site_indices": Indices of selected sites for the solution.
        - One or more objective/metric columns (e.g., "weighted_average",
          "unweighted_average", "90th_percentile", etc.).
        The DataFrame is reset to a zero-based index upon initialisation.
    site_problem : object
        The originating problem instance used to generate and evaluate
        the solutions.
    objectives : str or list of str
        Objective(s) used to evaluate and rank the solutions (e.g.,
        "weighted_average", "mclp").
    n_sites : int, optional
        Number of sites selected in each solution (e.g., ``p`` in a p-median
        or p-center problem).

    Attributes
    ----------
    solution_df : pandas.DataFrame
        DataFrame of evaluated solutions with metrics.
    site_problem : object
        Problem definition associated with the solutions.
    objectives : str or list of str
        Objective(s) used in evaluation.
    n_sites : int or None
        Number of sites in each solution.

    Notes
    -----
    Solutions are typically pre-sorted before being passed to this class
    (e.g., by objective value and tie-breakers). The optional ``rank_on``
    argument in methods allows overriding this ordering dynamically.

    The structure of ``solution_df`` is assumed to be consistent with the
    outputs of the optimisation or search routine that generated it.
    """

    def __init__(self, solution_df, site_problem, objectives, n_sites=None):
        """
        Initialise a SiteSolutionSet instance.

        Parameters
        ----------
        solution_df : pandas.DataFrame
            DataFrame containing evaluated solutions. Each row represents a
            candidate solution and typically includes columns such as
            "site_indices", "site_names", and one or more objective metrics.
            The index is reset to a zero-based integer index on initialisation.
        site_problem : object
            The originating problem instance used to generate and evaluate
            the solutions.
        objectives : str or list of str
            Objective(s) used to evaluate the solutions.
        n_sites : int, optional
            Number of sites selected in each solution.

        Notes
        -----
        The input DataFrame is copied with its index reset to ensure consistent
        positional indexing for downstream operations.
        """
        self.solution_df = solution_df.reset_index(drop=True)
        self.site_problem = site_problem
        self.objectives = objectives
        self.n_sites = n_sites

        self.pareto_metrics = None

    def copy(self):
        return copy.deepcopy(self)

    def show_solutions_colnames(self, return_list=False):
        if not return_list:
            print(self.solution_df.columns)
        else:
            return self.solution_df.columns

    def show_solutions(self, rounding=2, n_best=None):
        """
        Return the solution DataFrame with rounded values.

        Parameters
        ----------
        rounding : int, default=2
            Number of decimal places to round numeric columns to.

        Returns
        -------
        pandas.DataFrame
            The solution DataFrame with numeric values rounded to the specified
            precision.

        Notes
        -----
        This method does not modify the underlying DataFrame; it returns a
        rounded copy.
        """
        if rounding is None:
            return self.solution_df.head(n_best)
        else:
            return round(self.solution_df, rounding).head(n_best)

    def return_best_combination_details(self, rank_on=None, top_n=1):
        """
        Return details of the top-ranked solution(s).

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, solutions are
            sorted in ascending order before selecting the top entries.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.
        top_n : int, default=1
            Number of top solutions to return.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the top ``top_n`` solutions, including all
            available columns. The index is reset in the returned DataFrame.

        Notes
        -----
        Sorting is performed in ascending order, so lower values are assumed
        to represent better solutions for the specified ranking metric.
        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on).head(top_n).reset_index()
        else:
            return self.solution_df.head(top_n).reset_index()

    def return_best_combination_site_indices(self, rank_on=None):
        """
        Return the site indices for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, the solution
            with the lowest value in this column is selected.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_indices" column for the best solution.
            Typically a list or array of site indices.

        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on)["site_indices"].iloc[0]
        else:
            return self.solution_df["site_indices"].iloc[0]

    def return_best_combination_site_names(self, rank_on=None):
        """
        Return the site names for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, the solution
            with the lowest value in this column is selected.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_indices" column for the best solution.
            Typically a list or array of site indices.

        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on)["site_names"].iloc[0]
        else:
            return self.solution_df["site_names"].iloc[0]

    def summary_table(self):
        pass


class SolutionComparator(SolutionComparatorMethodsMixin, SolutionComparatorPlotsMixin):
    """
    Tools to compare two SiteSolutionSet objects.
    """

    def __init__(self, solution_set_a, solution_set_b, labels=("Set A", "Set B")):
        self.set_a = solution_set_a
        self.set_b = solution_set_b
        self.labels = labels
