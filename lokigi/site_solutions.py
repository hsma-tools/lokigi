import warnings

import numpy as np
import pandas as pd

from lokigi.utils import _min_max_normalize, _select_solution, _sort_solutions_by_metric

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
        Proportion of total demand that falls within the coverage threshold,
        weighted by the demand registered via `add_demand()`. `NaN` if no
        `threshold_for_coverage` was supplied.
    proportion_regions_within_coverage_threshold : float
        Proportion of demand *regions* that fall within the coverage
        threshold, counting every region equally regardless of its demand.
        Identical to `proportion_within_coverage_threshold` when demand is
        uniform (including when `add_demand()` was never called).



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
        self.coverage_regions_by_equity_group = {}
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
        #
        # "demand" and "equity" are canonicalised to lowercase here because
        # this method can be reached directly (evaluate_single_solution_
        # single_objective bypasses solve()'s own case-insensitive-but-
        # unnormalised validation), and every comparison below -- the
        # legacy-fallback check just below, and the col_name/direction
        # resolution in the loop that follows -- is case-sensitive for
        # these two built-in keywords. Without this, a differently-cased
        # key like "Demand" silently matched no column, leaving the
        # compound row weights all zero and crashing np.average with
        # "Weights sum to zero". Additional-data labels are user-defined
        # exact strings (not built-in keywords) and are deliberately left
        # untouched -- they are matched case-sensitively against whatever
        # was registered via add_additional_data(label=...).
        row_level_weights = (
            None
            if weights is None
            else {
                (k.lower() if k.lower() in ("demand", "equity") else k): v
                for k, v in weights.items()
                if k.lower() != "cost"
            }
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
        else:
            # print("Weighting p-median by custom options")
            # print(f"Weights: {weights}")

            # Initialize an array of zeros to build our blended row-level weights
            compound_weights = np.zeros(len(self.evaluated_combination_df))
            self.extra_metrics = {}

            for label, weight in row_level_weights.items():
                # Map the user's label to the correct DataFrame column name,
                # and resolve its weighting direction, BEFORE checking that
                # the column actually exists. This ordering matters: doing
                # the existence check first (as a previous version of this
                # code did) meant a genuinely unrecognised label -- one that
                # doesn't match "demand", "equity", or any registered
                # additional-data label -- had its column lookup silently
                # fail and skipped the whole block instead of raising the
                # "does not correspond to..." error below, leaving
                # compound_weights all zero and crashing np.average with a
                # confusing "Weights sum to zero" further down. Direct calls
                # to evaluate_single_solution_single_objective (which skip
                # solve()'s own case-insensitive key validation) are the
                # main way this was reachable.
                direction = None

                if label == "demand":
                    col_name = self.site_problem._demand_data_demand_col
                    direction = "higher_better"  # 'demand' defaults to higher_better

                elif label == "equity":
                    col_name = self.site_problem._equity_data_equity_col
                    if col_name is not None:
                        # Equity weighting exists to prioritise worse-off
                        # regions, so whichever end of the equity scale is
                        # disadvantaged is the end that gets the weight.
                        direction = (
                            "lower_better"
                            if self.site_problem._equity_data_disadvantaged_end
                            == "low"
                            else "higher_better"
                        )

                else:
                    col_name = label
                    # Look up directionality from the problem configuration
                    if self.site_problem.additional_data is not None:
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

                if (
                    direction is None
                    or col_name is None
                    or col_name not in self.evaluated_combination_df.columns
                ):
                    raise ValueError(
                        f"Weight key '{label}' does not correspond to demand, "
                        "equity, or any registered additional dataset. Register "
                        "it via add_additional_data() first, or remove it from "
                        "the weights dict."
                    )

                # Extract raw data
                column_data = self.evaluated_combination_df[col_name].astype(float)

                # A NaN here would silently poison the whole compound
                # weight vector (NaN * weight propagates through
                # np.average into every score), so fail loudly instead.
                # NaNs usually mean the weighted dataset (equity or
                # additional data) doesn't cover every demand location.
                if column_data.isna().any():
                    raise ValueError(
                        f"Cannot weight by '{label}': "
                        f"{int(column_data.isna().sum())} demand row(s) "
                        f"have missing values in '{col_name}'. This "
                        "usually means the dataset does not cover every "
                        "demand location. Please fill or filter the "
                        f"missing values, or remove '{label}' from the "
                        "weights dict."
                    )

                # Handle directionality by negating BEFORE normalising: for
                # lower_better this maps the smallest raw value to 1.0 and
                # the largest to 0.0 (identical to inverting afterwards),
                # but keeps the identical-values edge case on the equal
                # (full) baseline weight from constant_fill. Inverting
                # after normalising would turn that neutral 1.0 into an
                # all-zero weight vector and crash np.average with
                # "Weights sum to zero".
                if direction == "lower_better":
                    column_data = -column_data

                # Min-Max Normalization to a 0.0 - 1.0 scale. Edge case: if
                # all values are identical, give them equal (full) baseline
                # weight rather than 0.
                norm_data = _min_max_normalize(column_data, constant_fill=1.0)

                # Accumulate the normalized, directional weight
                compound_weights += norm_data * weight

            active_weights = pd.Series(
                compound_weights, index=self.evaluated_combination_df.index
            )

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

        # Primary travel-cost statistics + equity breakdown, computed from
        # "min_cost"/"within_threshold". Any registered secondary travel
        # matrices get the identical treatment below, applied to their own
        # suffixed columns -- see _compute_travel_metrics.
        primary_metrics = self._compute_travel_metrics(
            "min_cost", "within_threshold", active_weights
        )
        self.weighted_average = primary_metrics["weighted_average"]
        self.unweighted_average = primary_metrics["unweighted_average"]
        self.percentile_90th = primary_metrics["percentile_90th"]
        self.max = primary_metrics["max"]
        self.proportion_within_coverage_threshold = primary_metrics[
            "proportion_within_coverage_threshold"
        ]
        self.proportion_regions_within_coverage_threshold = primary_metrics[
            "proportion_regions_within_coverage_threshold"
        ]
        self.weighted_by_equity_group = primary_metrics["weighted_by_equity_group"]
        self.unweighted_by_equity_group = primary_metrics["unweighted_by_equity_group"]
        self.coverage_by_equity_group = primary_metrics["coverage_by_equity_group"]
        self.coverage_regions_by_equity_group = primary_metrics[
            "coverage_regions_by_equity_group"
        ]
        self.max_cost_by_equity_group = primary_metrics["max_cost_by_equity_group"]
        self.gap_absolute_weighted = primary_metrics["gap_absolute_weighted"]
        self.gap_absolute_desc = primary_metrics["gap_absolute_desc"]
        self.gap_relative_weighted = primary_metrics["gap_relative_weighted"]
        self.gap_relative_desc = primary_metrics["gap_relative_desc"]
        self.avg_lower_third_bins = primary_metrics["avg_lower_third_bins"]
        self.avg_middle_third_bins = primary_metrics["avg_middle_third_bins"]
        self.avg_upper_third_bins = primary_metrics["avg_upper_third_bins"]
        self.inter_tertile_ratio = primary_metrics["inter_tertile_ratio"]
        self.inter_tertile_desc = primary_metrics["inter_tertile_desc"]

        # Secondary travel matrices: same statistics, computed from their
        # own suffixed `min_cost__<label>` / `within_threshold__<label>`
        # columns (added in site.py's evaluate_single_solution_single_
        # objective). Only the core five + float equity aggregations are
        # kept in return_solution_metrics()'s schema -- see that method.
        self.secondary_metrics = {}
        for label in getattr(self.site_problem, "secondary_travel_matrices", {}):
            cost_col = f"min_cost__{label}"
            within_col = f"within_threshold__{label}"
            if cost_col not in self.evaluated_combination_df.columns:
                continue
            self.secondary_metrics[label] = self._compute_travel_metrics(
                cost_col, within_col, active_weights
            )

    def _coverage_demand_series(self):
        """
        Raw per-region demand, used to weight the coverage metrics.

        Deliberately the raw demand column rather than the compound
        `active_weights` vector that `weighted_average` uses: those weights
        blend demand with equity and any additional datasets whenever a
        multi-key `weights=` dict is passed, which would make "proportion of
        demand covered" silently mean something different from one solve()
        call to the next.

        Returns None when there is no demand column to weight by. That is
        only reachable via a direct call to
        `evaluate_single_solution_single_objective()`, which bypasses
        solve()'s equal-demand fallback; callers then fall back to the
        unweighted per-region proportion, since with no demand data the two
        metrics are the same quantity.
        """
        demand_col = getattr(self.site_problem, "_demand_data_demand_col", None)
        if (
            demand_col is None
            or demand_col not in self.evaluated_combination_df.columns
        ):
            return None
        return self.evaluated_combination_df[demand_col].astype(float)

    @staticmethod
    def _coverage_proportion(within_flags, demand):
        """
        Proportion covered -- demand-weighted when `demand` is supplied,
        otherwise the plain share of regions.

        The all-NaN check is load-bearing. `within_flags` is entirely NaN
        whenever no `threshold_for_coverage` was given, and that must stay
        NaN ("not measured"). It cannot be left to the arithmetic below,
        because pandas' `.sum()` skips NaN: the weighted branch would
        otherwise return a confident 0.0 -- "none of the demand is covered"
        -- for a problem where coverage was never assessed at all.
        """
        if within_flags.isna().all():
            return np.nan

        covered = within_flags.astype(float)

        if demand is None:
            return float(covered.sum() / len(covered))

        total_demand = demand.sum()
        if total_demand <= 0:
            # Defensive only, and not reachable through the public API: a
            # zero-demand problem already raises "Weights sum to zero" from
            # np.average when weighted_average is computed further up.
            return np.nan

        return float((covered * demand).sum() / total_demand)

    def _compute_travel_metrics(self, cost_col, within_col, active_weights):
        """
        Compute weighted/unweighted travel-cost summary statistics and the
        equity breakdown for one travel matrix's min-cost column.

        Used once for the primary matrix (`cost_col="min_cost"`) and once
        per registered secondary travel matrix (`cost_col="min_cost__
        <label>"`), so a secondary matrix gets identical treatment to the
        primary one rather than a parallel, potentially-diverging
        implementation.
        """
        df = self.evaluated_combination_df

        weighted_average = np.average(df[cost_col], weights=active_weights)
        unweighted_average = np.average(df[cost_col])
        percentile_90th = np.percentile(df[cost_col], q=90)
        max_cost = np.max(df[cost_col])

        demand_series = self._coverage_demand_series()
        proportion_within_coverage_threshold = self._coverage_proportion(
            df[within_col], demand_series
        )
        proportion_regions_within_coverage_threshold = self._coverage_proportion(
            df[within_col], None
        )

        weighted_by_equity_group = {}
        unweighted_by_equity_group = {}
        coverage_by_equity_group = {}
        coverage_regions_by_equity_group = {}
        max_cost_by_equity_group = {}
        gap_absolute_weighted = None
        gap_absolute_desc = "N/A (No equity data)"
        gap_relative_weighted = None
        gap_relative_desc = "N/A (No equity data)"
        avg_lower_third_bins = None
        avg_middle_third_bins = None
        avg_upper_third_bins = None
        inter_tertile_ratio = None
        inter_tertile_desc = "N/A (No equity data)"

        equity_col = getattr(self.site_problem, "_equity_data_equity_col", None)

        if equity_col and equity_col in df.columns:
            grouped_df = df.groupby(equity_col)

            # 1. Unweighted average by equity group
            unweighted_by_equity_group = grouped_df[cost_col].mean().round(2).to_dict()

            # 2. Weighted average by equity group (matching global composite weights logic)
            for band, group in grouped_df:
                # Extract matching row weights for this specific group
                group_weights = active_weights.loc[group.index]

                # Avoid ZeroDivisionError if the combined weight for this band is 0
                if group_weights.sum() > 0:
                    weighted_by_equity_group[band] = np.average(
                        group[cost_col], weights=group_weights
                    ).round(2)
                else:
                    weighted_by_equity_group[band] = group[cost_col].mean()

            # 3. Disparity Metrics & Verbal Descriptors
            if weighted_by_equity_group:
                weighted_vals = list(weighted_by_equity_group.values())
                group_min_cost = min(weighted_vals)
                group_max_cost = max(weighted_vals)

                gap_absolute_weighted = group_max_cost - group_min_cost
                gap_absolute_desc = f"Spread of {gap_absolute_weighted:.1f} units between best and worst groups"

                if group_min_cost > 0:
                    gap_relative_weighted = group_max_cost / group_min_cost

                    # Generate Relative Gap Descriptor
                    if gap_relative_weighted <= 1.005:
                        gap_relative_desc = "Perfect Parity"
                    elif gap_relative_weighted <= 1.10:
                        gap_relative_desc = (
                            "Minimal Disparity (Worst group travels <10% longer)"
                        )
                    elif gap_relative_weighted <= 1.30:
                        gap_relative_desc = (
                            "Moderate Disparity (Worst group travels 10-30% longer)"
                        )
                    else:
                        pct_longer = (gap_relative_weighted - 1.0) * 100
                        gap_relative_desc = f"Significant Disparity (Worst group travels {pct_longer:.0f}% longer)"
                else:
                    gap_relative_weighted = np.nan
                    gap_relative_desc = "N/A (Zero baseline cost)"

            # 4. Coverage Equity (Thresholds by Group)
            # Mirrors the global pair above, under the same naming rule: the
            # unqualified dict is demand-weighted, the `regions` one is the
            # plain share of regions in each band.
            if within_col in df.columns:
                coverage_regions_by_equity_group = {
                    band: round(self._coverage_proportion(group[within_col], None), 2)
                    for band, group in grouped_df
                }
                coverage_by_equity_group = {
                    band: round(
                        self._coverage_proportion(
                            group[within_col],
                            None
                            if demand_series is None
                            else demand_series.loc[group.index],
                        ),
                        2,
                    )
                    for band, group in grouped_df
                }

            # 5. Worst-Case Scenarios by Group
            max_cost_by_equity_group = grouped_df[cost_col].max().round(2).to_dict()

            # 6. Tertile Groupings (Averaging the bin results into thirds)
            # Sorts the bins (e.g., 1-10) and splits them into 3 roughly equal chunks
            unique_bins = sorted(list(weighted_by_equity_group.keys()))
            if len(unique_bins) >= 3:
                chunks = np.array_split(unique_bins, 3)
                avg_lower_third_bins = np.mean(
                    [weighted_by_equity_group[b] for b in chunks[0]]
                )
                avg_middle_third_bins = np.mean(
                    [weighted_by_equity_group[b] for b in chunks[1]]
                )
                avg_upper_third_bins = np.mean(
                    [weighted_by_equity_group[b] for b in chunks[2]]
                )

                if avg_upper_third_bins and avg_upper_third_bins > 0:
                    inter_tertile_ratio = avg_lower_third_bins / avg_upper_third_bins

                    # Generate Inter-Tertile Ratio Descriptor
                    # (Assuming lower bins = higher deprivation, e.g., IMD Deciles 1-3)
                    if 0.95 <= inter_tertile_ratio <= 1.05:
                        inter_tertile_desc = (
                            "Balanced (Macro travel times are broadly equal)"
                        )
                    elif 1.05 < inter_tertile_ratio <= 1.25:
                        pct = (inter_tertile_ratio - 1.0) * 100
                        inter_tertile_desc = f"Slightly Inequitable (Most deprived travel {pct:.0f}% longer)"
                    elif inter_tertile_ratio > 1.25:
                        pct = (inter_tertile_ratio - 1.0) * 100
                        inter_tertile_desc = f"Highly Inequitable (Most deprived travel {pct:.0f}% longer)"
                    elif 0.75 <= inter_tertile_ratio < 0.95:
                        pct = (1.0 - inter_tertile_ratio) * 100
                        inter_tertile_desc = f"Slightly Progressive (Most deprived travel {pct:.0f}% shorter)"
                    else:
                        pct = (1.0 - inter_tertile_ratio) * 100
                        inter_tertile_desc = f"Highly Progressive (Most deprived travel {pct:.0f}% shorter)"
                else:
                    inter_tertile_ratio = np.nan
                    inter_tertile_desc = "N/A (Zero upper-third travel time)"

        return {
            "weighted_average": weighted_average,
            "unweighted_average": unweighted_average,
            "percentile_90th": percentile_90th,
            "max": max_cost,
            "proportion_within_coverage_threshold": proportion_within_coverage_threshold,
            "proportion_regions_within_coverage_threshold": proportion_regions_within_coverage_threshold,
            "weighted_by_equity_group": weighted_by_equity_group,
            "unweighted_by_equity_group": unweighted_by_equity_group,
            "coverage_by_equity_group": coverage_by_equity_group,
            "coverage_regions_by_equity_group": coverage_regions_by_equity_group,
            "max_cost_by_equity_group": max_cost_by_equity_group,
            "gap_absolute_weighted": gap_absolute_weighted,
            "gap_absolute_desc": gap_absolute_desc,
            "gap_relative_weighted": gap_relative_weighted,
            "gap_relative_desc": gap_relative_desc,
            "avg_lower_third_bins": avg_lower_third_bins,
            "avg_middle_third_bins": avg_middle_third_bins,
            "avg_upper_third_bins": avg_upper_third_bins,
            "inter_tertile_ratio": inter_tertile_ratio,
            "inter_tertile_desc": inter_tertile_desc,
        }

    def show_result_df(self):
        return self.evaluated_combination_df

    def return_solution_metrics(self, full_secondary_metrics: bool = False):
        """
        Parameters
        ----------
        full_secondary_metrics : bool, default False
            If False (the default), each registered secondary travel matrix
            contributes only its core five metrics plus the float-valued
            equity aggregations (see point 6 below) -- matching the output
            shape prior to this parameter's introduction. If True, every
            registered secondary matrix also contributes its dict-valued
            equity breakdowns (`weighted_by_equity_group__<label>`, etc.)
            and description strings, exactly mirroring what the primary
            matrix already always returns unsuffixed. This costs nothing
            extra to compute -- `_compute_travel_metrics` already produces
            these values for every matrix regardless -- it only changes
            which of the already-computed keys get included here.

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
             across groups. Both are `NaN` if no `threshold_for_coverage` was given.
           - NAMING RULE: an unqualified coverage metric is weighted by the demand
             registered via `add_demand()`, so it answers "what share of *people*
             are covered". The `regions` variants
             ('proportion_regions_within_coverage_threshold',
             'coverage_regions_by_equity_group') count every region equally and
             answer "what share of *places* are covered". The two coincide when
             demand is uniform, including when `add_demand()` was never called.
           - The demand-weighted figure is the one 'mclp' optimises, matching the
             textbook Maximal Covering Location Problem.

        6. Secondary travel matrices (columns suffixed `__<label>`, e.g.
           'weighted_average__public_transport'):
           - Registered via `add_secondary_travel_matrix(label=...)`. Same metrics
             and sort direction as their unsuffixed counterparts above (1a/1b/5),
             computed against that matrix's own travel costs instead of the
             primary matrix. By default, only the core scalar metrics (both
             coverage proportions included) plus the
             float-valued equity aggregations (gap_absolute_weighted,
             gap_relative_weighted, avg_*_third_bins, inter_tertile_ratio) are
             included per matrix, to keep this table from growing unboundedly
             with each registered matrix -- pass `full_secondary_metrics=True`
             to also include the dict-valued equity breakdowns and description
             strings, matching what the primary matrix already returns. The
             underlying per-region `problem_df` always carries
             `min_cost__<label>` / `selected_site__<label>` /
             `within_threshold__<label>` regardless of this setting.
        """

        # Return weighted average
        metrics = {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "coverage_threshold": self.coverage_threshold,
            "weighted_average": self.weighted_average,
            "unweighted_average": self.unweighted_average,
            "90th_percentile": self.percentile_90th,
            "max": self.max,
            "total_cost": self.total_cost,
            "proportion_within_coverage_threshold": self.proportion_within_coverage_threshold,
            "proportion_regions_within_coverage_threshold": self.proportion_regions_within_coverage_threshold,
            # Granular Equity Collections
            "weighted_by_equity_group": self.weighted_by_equity_group,
            "unweighted_by_equity_group": self.unweighted_by_equity_group,
            "coverage_by_equity_group": self.coverage_by_equity_group,
            "coverage_regions_by_equity_group": self.coverage_regions_by_equity_group,
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

        # Secondary travel matrices: core five metrics + float-valued equity
        # aggregations by default (see the interpretation guide above); pass
        # full_secondary_metrics=True to also include the dict-valued
        # breakdowns and description strings, matching the primary matrix.
        for label, secondary in self.secondary_metrics.items():
            metrics[f"weighted_average__{label}"] = secondary["weighted_average"]
            metrics[f"unweighted_average__{label}"] = secondary["unweighted_average"]
            metrics[f"90th_percentile__{label}"] = secondary["percentile_90th"]
            metrics[f"max__{label}"] = secondary["max"]
            metrics[f"proportion_within_coverage_threshold__{label}"] = secondary[
                "proportion_within_coverage_threshold"
            ]
            metrics[f"proportion_regions_within_coverage_threshold__{label}"] = (
                secondary["proportion_regions_within_coverage_threshold"]
            )
            metrics[f"gap_absolute_weighted__{label}"] = secondary[
                "gap_absolute_weighted"
            ]
            metrics[f"gap_relative_weighted__{label}"] = secondary[
                "gap_relative_weighted"
            ]
            metrics[f"avg_lower_third_bins__{label}"] = secondary[
                "avg_lower_third_bins"
            ]
            metrics[f"avg_middle_third_bins__{label}"] = secondary[
                "avg_middle_third_bins"
            ]
            metrics[f"avg_upper_third_bins__{label}"] = secondary[
                "avg_upper_third_bins"
            ]
            metrics[f"inter_tertile_ratio__{label}"] = secondary["inter_tertile_ratio"]

            if full_secondary_metrics:
                metrics[f"weighted_by_equity_group__{label}"] = secondary[
                    "weighted_by_equity_group"
                ]
                metrics[f"unweighted_by_equity_group__{label}"] = secondary[
                    "unweighted_by_equity_group"
                ]
                metrics[f"coverage_by_equity_group__{label}"] = secondary[
                    "coverage_by_equity_group"
                ]
                metrics[f"coverage_regions_by_equity_group__{label}"] = secondary[
                    "coverage_regions_by_equity_group"
                ]
                metrics[f"max_cost_by_equity_group__{label}"] = secondary[
                    "max_cost_by_equity_group"
                ]
                metrics[f"gap_absolute_description__{label}"] = secondary[
                    "gap_absolute_desc"
                ]
                metrics[f"gap_relative_description__{label}"] = secondary[
                    "gap_relative_desc"
                ]
                metrics[f"inter_tertile_description__{label}"] = secondary[
                    "inter_tertile_desc"
                ]

        return metrics


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

    def _resolve_travel_columns(self, matrix=None):
        """
        Return (cost_col, selected_site_col, within_threshold_col, unit,
        suffix) for the given secondary travel matrix label, or for the
        primary travel matrix if `matrix` is None.

        Used by every travel-related plotting method so that a single
        `matrix=` keyword switches between the primary matrix and any
        registered secondary one, instead of each plot hardcoding
        "min_cost" / "selected_site" / "within_threshold".
        """
        if matrix is None:
            return (
                "min_cost",
                "selected_site",
                "within_threshold",
                self.site_problem._travel_matrix_unit,
                "",
            )

        registered = self.site_problem.secondary_travel_matrices
        if matrix not in registered:
            raise ValueError(
                f"Unknown secondary travel matrix '{matrix}'. Registered "
                f"labels: {sorted(registered)}."
            )

        suffix = f"__{matrix}"
        return (
            f"min_cost{suffix}",
            f"selected_site{suffix}",
            f"within_threshold{suffix}",
            registered[matrix]["unit"],
            suffix,
        )

    def _resolve_coverage_threshold(self, matrix, coverage_threshold_value):
        """
        Return the coverage threshold that applies for `matrix` (None = the
        primary matrix). Secondary matrices may have their own
        `threshold_for_coverage` set via `add_secondary_travel_matrix()`;
        if not, they fall back to `coverage_threshold_value` (the primary
        threshold passed to `solve()`), matching the fallback used when the
        per-solution `within_threshold__<label>` column was computed.
        """
        if matrix is None:
            return coverage_threshold_value
        matrix_threshold = self.site_problem.secondary_travel_matrices[matrix][
            "threshold_for_coverage"
        ]
        return (
            matrix_threshold if matrix_threshold is not None else coverage_threshold_value
        )

    def show_solutions_colnames(self, return_list=False):
        if not return_list:
            print(self.solution_df.columns)
        else:
            return self.solution_df.columns

    def _expand_dict_columns(self, df):
        """
        Return a copy of `df` with every dict-valued column (e.g. the
        equity-group breakdowns `weighted_by_equity_group`,
        `coverage_by_equity_group__<label>`, etc.) replaced by one column
        per dict key, named `<column>__<key>`.

        Columns are detected by content, not by name, so this picks up any
        dict-valued column regardless of which matrix or metric produced it.
        Columns whose values are not dicts (including ones that are simply
        empty, e.g. when no equity data is registered) are left untouched.
        """
        dict_cols = [
            col for col in df.columns if df[col].apply(lambda v: isinstance(v, dict)).any()
        ]
        if not dict_cols:
            return df

        df = df.copy()
        for col in dict_cols:
            expanded = pd.json_normalize(df[col]).add_prefix(f"{col}__")
            expanded.index = df.index
            col_pos = df.columns.get_loc(col)
            df = pd.concat(
                [df.iloc[:, :col_pos], expanded, df.iloc[:, col_pos + 1 :]], axis=1
            )
        return df

    def show_solutions(self, rounding=2, n_best=None, expand_dict_columns=False, inplace=False):
        """
        Return the solution DataFrame with rounded values.

        Parameters
        ----------
        rounding : int, default=2
            Number of decimal places to round numeric columns to.
        expand_dict_columns : bool, default False
            If True, any column holding dict values -- the equity-group
            breakdowns (`weighted_by_equity_group`, `coverage_by_equity_group`,
            etc.) and their secondary-matrix equivalents when
            `solve(..., full_secondary_metrics=True)` was used -- is expanded
            into one column per dict key, named `<column>__<key>`. This is
            usually more useful for display/export than a column of dicts,
            but is off by default to keep `solution_df`'s shape unchanged for
            existing callers.
        inplace : bool, default False
            If True (and `expand_dict_columns=True`), the expansion is also
            written back to `self.solution_df`, so it persists for
            subsequent calls, plotting, `rank_on`, etc. Has no effect unless
            `expand_dict_columns=True`, in which case a `UserWarning` is
            raised instead, since passing `inplace=True` alone does nothing
            and likely indicates the caller meant to also pass
            `expand_dict_columns=True`. Rounding is never made permanent --
            only the column expansion can be.

        Returns
        -------
        pandas.DataFrame
            The solution DataFrame with numeric values rounded to the specified
            precision.

        Notes
        -----
        Unless `inplace=True`, this method does not modify the underlying
        DataFrame; it returns a rounded (and optionally expanded) copy.
        """
        if inplace and not expand_dict_columns:
            warnings.warn(
                "show_solutions(inplace=True) has no effect unless "
                "expand_dict_columns=True is also passed.",
                stacklevel=2,
            )

        df = self.solution_df
        if expand_dict_columns:
            df = self._expand_dict_columns(df)
            if inplace:
                self.solution_df = df

        if rounding is None:
            return df.head(n_best)
        else:
            return round(df, rounding).head(n_best)

    def return_best_combination_details(self, rank_on=None, top_n=1):
        """
        Return details of the top-ranked solution(s).

        Parameters
        ----------
        rank_on : str, optional
            Column name to rank the solutions by before selecting the top
            entries. If None, the existing order of ``solution_df``, which is
            based on the objective selected, is used.
        top_n : int, default=1
            Number of top solutions to return.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the top ``top_n`` solutions, including all
            available columns. The index is reset in the returned DataFrame.

        Notes
        -----
        Sort direction is resolved from the metric, so "top" always means
        best: coverage proportions are ranked highest-first, and every other
        reported metric is a travel cost and is ranked lowest-first.
        """
        if rank_on is not None:
            return (
                _sort_solutions_by_metric(self.solution_df, rank_on)
                .head(top_n)
                .reset_index()
            )
        else:
            return self.solution_df.head(top_n).reset_index()

    def return_best_combination_site_indices(self, rank_on=None):
        """
        Return the site indices for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to rank the solutions by. If provided, the solution
            with the BEST value in this column is selected -- the highest for
            coverage proportions, the lowest for travel-cost metrics.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_indices" column for the best solution.
            Typically a list or array of site indices.

        """
        if rank_on is not None:
            return _sort_solutions_by_metric(self.solution_df, rank_on)[
                "site_indices"
            ].iloc[0]
        else:
            return self.solution_df["site_indices"].iloc[0]

    def return_best_combination_site_names(self, rank_on=None):
        """
        Return the site names for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to rank the solutions by. If provided, the solution
            with the BEST value in this column is selected -- the highest for
            coverage proportions, the lowest for travel-cost metrics.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_names" column for the best solution.
            Typically a list or array of site names.

        """
        if rank_on is not None:
            return _sort_solutions_by_metric(self.solution_df, rank_on)[
                "site_names"
            ].iloc[0]
        else:
            return self.solution_df["site_names"].iloc[0]

    def site_allocation_summary(
        self,
        by="demand",
        rank_on=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
    ):
        """
        Per-site summary of a chosen solution: the share of demand (or of
        regions) whose closest selected site is each site, and the average
        travel cost incurred by that group.

        Answers "how much work does this site actually do, and how far do
        the people it serves have to travel?" for one chosen solution --
        useful both for weighing up whether an additional site earns its
        cost (a site closest to only a small share of demand is a weak case
        for opening, even where it visibly lowers the average travel time),
        and for comparing how consolidating or closing sites changes
        typical travel distance for the people affected.

        Parameters
        ----------
        by : {"demand", "regions"}, default "demand"
            Basis for the `proportion` and `average_travel_cost` columns.
            "demand" weights each region by the demand registered via
            `add_demand()`, so `average_travel_cost` is the demand-weighted
            mean travel cost among a site's closest regions -- the same
            weighting as the solution-level `weighted_average`. "regions"
            counts every region equally, matching `unweighted_average`.
            Follows the same people-vs-places naming rule as the coverage
            metrics (see `EvaluatedCombination.return_solution_metrics`):
            unqualified means demand-weighted. The two coincide when demand
            is uniform, including when `add_demand()` was never called.
        rank_on : str, optional
        solution_rank : int, default 1
        site_names : list, optional
        site_indices : list, optional
            Solution selection, as in `plot_best_combination`. Priority is
            site_indices > site_names > rank_on/solution_rank.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Summarises allocation, and
            computes `average_travel_cost`, under that matrix's own
            `selected_site__<label>` / `min_cost__<label>` columns instead
            of the primary matrix's.

        Returns
        -------
        pandas.DataFrame
            One row per site in the chosen solution, indexed by site name
            ("site") in canonical site-index order. Columns: `n_regions`,
            `total_demand` (omitted when no demand data is registered),
            `proportion` (sums to 1.0 across the frame), and
            `average_travel_cost` (in the travel matrix's registered unit
            -- e.g. minutes, or miles if the matrix was built from
            distances rather than times).

        Raises
        ------
        ValueError
            If `by` is not "demand" or "regions", or if `by="demand"` but no
            demand column is registered on the problem (see Notes).

        Notes
        -----
        Every selected site appears, including any that is closest to no
        region at all -- it gets an explicit 0 row in `n_regions` and
        `proportion` rather than being dropped. That case is usually the
        finding being looked for, so silently losing it would defeat the
        point of the method. `average_travel_cost` is `NaN` for such a site
        rather than `0`: there is no travel cost to average over zero
        regions, and `0` would misleadingly read as "instant to reach".

        Regions exactly equidistant from two selected sites are assigned to
        one of them, not split: the underlying allocation uses
        `DataFrame.idxmin`, and its candidate columns are ordered by
        canonical site index, so exact ties go to the lowest-indexed site.
        Deterministic across runs, but arbitrary -- exact ties are rare on
        real travel matrices and common on synthetic ones.

        `average_travel_cost` was inspired by work from Gill Baker, who
        used average travel distance per patient -- split by which site
        was closest -- to show that centralising services onto fewer sites
        would roughly double typical travel distance for patients, while
        adding a third site offered only limited benefit over the existing
        two.
        """
        if by not in ("demand", "regions"):
            raise ValueError(f"by must be 'demand' or 'regions', got {by!r}.")

        cost_col, selected_site_col, _, _, _ = self._resolve_travel_columns(matrix)

        solution = _select_solution(
            self.solution_df,
            rank_on=rank_on,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
        )
        selected_sites = list(solution["site_names"].iloc[0])
        per_region = solution["problem_df"].iloc[0]

        demand_col = getattr(self.site_problem, "_demand_data_demand_col", None)
        has_demand = demand_col is not None and demand_col in per_region.columns
        if by == "demand" and not has_demand:
            raise ValueError(
                "by='demand' requires demand data. No demand column is "
                "registered on this problem -- call add_demand(), or pass "
                "by='regions' to count every region equally."
            )

        counts_raw = per_region.groupby(selected_site_col).size()
        unexpected = set(counts_raw.index) - set(selected_sites)
        if unexpected:
            # Reindexing below would silently DROP these, renormalising the
            # remaining proportions to look complete. Warn instead: an
            # allocation to a site that isn't in the solution means
            # problem_df and site_names have gone out of step.
            warnings.warn(
                f"site_allocation_summary: {selected_site_col} contains "
                f"site(s) not in this solution's site_names: "
                f"{sorted(unexpected)}. These are excluded from the summary.",
                stacklevel=2,
            )

        # Reindex against the solution's own site list, not the observed
        # groups. A selected site that is closest to NO region is absent
        # from the groupby entirely, and that is exactly the case this
        # method exists to surface: "we opened a third site and it picks up
        # nothing". Left as a plain groupby it would vanish from the table
        # and the reader would conclude the solution has two sites, not
        # three.
        n_regions = counts_raw.reindex(selected_sites, fill_value=0)

        result = pd.DataFrame({"n_regions": n_regions})
        result.index.name = "site"

        if has_demand:
            total_demand = per_region[demand_col].astype(float).sum()
            demand_by_site = (
                per_region.groupby(selected_site_col)[demand_col]
                .sum()
                .reindex(selected_sites, fill_value=0.0)
            )
            result["total_demand"] = demand_by_site

        if by == "demand":
            result["proportion"] = result["total_demand"] / total_demand

            # sum(cost * demand) / sum(demand) per site, mirroring how the
            # solution-level `weighted_average` is computed. Grouping and
            # dividing only over sites that actually appear in per_region
            # (i.e. those with n_regions > 0) means the reindex below is
            # the only place a zero-allocation site's average_travel_cost
            # is produced, and it comes out NaN (no group to divide) rather
            # than a 0/0 division warning.
            weighted_cost = per_region[cost_col] * per_region[demand_col].astype(float)
            weighted_cost_sum = weighted_cost.groupby(per_region[selected_site_col]).sum()
            demand_sum = per_region.groupby(selected_site_col)[demand_col].sum()
            result["average_travel_cost"] = (weighted_cost_sum / demand_sum).reindex(
                selected_sites
            )
        else:
            total_regions = len(per_region)
            result["proportion"] = result["n_regions"] / total_regions
            result["average_travel_cost"] = (
                per_region.groupby(selected_site_col)[cost_col]
                .mean()
                .reindex(selected_sites)
            )

        return result

    def two_step_floating_catchment(
        self,
        supply_col,
        catchment_size,
        rank_on=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
        per_capita=1,
        return_site_ratios=False,
    ):
        """
        Two-step floating catchment area (2SFCA) accessibility for one
        chosen solution.

        Resolves the solution's selected sites and delegates to
        `SiteProblem.two_step_floating_catchment()`, which does the actual
        step-1/step-2 calculation. Unlike `site_allocation_summary()`, this
        cannot use `problem_df` (which only carries each region's nearest
        site and cost, not the full per-site travel-cost row 2SFCA needs
        for every site's catchment) so it goes back to the problem's
        travel frame instead.

        Parameters
        ----------
        supply_col : str
            Column in `candidate_sites` holding each site's supply
            quantity (e.g. number of GPs, beds, weekly appointment slots).
        catchment_size : float
            Hard catchment threshold d0, in the travel matrix's registered
            units. A site is "in catchment" for a demand region if the
            travel cost between them is `<= catchment_size`.
        rank_on : str, optional
        solution_rank : int, default 1
        site_names : list, optional
        site_indices : list, optional
            Solution selection, as in `site_allocation_summary`. Priority
            is site_indices > site_names > rank_on/solution_rank.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Scores accessibility under
            that matrix's travel costs instead of the primary matrix's.
        per_capita : float, default 1
            Multiplier applied to the `accessibility` column, e.g. 1_000
            to express supply per 1,000 head instead of raw supply units
            per head.
        return_site_ratios : bool, default False
            If True, also return the step-1 per-site table.

        Returns
        -------
        pandas.DataFrame
            Per demand region, indexed by demand location ID: `accessibility`
            (supply units per head, x `per_capita`), `n_sites_in_catchment`,
            `demand`.
        pandas.DataFrame
            Only if `return_site_ratios=True`. Per site, indexed by site
            name: `supply`, `catchment_demand`, `n_regions_in_catchment`,
            `ratio`.

        See Also
        --------
        SiteProblem.two_step_floating_catchment : The underlying
            calculation, usable directly on the full candidate site set
            without needing a solved solution.
        """
        solution = _select_solution(
            self.solution_df,
            rank_on=rank_on,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
        )
        selected_sites = list(solution["site_names"].iloc[0])

        return self.site_problem.two_step_floating_catchment(
            supply_col=supply_col,
            catchment_size=catchment_size,
            site_names=selected_sites,
            matrix=matrix,
            per_capita=per_capita,
            return_site_ratios=return_site_ratios,
        )

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
