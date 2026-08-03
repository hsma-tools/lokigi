import warnings

import numpy as np
import pandas as pd

from lokigi.utils import (
    _min_max_normalize,
    _select_solution,
    _sort_solutions_by_metric,
    _population_impact_metrics,
    _split_bins_into_tertiles,
    _format_threshold,
    _get_required_site_indices,
)

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
from lokigi.mixins.site_accessibility import AccessibilityPlotMixin


# Reserved key `SiteProblem._resolve_baseline_costs()` uses to smuggle the
# baseline's `site_names` through `baseline_costs` (nominally a
# `dict[str, pandas.Series]` of cost columns) to `EvaluatedCombination.
# __init__`, which pops it out before the cost-Series validation loop ever
# sees it. Avoids threading a second `baseline_site_names` parameter through
# every brute-force/greedy/GRASP call site that already passes
# `baseline_costs` through unopened -- this key is never seen by any of
# them, only by the two places that actually construct/consume the dict.
# Dunder-style so it can never collide with a real cost-column name
# ("min_cost", "min_cost__<label>").
_BASELINE_SITE_NAMES_KEY = "__baseline_site_names__"


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
    baseline_costs : dict[str, pandas.Series], optional
        Maps a cost-column name ("min_cost", or "min_cost__<label>" for a
        registered secondary travel matrix) to a baseline per-region cost
        Series, indexed by demand-location ID
        (`site_problem._demand_data_id_col`). When given, `population_impact`
        (and the corresponding `demand_improved`/`regions_improved`/etc.
        keys in `return_solution_metrics()`) diff this combination's own
        costs against it. `None` (the default) skips this entirely.
    meaningful_change_threshold : float, default 0.0
        Passed through to `_population_impact_metrics()` -- see there for
        the improved/worsened/unchanged classification rule.
    beyond_thresholds : float or sequence of float, optional
        One or more "left behind" travel-cost thresholds. For each value
        `t`, adds `demand_beyond_threshold_<t>` / `regions_beyond_
        threshold_<t>` (headcount/region-count with a cost beyond `t`) to
        `return_solution_metrics()`, plus a `_by_equity_group` dict variant
        of each when equity data is registered. Deliberately a distinct
        parameter from `coverage_threshold`/`threshold_for_coverage`
        (opposite crossing direction: "covered" is good, "beyond" is bad)
        and, unlike it, supports more than one threshold at once. `None`
        (the default) adds no columns, keeping `solution_df`'s schema
        unchanged for callers that never ask for this.

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
    population_impact : dict or None
        `_population_impact_metrics()`'s output diffing this combination's
        `min_cost` against `baseline_costs["min_cost"]`, or `None` if no
        `baseline_costs` was supplied.



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
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
    ):
        self.solution_type = solution_type
        self.site_names = site_names
        self.site_indices = site_indices
        self.evaluated_combination_df = evaluated_combination_df
        self.site_problem = site_problem
        self._meaningful_change_threshold = meaningful_change_threshold

        if beyond_thresholds is None:
            self.beyond_thresholds = ()
        elif isinstance(beyond_thresholds, (int, float, np.integer, np.floating)):
            # np.integer doesn't subclass int (unlike np.floating, which
            # does subclass float) -- without it, a numpy-integer scalar
            # (e.g. from a column's .max()) fell into the `else` branch and
            # crashed with "'numpy.int64' object is not iterable".
            self.beyond_thresholds = (float(beyond_thresholds),)
        else:
            self.beyond_thresholds = tuple(float(t) for t in beyond_thresholds)

        # Population-impact-vs-baseline support (see _population_impact_
        # metrics in utils.py). `baseline_costs` maps a cost-column name
        # ("min_cost", or "min_cost__<label>" for a secondary travel
        # matrix) to a baseline pd.Series indexed by demand-location ID
        # (site_problem._demand_data_id_col). Aligned once here, by ID --
        # NEVER by position -- and reused by every _compute_travel_metrics
        # call below (primary matrix, each secondary matrix, each
        # secondary demand scenario), since they all key off the same
        # cost-column names.
        # Popped out before the cost-Series validation loop below ever sees
        # it -- see _BASELINE_SITE_NAMES_KEY's own docstring. None if
        # baseline_costs didn't carry it (e.g. a direct, advanced caller of
        # evaluate_single_solution_single_objective(baseline_costs=...) that
        # built the dict by hand rather than via solve(baseline=...)).
        self._baseline_site_names = None
        if baseline_costs is not None:
            baseline_costs = dict(baseline_costs)
            self._baseline_site_names = baseline_costs.pop(
                _BASELINE_SITE_NAMES_KEY, None
            )

        self._baseline_cost_arrays = {}
        if baseline_costs is not None:
            id_col = getattr(site_problem, "_demand_data_id_col", None)
            if id_col is None or id_col not in evaluated_combination_df.columns:
                raise ValueError(
                    "baseline_costs was supplied, but this combination's "
                    "problem_df has no demand-location ID column "
                    f"({id_col!r}) to align against. This should not be "
                    "reachable through the public API."
                )
            region_ids = evaluated_combination_df[id_col]
            for cost_col, baseline_series in baseline_costs.items():
                missing_mask = ~region_ids.isin(baseline_series.index)
                if missing_mask.any():
                    missing_ids = region_ids[missing_mask].tolist()
                    raise ValueError(
                        f"Baseline is missing {len(missing_ids)} demand "
                        f"location(s) present in this solution's problem_df "
                        f"for '{cost_col}': {missing_ids[:10]}"
                        f"{', ...' if len(missing_ids) > 10 else ''}. The "
                        "baseline and candidate solutions must be evaluated "
                        "against the exact same demand locations."
                    )
                self._baseline_cost_arrays[cost_col] = baseline_series.reindex(
                    region_ids
                ).to_numpy()

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

                elif label in getattr(
                    self.site_problem, "secondary_demand_matrices", {}
                ):
                    # A registered secondary demand scenario: its column
                    # was merged into evaluated_combination_df as
                    # `demand__<label>` (see evaluate_single_solution_
                    # single_objective). 'demand' defaults to
                    # higher_better for the same reason the primary demand
                    # column does above.
                    col_name = f"demand__{label}"
                    direction = "higher_better"

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
            "min_cost",
            "within_threshold",
            active_weights,
            unit=getattr(self.site_problem, "_travel_matrix_unit", None),
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
        self.demand_within_coverage_threshold = primary_metrics[
            "demand_within_coverage_threshold"
        ]
        self.regions_within_coverage_threshold = primary_metrics[
            "regions_within_coverage_threshold"
        ]
        self.beyond_threshold_metrics = primary_metrics["beyond_threshold_metrics"]
        self.weighted_by_equity_group = primary_metrics["weighted_by_equity_group"]
        self.unweighted_by_equity_group = primary_metrics["unweighted_by_equity_group"]
        self.coverage_by_equity_group = primary_metrics["coverage_by_equity_group"]
        self.coverage_regions_by_equity_group = primary_metrics[
            "coverage_regions_by_equity_group"
        ]
        self.max_cost_by_equity_group = primary_metrics["max_cost_by_equity_group"]
        self.demand_improved_by_equity_group = primary_metrics[
            "demand_improved_by_equity_group"
        ]
        self.demand_worsened_by_equity_group = primary_metrics[
            "demand_worsened_by_equity_group"
        ]
        self.gap_absolute_weighted = primary_metrics["gap_absolute_weighted"]
        self.gap_absolute_desc = primary_metrics["gap_absolute_desc"]
        self.gap_relative_weighted = primary_metrics["gap_relative_weighted"]
        self.gap_relative_desc = primary_metrics["gap_relative_desc"]
        self.avg_lower_third_bins = primary_metrics["avg_lower_third_bins"]
        self.avg_middle_third_bins = primary_metrics["avg_middle_third_bins"]
        self.avg_upper_third_bins = primary_metrics["avg_upper_third_bins"]
        self.inter_tertile_ratio = primary_metrics["inter_tertile_ratio"]
        self.inter_tertile_desc = primary_metrics["inter_tertile_desc"]
        self.population_impact = primary_metrics["population_impact"]

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
                cost_col,
                within_col,
                active_weights,
                unit=self.site_problem.secondary_travel_matrices[label]["unit"],
            )

        # Secondary demand scenarios: re-weight the primary matrix's
        # min_cost/within_threshold (and, opt-in via also_weight_matrices,
        # any secondary travel matrix's) using the scenario's own demand
        # column, keyed by (secondary travel label or None) -> metrics.
        # Always weighted by the scenario's *pure* demand -- never the
        # compound `weights=` blend -- so a demand-scenario column is a
        # stable, single-scenario view regardless of how the objective is
        # weighted.
        self.secondary_demand_metrics = {}
        for dlabel, dmeta in getattr(
            self.site_problem, "secondary_demand_matrices", {}
        ).items():
            demand_col = f"demand__{dlabel}"
            if demand_col not in self.evaluated_combination_df.columns:
                continue
            dweights = self.evaluated_combination_df[demand_col]

            per_matrix = {
                None: self._compute_travel_metrics(
                    "min_cost",
                    "within_threshold",
                    active_weights=dweights,
                    coverage_demand=dweights,
                    unit=getattr(self.site_problem, "_travel_matrix_unit", None),
                )
            }
            for tlabel in dmeta["also_weight_matrices"]:
                t_cost_col = f"min_cost__{tlabel}"
                if t_cost_col not in self.evaluated_combination_df.columns:
                    continue
                per_matrix[tlabel] = self._compute_travel_metrics(
                    t_cost_col,
                    f"within_threshold__{tlabel}",
                    active_weights=dweights,
                    coverage_demand=dweights,
                    unit=self.site_problem.secondary_travel_matrices[tlabel]["unit"],
                )
            self.secondary_demand_metrics[dlabel] = per_matrix

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

        Thin wrapper around `_coverage_stats()` for the many call sites
        that only need the proportion (e.g. the per-equity-band
        breakdowns), keeping their signature unchanged.
        """
        proportion, _, _ = EvaluatedCombination._coverage_stats(within_flags, demand)
        return proportion

    @staticmethod
    def _coverage_stats(within_flags, demand):
        """
        Proportion, absolute demand, and region count covered --
        demand-weighted when `demand` is supplied, otherwise the plain
        share/count of regions.

        The all-NaN check is load-bearing. `within_flags` is entirely NaN
        whenever no `threshold_for_coverage` was given, and that must stay
        NaN ("not measured"). It cannot be left to the arithmetic below,
        because pandas' `.sum()` skips NaN: the weighted branch would
        otherwise return a confident 0.0 -- "none of the demand is covered"
        -- for a problem where coverage was never assessed at all.

        Returns
        -------
        (proportion, covered_demand, covered_regions) : (float, float, int)
            `covered_demand` is `NaN` whenever `demand` is None (there is no
            headcount to report, only a share of regions); `proportion` and
            `covered_demand` are both `NaN` if `within_flags` is all-NaN.
        """
        if within_flags.isna().all():
            return np.nan, np.nan, 0

        covered = within_flags.astype(float)
        covered_regions = int(covered.sum())

        if demand is None:
            return float(covered.sum() / len(covered)), np.nan, covered_regions

        total_demand = demand.sum()
        if total_demand <= 0:
            # Defensive only, and not reachable through the public API: a
            # zero-demand problem already raises "Weights sum to zero" from
            # np.average when weighted_average is computed further up.
            return np.nan, np.nan, covered_regions

        covered_demand = float((covered * demand).sum())
        return covered_demand / total_demand, covered_demand, covered_regions

    def _compute_travel_metrics(
        self, cost_col, within_col, active_weights, coverage_demand=None, unit=None
    ):
        """
        Compute weighted/unweighted travel-cost summary statistics and the
        equity breakdown for one travel matrix's min-cost column.

        Used once for the primary matrix (`cost_col="min_cost"`), once per
        registered secondary travel matrix (`cost_col="min_cost__<label>"`),
        and once per registered secondary demand scenario, so every
        combination gets identical treatment rather than a parallel,
        potentially-diverging implementation.

        `unit` names the travel-cost unit this `cost_col` was registered
        with (`add_travel_matrix(unit=...)`/`add_secondary_travel_matrix(
        unit=...)`), used only in `gap_absolute_desc`'s wording ("Spread of
        2.6 minutes" rather than the meaningless bare "Spread of 2.6
        units"). Falls back to the literal word "units" if `None` -- e.g.
        no unit was ever registered -- so the sentence still reads.

        `coverage_demand` overrides the demand series used to weight
        `proportion_within_coverage_threshold` (and its equity breakdown).
        It defaults to the primary demand column (via
        `_coverage_demand_series()`), which is correct for the primary
        matrix and every secondary *travel* matrix -- both are always
        weighted by the primary demand. A secondary *demand* scenario
        passes its own scenario series here instead, so "proportion of
        demand covered" reflects that scenario's demand rather than the
        primary one. It also determines the weighting for
        `population_impact` (see below), for the same reason.

        The returned dict also carries a `population_impact` key -- `None`
        unless a baseline was supplied for `cost_col` (i.e.
        `cost_col in self._baseline_cost_arrays`), in which case it is the
        `_population_impact_metrics()` dict diffing `df[cost_col]` against
        that baseline, weighted by `demand_series` above.

        It also carries a `beyond_threshold_metrics` dict, keyed by each
        value in `self.beyond_thresholds` -- empty unless that was
        populated (see `EvaluatedCombination`'s `beyond_thresholds`
        parameter).
        """
        df = self.evaluated_combination_df

        weighted_average = np.average(df[cost_col], weights=active_weights)
        unweighted_average = np.average(df[cost_col])
        percentile_90th = np.percentile(df[cost_col], q=90)
        max_cost = np.max(df[cost_col])

        demand_series = (
            coverage_demand if coverage_demand is not None else self._coverage_demand_series()
        )
        (
            proportion_within_coverage_threshold,
            demand_within_coverage_threshold,
            regions_within_coverage_threshold,
        ) = self._coverage_stats(df[within_col], demand_series)
        proportion_regions_within_coverage_threshold = self._coverage_proportion(
            df[within_col], None
        )

        # "Left behind" headcounts: how many people/regions have a travel
        # cost beyond each threshold in self.beyond_thresholds, computed
        # directly from cost_col rather than the single pre-baked
        # within_col -- unlike coverage, this supports several thresholds
        # at once. NaN (unreachable) cost counts as beyond every
        # threshold, matching within_col's "NaN cost -> not covered"
        # convention in direction (both treat unreachable as the bad
        # outcome).
        beyond_threshold_metrics = {}
        beyond_flags_by_threshold = {
            t: ~(df[cost_col] <= t) for t in self.beyond_thresholds
        }
        for t, beyond_flags in beyond_flags_by_threshold.items():
            beyond_threshold_metrics[t] = {
                "regions_beyond": int(beyond_flags.sum()),
                "demand_beyond": (
                    np.nan
                    if demand_series is None
                    else float((beyond_flags.astype(float) * demand_series).sum())
                ),
                "regions_beyond_by_equity_group": {},
                "demand_beyond_by_equity_group": {},
            }

        weighted_by_equity_group = {}
        unweighted_by_equity_group = {}
        coverage_by_equity_group = {}
        coverage_regions_by_equity_group = {}
        max_cost_by_equity_group = {}
        demand_improved_by_equity_group = {}
        demand_worsened_by_equity_group = {}
        gap_absolute_weighted = None
        gap_absolute_desc = "N/A (No equity data)"
        gap_relative_weighted = None
        gap_relative_desc = "N/A (No equity data)"
        avg_lower_third_bins = None
        avg_middle_third_bins = None
        avg_upper_third_bins = None
        inter_tertile_ratio = None
        inter_tertile_desc = "N/A (No equity data)"

        # Resolved here (rather than just before the return statement, as
        # in earlier versions of this method) so the equity-group breakdown
        # below can also compute a per-band population impact.
        baseline_array = self._baseline_cost_arrays.get(cost_col)

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
                unit_word = unit if unit else "units"
                gap_absolute_desc = (
                    f"Spread of {gap_absolute_weighted:.1f} {unit_word} "
                    "between best and worst groups"
                )

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

            # 5b. Left-behind headcounts by group -- who's beyond each
            # threshold, split by equity band (e.g. "how many of the
            # people beyond 45 minutes are in the most deprived third?").
            for t, beyond_flags in beyond_flags_by_threshold.items():
                beyond_threshold_metrics[t]["regions_beyond_by_equity_group"] = {
                    band: int(beyond_flags.loc[group.index].sum())
                    for band, group in grouped_df
                }
                if demand_series is not None:
                    beyond_threshold_metrics[t]["demand_beyond_by_equity_group"] = {
                        band: float(
                            (
                                beyond_flags.loc[group.index].astype(float)
                                * demand_series.loc[group.index]
                            ).sum()
                        )
                        for band, group in grouped_df
                    }

            # 5c. Population impact by group -- who actually benefited from
            # this solution vs the baseline, split by equity band (e.g.
            # "are the most deprived areas improving at a higher rate than
            # the least deprived?"). Only computed when a baseline was
            # supplied for this cost_col. `df.index.get_indexer` converts
            # each group's (label) index into positions, since
            # `baseline_array` and `df[cost_col].to_numpy()` are positional
            # arrays aligned to `df`'s row order, not necessarily identical
            # to its index labels.
            if baseline_array is not None:
                cost_array = df[cost_col].to_numpy()
                for band, group in grouped_df:
                    positions = df.index.get_indexer(group.index)
                    group_demand = (
                        None
                        if demand_series is None
                        else demand_series.loc[group.index].to_numpy()
                    )
                    band_impact = _population_impact_metrics(
                        current=cost_array[positions],
                        baseline=baseline_array[positions],
                        demand=group_demand,
                        meaningful_change_threshold=self._meaningful_change_threshold,
                    )
                    demand_improved_by_equity_group[band] = band_impact[
                        "demand_improved"
                    ]
                    demand_worsened_by_equity_group[band] = band_impact[
                        "demand_worsened"
                    ]

            # 6. Tertile Groupings (Averaging the bin results into thirds)
            # Sorts the bins (e.g., 1-10) and splits them into 3 roughly
            # equal chunks, ordered most- to least-disadvantaged per
            # `_equity_data_disadvantaged_end` (defaults to "low" --
            # lower bins = more deprived -- when unset, matching every
            # pre-existing caller's assumption).
            unique_bins = sorted(list(weighted_by_equity_group.keys()))
            disadvantaged_end = getattr(
                self.site_problem, "_equity_data_disadvantaged_end", None
            )
            lower_chunk, middle_chunk, upper_chunk = _split_bins_into_tertiles(
                unique_bins, disadvantaged_end
            )
            if lower_chunk is not None:
                chunks = (lower_chunk, middle_chunk, upper_chunk)
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

                    # Generate Inter-Tertile Ratio Descriptor. "Most
                    # deprived" always refers to `chunks[0]` above, which
                    # `_split_bins_into_tertiles` has already ordered per
                    # `disadvantaged_end` -- so this wording is correct
                    # regardless of which end of the raw bin scale that is.
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

        population_impact = (
            None
            if baseline_array is None
            else _population_impact_metrics(
                current=df[cost_col].to_numpy(),
                baseline=baseline_array,
                demand=None if demand_series is None else demand_series.to_numpy(),
                meaningful_change_threshold=self._meaningful_change_threshold,
            )
        )

        return {
            "weighted_average": weighted_average,
            "unweighted_average": unweighted_average,
            "percentile_90th": percentile_90th,
            "max": max_cost,
            "proportion_within_coverage_threshold": proportion_within_coverage_threshold,
            "proportion_regions_within_coverage_threshold": proportion_regions_within_coverage_threshold,
            "demand_within_coverage_threshold": demand_within_coverage_threshold,
            "regions_within_coverage_threshold": regions_within_coverage_threshold,
            "beyond_threshold_metrics": beyond_threshold_metrics,
            "weighted_by_equity_group": weighted_by_equity_group,
            "unweighted_by_equity_group": unweighted_by_equity_group,
            "coverage_by_equity_group": coverage_by_equity_group,
            "coverage_regions_by_equity_group": coverage_regions_by_equity_group,
            "max_cost_by_equity_group": max_cost_by_equity_group,
            "demand_improved_by_equity_group": demand_improved_by_equity_group,
            "demand_worsened_by_equity_group": demand_worsened_by_equity_group,
            "gap_absolute_weighted": gap_absolute_weighted,
            "gap_absolute_desc": gap_absolute_desc,
            "gap_relative_weighted": gap_relative_weighted,
            "gap_relative_desc": gap_relative_desc,
            "avg_lower_third_bins": avg_lower_third_bins,
            "avg_middle_third_bins": avg_middle_third_bins,
            "avg_upper_third_bins": avg_upper_third_bins,
            "inter_tertile_ratio": inter_tertile_ratio,
            "inter_tertile_desc": inter_tertile_desc,
            "population_impact": population_impact,
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

        Notes
        -----
        INTERPRETATION GUIDE FOR SUMMARY TABLES & SORTING:

        1a. 'weighted_average'
            LOWER is better. Represents travel time adjusted for specified weighting factors.

        1b. Travel Costs ('unweighted_average', '90th_percentile', 'max'):
            LOWER is better. Represents travel time or distance.

        1c. 'total_cost'
            LOWER is better. Total fixed cost of the selected sites (sum of
            the `cost_col` configured via `add_sites()`). `NaN` if no
            `cost_col` was configured. Only influences which solution is
            selected if explicitly passed as a weight (weights={"cost": ...}).

        2. Absolute Equity Gap ('gap_absolute_weighted'):
            CLOSER TO 0 is better. Measures the flat minute/distance difference
            between the best-served and worst-served equity bands. High numbers
            mean severe geographical disparity.

        3. Relative Equity Gap ('gap_relative_weighted'):
            CLOSER TO 1.0 is better. If it's 1.5, the worst-served group travels
            1.5x longer than the best-served group.

        4. Inter-Tertile Ratio ('inter_tertile_ratio'):
            Measures macro-equity between the most- and least-disadvantaged
            equity-band tertiles, per `add_equity_data(disadvantaged_end=...)`
            -- lower bins = more disadvantaged (e.g. IMD 1-3) when
            `disadvantaged_end="low"` or unset; the reverse when `"high"`.
            SORTING CRITERIA:

            * ITR > 1.0: Inequity. The most deprived third faces longer travel times
              than the least deprived third (e.g., 1.25 = 25% longer travel).
            * ITR = 1.0: Perfect equality in macro travel times.
            * ITR < 1.0: Progressive equity. Travel times are shorter for the most
              deprived communities.

        5. Coverage Metrics ('proportion_within_coverage_threshold', 'coverage_by_equity_group'):
            HIGHER is better (Scale: 0.0 to 1.0). Represents accessibility. Look for
            solutions where coverage is both globally high and uniformly distributed
            across groups. Both are `NaN` if no `threshold_for_coverage` was given.

            NAMING RULE: an unqualified coverage metric is weighted by the demand
            registered via `add_demand()`, so it answers "what share of *people*
            are covered". The `regions` variants
            ('proportion_regions_within_coverage_threshold',
            'coverage_regions_by_equity_group') count every region equally and
            answer "what share of *places* are covered". The two coincide when
            demand is uniform, including when `add_demand()` was never called.

            The demand-weighted figure is the one 'mclp' optimises, matching the
            textbook Maximal Covering Location Problem.

        5b. Absolute coverage headcounts ('demand_within_coverage_threshold',
            'regions_within_coverage_threshold'):
            HIGHER is better. The literal headcount/region-count behind the
            two coverage proportions above -- e.g. "391,823 people" rather
            than "75.4%" -- for reporting to audiences who find an absolute
            number more legible than a percentage. `NaN`/`0` under the same
            conditions as the proportions.

        6. Secondary travel matrices (columns suffixed `__<label>`, e.g.
           'weighted_average__public_transport'):

            Registered via `add_secondary_travel_matrix(label=...)`. Same metrics
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

        7. Population impact vs baseline (`demand_improved`,
           `demand_worsened`, `demand_unchanged`, `proportion_demand_
           improved`, `proportion_demand_worsened`, `regions_improved`,
           `regions_worsened`, `regions_unchanged`,
           `mean_reduction_among_improved`, `mean_increase_among_worsened`,
           `max_reduction`, `max_increase`):

            Only present when a baseline was supplied (`solve(baseline=...)`,
            or a baseline was passed to `evaluate_single_solution_single_
            objective()` directly) -- absent otherwise, not `NaN`, so
            `solution_df`'s schema is unchanged for callers that never ask
            for a baseline. `demand_*`/`mean_*` are HIGHER-is-better for the
            `_improved`/`reduction` names (more people better off, or a
            bigger improvement) and LOWER-is-better for the `_worsened`/
            `increase` names; `*_unchanged` is directionless. See
            `SolutionComparator.population_impact_summary()` for the
            equivalent baseline-vs-candidate comparison without solving via
            `solve(baseline=...)`. Secondary travel matrices and secondary
            demand scenarios get only the demand-weighted subset
            (`demand_improved__<label>`, `demand_worsened__<label>`,
            `demand_unchanged__<label>`, `mean_reduction_among_improved__
            <label>`, `mean_increase_among_worsened__<label>`) by default;
            `full_secondary_metrics=True` also adds the region counts and
            max change (`regions_improved__<label>`, etc.) for secondary
            travel matrices. Demand scenarios never get the region-count/max
            variants (region membership doesn't vary with demand), matching
            point 6's "only what varies with demand" rule.

            When equity data is also registered, `demand_improved_by_equity_
            group` / `demand_worsened_by_equity_group` (dict per band) are
            also added -- e.g. "are the most deprived areas improving at a
            higher rate than the least deprived?". Primary matrix only;
            not yet broken down for secondary travel matrices/demand
            scenarios.

        7b. Which sites actually changed (`sites_closed_vs_baseline`,
            `sites_added_vs_baseline`):

            `sites_closed_vs_baseline` is the baseline's own site names
            absent from this solution's `site_names`; `sites_added_vs_
            baseline` is `site_names` absent from the baseline's. Answers
            "which sites changed?" directly, rather than requiring a caller
            to compute the set difference themselves -- useful when several
            solutions share nearly all the same sites and only differ in
            one or two, which a bare `site_names` column doesn't make
            obvious at a glance. Present under the same condition as the
            rest of this point (a baseline was supplied); absent, not empty
            lists, otherwise.

        8. "Left behind" headcounts (`demand_beyond_threshold_<t>`,
           `regions_beyond_threshold_<t>`, one pair per threshold `t` in
           `beyond_thresholds`, plus a `_by_equity_group` dict variant of
           each when equity data is registered):

            LOWER is better -- the opposite direction from the coverage
            metrics in point 5, since these count people/regions a
            threshold crossing leaves *outside* rather than within. A
            distinct concept from `threshold_for_coverage`/'coverage
            metrics' even though both compare a travel cost against a
            cutoff: "covered" (good, one threshold) vs "beyond" (bad, any
            number of thresholds at once). Thresholds are independent, not
            cumulative -- `demand_beyond_threshold_45` is not a subset of
            `demand_beyond_threshold_30`'s complement; each is a fresh
            count against its own cutoff. Only present when
            `beyond_thresholds` was supplied -- absent, not `NaN`,
            otherwise. Secondary travel matrices get the demand-weighted
            headcount by default (`demand_beyond_threshold_<t>__<label>`);
            `full_secondary_metrics=True` also adds region counts and the
            equity breakdowns, matching point 6/7's pattern. Secondary
            demand scenarios get only the demand-weighted headcount, never
            region counts or equity breakdowns, matching point 6's "only
            what varies with demand" rule.

        9. Additional sites chosen beyond required (`additional_site_names`):

            `site_names` with the required sites (flagged via
            `add_sites(required_sites_col=...)`) removed -- e.g. for a "we
            have 4 sites and are opening 1 more" problem, this is just the
            new site, rather than all 5 names, which otherwise look
            identical across every solution that all correctly keep the
            same 4 required sites. Only present when at least one required
            site is configured; absent, not an empty list, otherwise.

        10. Sites not selected (`unselected_site_names`):

            Every registered candidate site absent from `site_names`, in
            canonical site-index order -- the complement of `site_names`
            among ALL candidates, not just the required ones point 9
            removes. Always present (unlike point 9), since it doesn't
            depend on any optional registered data.
        """

        # getattr-guarded: a minimal stub site_problem (as used by some
        # tests to isolate a single metric) may not define candidate_sites
        # at all -- treated the same as "nothing to report" rather than
        # raising, matching _get_required_site_indices's own tolerance.
        candidate_sites = getattr(self.site_problem, "candidate_sites", None)
        id_col = getattr(self.site_problem, "_candidate_sites_candidate_id_col", None)
        if candidate_sites is not None and id_col is not None:
            all_sites_in_order = candidate_sites.sort_values("canonical_site_index")[
                id_col
            ].tolist()
            selected = set(self.site_names)
            unselected_site_names = [
                name for name in all_sites_in_order if name not in selected
            ]
        else:
            unselected_site_names = []

        # Return weighted average
        metrics = {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "unselected_site_names": unselected_site_names,
            "coverage_threshold": self.coverage_threshold,
            "weighted_average": self.weighted_average,
            "unweighted_average": self.unweighted_average,
            "90th_percentile": self.percentile_90th,
            "max": self.max,
            "total_cost": self.total_cost,
            "proportion_within_coverage_threshold": self.proportion_within_coverage_threshold,
            "proportion_regions_within_coverage_threshold": self.proportion_regions_within_coverage_threshold,
            "demand_within_coverage_threshold": self.demand_within_coverage_threshold,
            "regions_within_coverage_threshold": self.regions_within_coverage_threshold,
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

        # Additional sites chosen beyond required: only present when at
        # least one required site is configured (see point 9 above), so
        # solution_df's schema is unchanged for every caller that never
        # uses required_sites_col.
        required_indices = _get_required_site_indices(self.site_problem)
        if required_indices:
            candidate_sites = self.site_problem.candidate_sites
            id_col = self.site_problem._candidate_sites_candidate_id_col
            required_names = set(
                candidate_sites.loc[
                    candidate_sites["canonical_site_index"].isin(required_indices),
                    id_col,
                ]
            )
            metrics["additional_site_names"] = [
                name for name in self.site_names if name not in required_names
            ]

        # Population impact vs baseline: only present when a baseline was
        # actually supplied (self.population_impact is None otherwise), so
        # solution_df's schema is unchanged for every caller that doesn't
        # ask for one. See point 7 in the docstring above.
        if self.population_impact is not None:
            pi = self.population_impact
            metrics["demand_improved"] = pi["demand_improved"]
            metrics["demand_worsened"] = pi["demand_worsened"]
            metrics["demand_unchanged"] = pi["demand_unchanged"]
            metrics["proportion_demand_improved"] = pi["proportion_demand_improved"]
            metrics["proportion_demand_worsened"] = pi["proportion_demand_worsened"]
            metrics["regions_improved"] = pi["regions_improved"]
            metrics["regions_worsened"] = pi["regions_worsened"]
            metrics["regions_unchanged"] = pi["regions_unchanged"]
            metrics["mean_reduction_among_improved"] = pi[
                "mean_reduction_among_improved"
            ]
            metrics["mean_increase_among_worsened"] = pi[
                "mean_increase_among_worsened"
            ]
            metrics["max_reduction"] = pi["max_reduction"]
            metrics["max_increase"] = pi["max_increase"]
            if self.demand_improved_by_equity_group:
                metrics["demand_improved_by_equity_group"] = (
                    self.demand_improved_by_equity_group
                )
                metrics["demand_worsened_by_equity_group"] = (
                    self.demand_worsened_by_equity_group
                )

        # Which sites actually changed vs the baseline -- see point 7b in
        # the docstring above. Gated independently of population_impact
        # (rather than reusing that same `if`): both are normally present
        # together, since both come from the same solve(baseline=...) call,
        # but an advanced direct caller of evaluate_single_solution_single_
        # objective(baseline_costs=...) could supply cost Series without
        # the site-names sentinel, and these two should reflect exactly
        # what was actually provided.
        if self._baseline_site_names is not None:
            current_names = set(self.site_names)
            baseline_names = set(self._baseline_site_names)
            metrics["sites_closed_vs_baseline"] = [
                name for name in self._baseline_site_names if name not in current_names
            ]
            metrics["sites_added_vs_baseline"] = [
                name for name in self.site_names if name not in baseline_names
            ]

        # "Left behind" headcounts: only present when beyond_thresholds was
        # supplied (self.beyond_threshold_metrics is empty otherwise), so
        # solution_df's schema is unchanged for callers that never ask for
        # this. One demand/regions pair per threshold, plus a
        # `_by_equity_group` dict variant of each whenever equity data is
        # registered (empty dict otherwise, so no column is added).
        for t, bt in self.beyond_threshold_metrics.items():
            t_suffix = _format_threshold(t)
            metrics[f"demand_beyond_threshold_{t_suffix}"] = bt["demand_beyond"]
            metrics[f"regions_beyond_threshold_{t_suffix}"] = bt["regions_beyond"]
            if bt["demand_beyond_by_equity_group"]:
                metrics[f"demand_beyond_threshold_{t_suffix}_by_equity_group"] = bt[
                    "demand_beyond_by_equity_group"
                ]
            if bt["regions_beyond_by_equity_group"]:
                metrics[f"regions_beyond_threshold_{t_suffix}_by_equity_group"] = bt[
                    "regions_beyond_by_equity_group"
                ]

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
            metrics[f"demand_within_coverage_threshold__{label}"] = secondary[
                "demand_within_coverage_threshold"
            ]
            metrics[f"regions_within_coverage_threshold__{label}"] = secondary[
                "regions_within_coverage_threshold"
            ]
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

            secondary_pi = secondary.get("population_impact")
            if secondary_pi is not None:
                metrics[f"demand_improved__{label}"] = secondary_pi["demand_improved"]
                metrics[f"demand_worsened__{label}"] = secondary_pi["demand_worsened"]
                metrics[f"demand_unchanged__{label}"] = secondary_pi[
                    "demand_unchanged"
                ]
                metrics[f"mean_reduction_among_improved__{label}"] = secondary_pi[
                    "mean_reduction_among_improved"
                ]
                metrics[f"mean_increase_among_worsened__{label}"] = secondary_pi[
                    "mean_increase_among_worsened"
                ]
                if full_secondary_metrics:
                    metrics[f"regions_improved__{label}"] = secondary_pi[
                        "regions_improved"
                    ]
                    metrics[f"regions_worsened__{label}"] = secondary_pi[
                        "regions_worsened"
                    ]
                    metrics[f"regions_unchanged__{label}"] = secondary_pi[
                        "regions_unchanged"
                    ]
                    metrics[f"max_reduction__{label}"] = secondary_pi["max_reduction"]
                    metrics[f"max_increase__{label}"] = secondary_pi["max_increase"]

            for t, bt in secondary.get("beyond_threshold_metrics", {}).items():
                t_suffix = _format_threshold(t)
                metrics[f"demand_beyond_threshold_{t_suffix}__{label}"] = bt[
                    "demand_beyond"
                ]
                if full_secondary_metrics:
                    metrics[f"regions_beyond_threshold_{t_suffix}__{label}"] = bt[
                        "regions_beyond"
                    ]
                    if bt["demand_beyond_by_equity_group"]:
                        metrics[
                            f"demand_beyond_threshold_{t_suffix}_by_equity_group__{label}"
                        ] = bt["demand_beyond_by_equity_group"]
                    if bt["regions_beyond_by_equity_group"]:
                        metrics[
                            f"regions_beyond_threshold_{t_suffix}_by_equity_group__{label}"
                        ] = bt["regions_beyond_by_equity_group"]

        # Secondary demand scenarios: only the two metrics that actually
        # vary with demand (weighted_average, proportion_within_coverage_
        # threshold -- see _compute_travel_metrics' coverage_demand
        # docstring). Suffix order is travel-label-first, demand-label-
        # second (`weighted_average__<travel>__<demand>`) for the opt-in
        # cross with a secondary travel matrix registered via
        # also_weight_matrices; `weighted_average__<demand>` for the
        # (default) primary-travel case. Per-scenario equity breakdowns are
        # not yet supported and are never emitted here, regardless of
        # full_secondary_metrics.
        for dlabel, per_matrix in self.secondary_demand_metrics.items():
            for tlabel, dmetrics in per_matrix.items():
                suffix = dlabel if tlabel is None else f"{tlabel}__{dlabel}"
                metrics[f"weighted_average__{suffix}"] = dmetrics["weighted_average"]
                metrics[f"proportion_within_coverage_threshold__{suffix}"] = dmetrics[
                    "proportion_within_coverage_threshold"
                ]

                demand_pi = dmetrics.get("population_impact")
                if demand_pi is not None:
                    metrics[f"demand_improved__{suffix}"] = demand_pi[
                        "demand_improved"
                    ]
                    metrics[f"demand_worsened__{suffix}"] = demand_pi[
                        "demand_worsened"
                    ]
                    metrics[f"demand_unchanged__{suffix}"] = demand_pi[
                        "demand_unchanged"
                    ]
                    metrics[f"mean_reduction_among_improved__{suffix}"] = demand_pi[
                        "mean_reduction_among_improved"
                    ]
                    metrics[f"mean_increase_among_worsened__{suffix}"] = demand_pi[
                        "mean_increase_among_worsened"
                    ]

                for t, bt in dmetrics.get("beyond_threshold_metrics", {}).items():
                    t_suffix = _format_threshold(t)
                    metrics[f"demand_beyond_threshold_{t_suffix}__{suffix}"] = bt[
                        "demand_beyond"
                    ]

        return metrics


# Groups of `solution_df` base column names (before stripping any
# `__<label>` secondary-matrix/demand-scenario suffix), for
# `SiteSolutionSet.describe_solution_columns()`. Printed in this order.
_SOLUTION_COLUMN_GROUPS = [
    (
        "Which sites",
        "Which candidate sites make up this solution.",
        {
            "solution_rank",
            "site_names",
            "site_indices",
            "additional_site_names",
            "unselected_site_names",
        },
    ),
    (
        "Travel cost",
        "How far people have to travel under this solution (lower is better).",
        {"weighted_average", "unweighted_average", "90th_percentile", "max", "total_cost"},
    ),
    (
        "Coverage",
        "Share/headcount of people or places within threshold_for_coverage (higher is better).",
        {
            "coverage_threshold",
            "proportion_within_coverage_threshold",
            "proportion_regions_within_coverage_threshold",
            "demand_within_coverage_threshold",
            "regions_within_coverage_threshold",
        },
    ),
    (
        "Equity",
        "How unevenly travel cost is spread across the groups registered via add_equity_data().",
        {
            "gap_absolute_weighted",
            "gap_relative_weighted",
            "avg_lower_third_bins",
            "avg_middle_third_bins",
            "avg_upper_third_bins",
            "inter_tertile_ratio",
            "gap_absolute_description",
            "gap_relative_description",
            "inter_tertile_description",
            "weighted_by_equity_group",
            "unweighted_by_equity_group",
            "coverage_by_equity_group",
            "coverage_regions_by_equity_group",
            "max_cost_by_equity_group",
        },
    ),
    (
        "Change vs a baseline",
        "Per-person/per-region comparison against the baseline passed to solve(baseline=...) -- present only if one was supplied.",
        {
            "demand_improved",
            "demand_worsened",
            "demand_unchanged",
            "proportion_demand_improved",
            "proportion_demand_worsened",
            "regions_improved",
            "regions_worsened",
            "regions_unchanged",
            "mean_reduction_among_improved",
            "mean_increase_among_worsened",
            "max_reduction",
            "max_increase",
            "regions_newly_covered",
            "regions_newly_uncovered",
            "demand_newly_covered",
            "demand_newly_uncovered",
            "demand_improved_by_equity_group",
            "demand_worsened_by_equity_group",
            "sites_closed_vs_baseline",
            "sites_added_vs_baseline",
        },
    ),
    (
        "Underlying per-region data",
        "The full per-region breakdown behind every metric above.",
        {"problem_df"},
    ),
]


def _classify_solution_column(col):
    """
    Return the `_SOLUTION_COLUMN_GROUPS` category name for a `solution_df`
    column, or "Left behind (beyond a threshold)" for a
    `demand_beyond_threshold_<t>`/`regions_beyond_threshold_<t>` column
    (any suffix -- numeric threshold, `_by_equity_group`, `__<label>`), or
    "Other" if it doesn't match anything known (e.g. a future metric this
    grouping hasn't been updated for yet).

    Secondary travel-matrix/demand-scenario columns (`<base>__<label>`) are
    classified by their base name, same as the primary column.
    """
    if col.startswith("demand_beyond_threshold") or col.startswith(
        "regions_beyond_threshold"
    ):
        return "Left behind (beyond a threshold)"

    base = col.split("__", 1)[0]
    for name, _description, members in _SOLUTION_COLUMN_GROUPS:
        if base in members:
            return name
    return "Other"


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
    AccessibilityPlotMixin,
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

    def describe_solution_columns(self, return_dict=False):
        """
        Print (or return) `solution_df`'s columns grouped by what they
        mean, for a first look at a table that can otherwise run to
        30-40 columns (`show_solutions_colnames()` lists them, but flatly,
        with no indication of which ones are related or why they're there).

Groups whose columns are genuinely absent from
        `solution_df` (e.g. "Change vs a baseline" without `solve(baseline=
        ...)`) are omitted, as are "Coverage" and "Equity" when their
        columns are present but hold nothing but placeholder `NaN`/`None`/
        "N/A ..." values (those two are always part of `solution_df`'s
        schema, unlike the others, so column presence alone can't be used
        to decide whether to show them). Within a group, columns are
        listed in `solution_df`'s own column order. A group named "Other"
        is printed last if any column doesn't match a known category (e.g.
        a metric added after this method was last updated) -- see
        `return_solution_metrics()`'s docstring for what every column
        actually means.

        Parameters
        ----------
        return_dict : bool, default False
            If True, return `{group_name: [column_name, ...]}` instead of
            printing it.

        Returns
        -------
        dict or None
            The grouped columns if `return_dict=True`; otherwise None (the
            grouping is printed instead).
        """
        descriptions = {
            name: description for name, description, _members in _SOLUTION_COLUMN_GROUPS
        }
        descriptions["Left behind (beyond a threshold)"] = (
            "Headcounts beyond each cutoff in beyond_thresholds (lower is better)."
        )
        descriptions["Other"] = (
            "Not yet categorised -- see return_solution_metrics()'s docstring."
        )

        # Printed in this order; "Underlying per-region data" and "Other"
        # stay last regardless of where they sit in _SOLUTION_COLUMN_GROUPS.
        ordered_names = [
            name
            for name, _description, _members in _SOLUTION_COLUMN_GROUPS
            if name != "Underlying per-region data"
        ]
        ordered_names += ["Left behind (beyond a threshold)", "Underlying per-region data", "Other"]

        groups = {name: [] for name in ordered_names}
        for col in self.solution_df.columns:
            groups[_classify_solution_column(col)].append(col)

        # Unlike the genuinely conditional groups above (present/absent as
        # actual columns), "Coverage" and "Equity" columns are always part
        # of solution_df's schema, holding placeholder NaN/None/"N/A ..."
        # values when not applicable -- so gate these two explicitly on
        # whether they're actually meaningful, matching
        # show_solutions_summary()'s "no group full of placeholders" rule.
        if self.site_problem.equity_data is None:
            groups["Equity"] = []
        if not self.solution_df["coverage_threshold"].notna().any():
            groups["Coverage"] = []

        groups = {name: cols for name, cols in groups.items() if cols}

        if return_dict:
            return groups

        for name, cols in groups.items():
            print(f"=== {name} ===")
            print(descriptions[name])
            for col in cols:
                print(f"  {col}")
            print()

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

    def _resolve_site_diff(self, df, diff_against):
        """
        Per-row site-name diff against a chosen reference, for telling
        apart near-identical top-N rows that only differ by one or two
        sites out of many -- unlike `sites_closed_vs_baseline`/
        `sites_added_vs_baseline`, this doesn't require `solve(baseline=
        ...)`; it diffs rows against EACH OTHER (or a fixed reference
        site set), which is the only option available when there's no
        external baseline to compare against at all.

        Parameters
        ----------
        df : pandas.DataFrame
            `solution_df` (or a `.head(n_best)` slice of it).
        diff_against : {"default", "rank_1", "previous_rank", "required_sites"}
            "default" resolves to "required_sites" if at least one site is
            flagged via `add_sites(required_sites_col=...)`, else
            "rank_1" -- required sites are the closest thing to a "we
            already have these, what's new" reference when one is
            configured, and the top-ranked solution is the next most
            natural default otherwise. "rank_1" diffs every row against
            the `solution_rank == 1` row's site_names. "previous_rank"
            diffs every row against the row immediately above it once
            sorted by `solution_rank` (the first row has nothing above
            it, so its diff is empty/zero, not `NaN` -- `previous_rank`
            answers "what changed one rank down", and there IS no rank
            below rank 1). "required_sites" diffs every row against the
            required-sites set directly, raising if none are configured
            (silently falling back would be surprising for an explicit
            request, unlike "default"'s silent fallback).

        Returns
        -------
        dict or None
            `{"added": pandas.Series, "removed": pandas.Series, "changed":
            pandas.Series, "label": str}` -- `added`/`removed` are
            comma-joined site-name strings, `changed` is the size of the
            symmetric difference (`len(added) + len(removed)`), and
            `label` names the reference for use in column headers (e.g.
            "rank 1"). `None` if `df` has no `solution_rank` column at
            all (a single directly-evaluated solution has nothing to
            diff against).

        Raises
        ------
        ValueError
            If `diff_against` isn't one of the values above, or if
            `diff_against="required_sites"` was explicitly requested but
            no required sites are configured.
        """
        valid = {"default", "rank_1", "previous_rank", "required_sites"}
        if diff_against not in valid:
            raise ValueError(
                f"diff_against must be one of {sorted(valid)}, got {diff_against!r}."
            )

        if "solution_rank" not in df.columns:
            return None

        required_indices = _get_required_site_indices(self.site_problem)

        if diff_against == "default":
            diff_against = "required_sites" if required_indices else "rank_1"

        if diff_against == "required_sites":
            if not required_indices:
                raise ValueError(
                    "diff_against='required_sites' requires at least one site "
                    "flagged via add_sites(required_sites_col=...) -- none are "
                    "configured on this problem. Pass diff_against='rank_1' or "
                    "'previous_rank' instead, or leave diff_against='default' "
                    "to fall back to 'rank_1' automatically."
                )
            candidate_sites = self.site_problem.candidate_sites
            id_col = self.site_problem._candidate_sites_candidate_id_col
            reference_names = set(
                candidate_sites.loc[
                    candidate_sites["canonical_site_index"].isin(required_indices),
                    id_col,
                ]
            )
            added = df["site_names"].apply(lambda names: sorted(set(names) - reference_names))
            removed = df["site_names"].apply(lambda names: sorted(reference_names - set(names)))
            label = "required sites"

        elif diff_against == "rank_1":
            rank_1_names = set(
                self.solution_df.loc[
                    self.solution_df["solution_rank"] == 1, "site_names"
                ].iloc[0]
            )
            added = df["site_names"].apply(lambda names: sorted(set(names) - rank_1_names))
            removed = df["site_names"].apply(lambda names: sorted(rank_1_names - set(names)))
            label = "rank 1"

        else:  # previous_rank
            ordered = df.sort_values("solution_rank", kind="stable")
            previous_names = ordered["site_names"].shift(1)
            added = pd.Series(
                [
                    sorted(set(current) - set(previous))
                    if isinstance(previous, list)
                    else []
                    for current, previous in zip(ordered["site_names"], previous_names)
                ],
                index=ordered.index,
            ).reindex(df.index)
            removed = pd.Series(
                [
                    sorted(set(previous) - set(current))
                    if isinstance(previous, list)
                    else []
                    for current, previous in zip(ordered["site_names"], previous_names)
                ],
                index=ordered.index,
            ).reindex(df.index)
            label = "previous rank"

        changed = added.apply(len) + removed.apply(len)
        return {
            "added": added.apply(lambda names: ", ".join(names)),
            "removed": removed.apply(lambda names: ", ".join(names)),
            "changed": changed,
            "label": label,
        }

    def show_solutions_summary(self, n_best=None, diff_against="default"):
        """
        Return a stakeholder-facing view of the solution table: a handful
        of plain-English columns with units in the header, in place of
        `show_solutions()`'s full ~30-40 column schema (jargon names, no
        units, and placeholder `None`/`NaN`/"N/A (No equity data)" columns
        whenever a given input -- equity data, a coverage threshold, a
        baseline -- wasn't registered).

        Always included: `Sites in this option` (`site_names`) and `Sites
        not in this option` (`unselected_site_names`), both joined into one
        readable string rather than a list pandas truncates mid-entry,
        `Average travel time (mins)` (`weighted_average`), and `Longest
        journey (mins)` (`max`). `Rank` is also included for a
        multi-solution `SiteSolutionSet` (from `solve()`), but omitted for a
        single directly-evaluated solution (`evaluate_baseline()` or
        `evaluate_single_solution_single_objective()`), which has no
        `solution_rank` column to show.

        `Additional sites chosen` (site_names minus the sites flagged via
        `add_sites(required_sites_col=...)`) is only added when at least one
        required site is configured -- e.g. for a "we have 4 sites and are
        opening 1 more" problem, it's just the new site, rather than
        `Sites in this option` repeating the same 4 required names alongside
        it in every row.

        `Sites added (vs <reference>)`/`Sites removed (vs <reference>)`/
        `Sites changed (vs <reference>)` (a per-row site-name diff against
        `diff_against`, see below) are added whenever there's more than one
        solution to tell apart (i.e. `solution_rank` exists) -- this is
        what makes several near-identical top-N rows (e.g. a brute-force
        search that only ever swaps one or two sites out of many)
        distinguishable at a glance without scanning two long, truncated
        `Sites in this option` strings against each other by eye.

        Coverage columns (`People within <threshold> mins`, `% within
        <threshold> mins`) are only added if `threshold_for_coverage` was
        set on this solve. `Sites closed` and `Sites added` (`sites_closed_
        vs_baseline`/`sites_added_vs_baseline`, joined into readable
        strings) and the population-impact-vs-baseline columns (`People
        with a longer/shorter journey`, `% of cohort with a longer/shorter
        journey`, `Avg increase/reduction for them (mins)`) are only added
        if a baseline was supplied (`solve(baseline=...)` or
        `evaluate_baseline()`/`evaluate_single_solution_single_
        objective(baseline=...)`). Equity columns (`Equity gap (mins,
        best vs worst group)` and the two plain-English equity verdicts)
        are only added if equity data was registered via `add_equity_data()`.
        A section absent from the input is omitted entirely rather than
        shown full of placeholders.

        Parameters
        ----------
        n_best : int, optional
            Number of top-ranked solutions to include. If None, every row
            in `solution_df` is included.
        diff_against : {"default", "rank_1", "previous_rank", "required_sites"}, default "default"
            What each row's site-name diff columns are computed against.
            "default" resolves to "required_sites" if at least one site is
            flagged via `add_sites(required_sites_col=...)` (the closest
            thing to "what's new beyond what we already have"), else falls
            back to "rank_1" silently. "rank_1" diffs every row against
            the top-ranked solution's site_names -- a single, stable
            reference point for comparing any row directly to the best
            one. "previous_rank" diffs every row against the row one rank
            better than it -- reads as "what changes if you loosen the
            criteria one more notch", though unlike "rank_1" the
            reference itself moves every row, so two non-adjacent ranks
            can't be compared directly from this column alone.
            "required_sites" diffs directly against the required-sites
            set, raising `ValueError` if none are configured (an explicit
            request for a reference that doesn't exist, unlike
            "default"'s silent fallback). See `_resolve_site_diff` for the
            full definition.

        Returns
        -------
        pandas.DataFrame
            One row per solution, columns as described above. People counts
            are whole numbers, travel times are rounded to 1 decimal place,
            and coverage is given as both a headcount and a percentage. Any
            `NaN` (e.g. `mean_reduction_among_improved` when nobody's
            journey actually improved) is filled with 0, since a stakeholder
            reading of "no one" is better served by 0 than an unexplained
            blank.

        Notes
        -----
        This is a read-only, additive view -- it never modifies
        `solution_df`, and `show_solutions()`'s own column set is
        unaffected. Use `show_solutions()` (optionally with
        `expand_dict_columns=True`) for the full underlying data, e.g. for
        further computation or export.

        The equity verdict columns are full sentences and can exceed 50
        characters. Displaying this DataFrame bare in Jupyter truncates any
        cell over pandas' `display.max_colwidth` (50 by default) with
        "..." -- run `pd.set_option("display.max_colwidth", None)` once at
        the top of a notebook to see them in full, or call
        `.to_string()` on the result.
        """
        df = self.solution_df.head(n_best)

        summary = pd.DataFrame(index=df.index)
        if "solution_rank" in df.columns:
            summary["Rank"] = df["solution_rank"]
        summary["Sites in this option"] = df["site_names"].apply(
            lambda names: ", ".join(names)
        )
        if "additional_site_names" in df.columns:
            summary["Additional sites chosen"] = df["additional_site_names"].apply(
                lambda names: ", ".join(names)
            )
        summary["Sites not in this option"] = df["unselected_site_names"].apply(
            lambda names: ", ".join(names)
        )

        site_diff = self._resolve_site_diff(df, diff_against)
        if site_diff is not None:
            label = site_diff["label"]
            summary[f"Sites added (vs {label})"] = site_diff["added"]
            summary[f"Sites removed (vs {label})"] = site_diff["removed"]
            summary[f"Sites changed (vs {label})"] = site_diff["changed"]

        summary["Average travel time (mins)"] = df["weighted_average"].round(1)
        summary["Longest journey (mins)"] = df["max"].round(1)

        if df["coverage_threshold"].notna().any():
            threshold = df["coverage_threshold"].iloc[0]
            summary[f"People within {threshold:.0f} mins"] = df[
                "demand_within_coverage_threshold"
            ].round(0).astype("Int64")
            summary[f"% within {threshold:.0f} mins"] = (
                df["proportion_within_coverage_threshold"] * 100
            ).round(1)

        if "sites_closed_vs_baseline" in df.columns:
            summary["Sites closed"] = df["sites_closed_vs_baseline"].apply(
                lambda names: ", ".join(names)
            )
            summary["Sites added"] = df["sites_added_vs_baseline"].apply(
                lambda names: ", ".join(names)
            )

        if "demand_worsened" in df.columns:
            summary["People with a longer journey"] = (
                df["demand_worsened"].round(0).astype("Int64")
            )
            summary["% of cohort with a longer journey"] = (
                df["proportion_demand_worsened"] * 100
            ).round(1)
            summary["Avg increase for them (mins)"] = df[
                "mean_increase_among_worsened"
            ].round(1)
            summary["People with a shorter journey"] = (
                df["demand_improved"].round(0).astype("Int64")
            )
            summary["% of cohort with a shorter journey"] = (
                df["proportion_demand_improved"] * 100
            ).round(1)
            summary["Avg reduction for them (mins)"] = df[
                "mean_reduction_among_improved"
            ].round(1)

        if self.site_problem.equity_data is not None:
            summary["Equity gap (mins, best vs worst group)"] = df[
                "gap_absolute_weighted"
            ].round(1)
            summary["Equity verdict (best vs worst group)"] = df[
                "gap_relative_description"
            ]
            summary["Equity verdict (most vs least deprived third)"] = df[
                "inter_tertile_description"
            ]

        return summary.fillna(0)

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
        demand=None,
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
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Computes `total_demand` and (for
            `by="demand"`) `proportion` / `average_travel_cost` under that
            scenario's demand instead of the primary demand data. Combines
            freely with `matrix=`.

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

        if demand is None:
            demand_col = getattr(self.site_problem, "_demand_data_demand_col", None)
        else:
            registered_demand_labels = getattr(
                self.site_problem, "secondary_demand_matrices", {}
            )
            if demand not in registered_demand_labels:
                raise ValueError(
                    f"Unknown secondary demand scenario '{demand}'. "
                    f"Registered labels: {sorted(registered_demand_labels)}."
                )
            demand_col = f"demand__{demand}"

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
        catchment_size=None,
        distance_decay=None,
        rank_on=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
        demand=None,
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
        catchment_size : float, optional
            Classic 2SFCA's hard catchment threshold d0, in the travel
            matrix's registered units. Mutually exclusive with
            `distance_decay` -- exactly one of the two must be given. See
            `SiteProblem.two_step_floating_catchment` for details.
        distance_decay : list of (float, float) or dict, optional
            Enhanced 2SFCA step-decay bands or a continuous decay kernel,
            forwarded as-is. See `SiteProblem.two_step_floating_catchment`.
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
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Scores accessibility under that
            scenario's demand instead of the primary demand data.
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
            distance_decay=distance_decay,
            site_names=selected_sites,
            matrix=matrix,
            demand=demand,
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
