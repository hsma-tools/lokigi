from lokigi.utils import (
    SOLVER_DEFINITIONS,
    SUPPORTED_OBJECTIVES,
    PLANNED_OBJECTIVES,
    _FOR_RANKING_BASE_COLUMNS,
    _RANK_SCORE_COL,
    _add_rank_score_column,
    _resolve_ranking_metric,
    _add_rank_column,
    _apply_cost_weighting,
    _get_required_site_indices,
)

from lokigi.site_solutions import (
    EvaluatedCombination,
    SiteSolutionSet,
    _BASELINE_SITE_NAMES_KEY,
)

# Data manipulation imports
import pandas as pd

# Other imports
from warnings import warn
import numpy as np
from typing import Literal
from .mixins.site_solvers import BruteForceMixin, GreedyMixin, GraspMixin
from .mixins.site_attributes import SiteAttributeMixin
from .mixins.site_accessibility import SFCAMixin, AccessibilityPlotMixin
from .mixins.site_eda import (
    SiteProblemEDAMixin,
    HotspotPlotMixin,
    SiteProblemHotspotCalculationMixin,
)
import copy
from lokigi.problem import _Problem


def _safe_idxmin(cost_frame, min_cost):
    """`cost_frame.idxmin(axis=1)`, tolerant of rows where every column is
    NaN. Pandas raises `ValueError` ("Encountered all NA values") on those
    rather than returning NaN, since >= 2.0 -- reachable whenever a demand
    location has no feasible journey to any selected site.

    `min_cost` (`cost_frame.min(axis=1)`, already computed by the caller)
    is NaN precisely for those all-NaN rows, so its mask is reused rather
    than re-derived. Returns an object-dtype Series: the selected site name
    for reachable rows, `pandas.NA` for unreachable ones.
    """
    reachable = min_cost.notna()
    selected_site = pd.Series(pd.NA, index=cost_frame.index, dtype=object)
    if reachable.any():
        selected_site.loc[reachable] = cost_frame.loc[reachable].idxmin(axis=1)
    return selected_site


class SiteProblem(
    _Problem,
    SiteAttributeMixin,
    SFCAMixin,
    AccessibilityPlotMixin,
    BruteForceMixin,
    GreedyMixin,
    GraspMixin,
    SiteProblemEDAMixin,
    HotspotPlotMixin,
    SiteProblemHotspotCalculationMixin,
):
    """
    Facility location optimization for healthcare site planning.

    A comprehensive toolkit for solving spatial optimization problems in healthcare
    service delivery. This class supports multiple location-allocation models
    including p-median, p-center, and maximal covering location problems (MCLP),
    with various solution strategies from exact brute-force to heuristic methods.

    The class handles the complete workflow from data ingestion (demand patterns,
    candidate sites, travel costs) through optimization to solution evaluation,
    with built-in support for geographic data and spatial visualizations.

    Parameters
    ----------
    preferred_crs : str, default "EPSG:27700"
        The coordinate reference system for spatial data. All geographic inputs
        will be transformed to this CRS. Defaults to British National Grid.
    debug_mode : bool, default True
        If True, enables verbose output during optimization and data processing.

    Attributes
    ----------
    demand_data : pandas.DataFrame or geopandas.GeoDataFrame or None
        Patient or service demand locations with associated weights.
    candidate_sites : geopandas.GeoDataFrame or None
        Potential facility locations available for optimization.
    travel_matrix : pandas.DataFrame or None
        Cost matrix (time/distance) between demand points and candidate sites.
    region_geometry_layer : geopandas.GeoDataFrame or None
        Optional geographic boundaries for visualization (e.g., LSOA polygons).
    travel_and_demand_df : pandas.DataFrame or geopandas.GeoDataFrame or None
        Internal merged dataset combining demand and travel cost data.
    total_n_sites : int or None
        Total number of candidate facilities available for optimization.

    Notes
    -----
    The class implements three inheritance mixins providing different solution
    strategies:

    - BruteForceMixin: Exhaustive enumeration for small problems
    - GreedyMixin: Fast constructive heuristic for larger problems
    - GraspMixin: Randomized adaptive search with local optimization

    Supported optimization objectives:

    - 'simple_p_median': Minimize total unweighted travel distance/time
    - 'hybrid_simple_p_median': Simple p-median with maximum distance/time constraint
    - 'p_median': Minimize total weighted travel distance/time
    - 'hybrid_p_median': P-median with maximum distance/time constraint
    - 'p_center': Minimize maximum travel distance/time
    - 'mclp': Maximize coverage within a distance/time threshold
    """

    # MARK: init
    def __init__(self, preferred_crs="EPSG:27700", debug_mode=True):
        self.preferred_crs = preferred_crs

        self.demand_data = None  # Patient GeoDataFrame
        self._demand_data_type = None
        self._demand_data_id_col = None
        self._demand_data_demand_col = None

        self.candidate_sites = None  # Potential Clinic GeoDataFrame
        self._candidate_sites_type = None
        self._candidate_sites_candidate_id_col = None
        self._candidate_sites_vertical_col = None
        self._candidate_sites_horizontal_col = None
        self._candidate_sites_capacity_col = None
        self._candidate_sites_cost_col = None
        self._candidate_sites_current_load_col = None
        self._candidate_sites_utilisation_col = None
        self._candidate_sites_required_sites_col = None
        self.total_n_sites = None

        self.travel_and_demand_df = None
        self._joined_demand_travel_df_key_col = None

        super().__init__(preferred_crs, debug_mode)

    @property
    def total_demand(self):
        """
        Sum of the demand column registered via `add_demand()`, or `None`
        if no demand data has been registered. A stable, discoverable
        denominator for sanity-checking absolute-headcount metrics (e.g.
        `demand_within_coverage_threshold / total_demand ==
        proportion_within_coverage_threshold`) without recomputing it ad
        hoc, as `_coverage_stats()` and `site_allocation_summary()` each do
        internally.
        """
        if self.demand_data is None or self._demand_data_demand_col is None:
            return None
        return float(self.demand_data[self._demand_data_demand_col].sum())

    ####################################
    # MARK: Single solution evaluation
    ####################################
    def evaluate_single_solution_single_objective(
        self,
        objective: str = "p_median",
        weights=None,
        site_names=None,
        site_indices=None,
        capacitated=False,
        threshold_for_coverage=None,
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
    ):
        """
        Evaluate a specific set of facility sites against a single objective.

        This method calculates the performance of a given facility configuration
        (a 'solution'). It determines which demand points are assigned to which
        sites based on minimum travel cost and calculates coverage metrics if a
        threshold is provided.

        Parameters
        ----------
        objective : str, default "p_median"
            The name of the objective function to evaluate. Must be a value
            defined in `SUPPORTED_OBJECTIVES`.
        site_names : list of str, optional
            A list of site identifiers (column names in the travel matrix)
            representing the chosen solution.
        site_indices : list of int, optional
            A list of integer positions (column indices) representing the
            chosen solution.
        capacitated : bool, default False
            Whether to consider site capacity constraints. Currently, only
            `False` is supported.
        threshold_for_coverage : float or int, optional
            A distance or time value. Demand points with a minimum travel cost
            lower than this value are flagged as 'covered'. The resulting
            `proportion_within_coverage_threshold` metric is weighted by the
            demand registered via `add_demand()`; the unweighted share of
            regions is reported alongside it as
            `proportion_regions_within_coverage_threshold`.
        baseline_costs : dict[str, pandas.Series], optional
            Internal/advanced: forwarded to `EvaluatedCombination` to
            compute population-impact-vs-baseline metrics. Most callers
            should use `evaluate_baseline()` and
            `SolutionComparator.population_impact_summary()`, or
            `solve(baseline=...)`, rather than passing this directly.
        meaningful_change_threshold : float, default 0.0
            Only used when `baseline_costs` is given -- see
            `lokigi.utils._population_impact_metrics`.
        beyond_thresholds : float or sequence of float, optional
            One or more "left behind" travel-cost thresholds -- forwarded
            to `EvaluatedCombination`, see its `beyond_thresholds`
            parameter for the resulting `demand_beyond_threshold_<t>` /
            `regions_beyond_threshold_<t>` columns. Deliberately distinct
            from `threshold_for_coverage`: "covered" (good) and "beyond"
            (bad) cross the threshold in opposite directions, and this
            parameter accepts more than one value at once.
        unreachable_cost : float, optional
            Forwarded to `EvaluatedCombination` -- see its parameter of the
            same name. Produces `weighted_average_for_ranking`/
            `unweighted_average_for_ranking`/`max_for_ranking` on the
            returned combination, identical to their honest counterparts
            unless this is set. Most callers should use `solve(
            unreachable_cost=...)` rather than passing this directly;
            passing it here alone has no effect beyond that combination's
            own reported numbers, since a direct call doesn't rank/prune
            against other combinations at all.

        Returns
        -------
        EvaluatedCombination
            A results container containing the objective type, resolved site
            indices/names, and a detailed DataFrame of the demand assignments.

        Raises
        ------
        ValueError
            If an unsupported objective is passed, or if neither (or both)
            `site_names` and `site_indices` are provided.
        KeyError
            If provided `site_names` do not exist in the travel matrix columns.
        IndexError
            If provided `site_indices` are out of the bounds of the travel matrix.
        NotImplementedError
            If `capacitated=True` is requested.

        Notes
        -----
        The method assumes an uncapacitated assignment logic where every demand
        point is assigned to its nearest (lowest cost) active facility.

        If `self.travel_and_demand_df` has not been generated via a merge yet,
        this method calls `_create_joined_demand_travel_df` automatically.

        See Also
        --------
        EvaluatedCombination : The class used to wrap the output of this method.
        """
        # Check for valid objectives
        if isinstance(objective, list) and len(objective) > 1:
            warn(
                "Multi-objective optimization is coming in a future release."
                f"For now, just your first objective {objective[0]} has been taken."
            )

        objective = objective if isinstance(objective, str) else objective[0]

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"Unsupported objective ({objective}) passed.")

        # Ensure exactly one argument is provided out of site_names and
        # site_indices. Checked via `is not None` rather than truthiness --
        # `site_names and site_indices` treated an explicitly-passed empty
        # list ([]) as "not provided", so e.g. site_names=[] alongside
        # site_indices=[1] silently skipped the "not both" check.
        if (site_names is None and site_indices is None) or (
            site_names is not None and site_indices is not None
        ):
            raise ValueError(
                "Please provide either 'site_names' or 'site_indices', but not both. "
                "This helps prevent 'off-by-one' errors with numeric site IDs."
            )

        if (site_names is not None and len(site_names) == 0) or (
            site_indices is not None and len(site_indices) == 0
        ):
            raise ValueError("'site_names'/'site_indices' must not be an empty list.")

        if site_indices is not None and len(site_indices) != len(set(site_indices)):
            raise ValueError(
                f"site_indices contains duplicate entries: {site_indices}. "
                "Each site may only be selected once per solution."
            )

        # If we haven't come via the solve method, we will need to make the joined
        # demand and travel dataframe. Otherwise, that's handled once in that method.
        # Kept outside the try below so its own errors (e.g. no travel matrix
        # registered) aren't rewrapped as "Error mapping site names".
        if self.travel_and_demand_df is None:
            self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
            self._build_secondary_travel_frames()
            self._build_secondary_demand_frames()

        try:
            # We need to make sure that we use IDs and names completely consistently throughout.
            # 1. Resolve site_indices to actual Site IDs (names)
            if site_indices is not None:
                # .isin() silently drops any index that doesn't exist,
                # which is why "is the result non-empty" alone (the old
                # check here) wasn't enough: a partially-invalid list still
                # produced a non-empty result, silently evaluating a
                # smaller solution than the caller asked for. Check for
                # exactly which requested indices don't exist.
                valid_indices = set(self.candidate_sites["canonical_site_index"])
                invalid_indices = sorted(set(site_indices) - valid_indices)
                if invalid_indices:
                    raise IndexError(
                        f"Site indices {invalid_indices} not found in candidate "
                        f"sites (valid range: 0 to {self.total_n_sites - 1})."
                    )

                # Use .iloc to get the actual ID/Name from the master site list
                resolved_names = self.candidate_sites[
                    self.candidate_sites["canonical_site_index"].isin(site_indices)
                ][self._candidate_sites_candidate_id_col].tolist()
                # print(f"Site indices provided. Resolved names: {resolved_names}")
            else:
                # print(f"Name provided. Resolved names: {site_names}")
                resolved_names = site_names

            # 2. Map those names to the Travel Matrix column positions
            resolved_matrix_indices = self.travel_and_demand_df.columns.get_indexer(
                resolved_names
            )

            # print(f"Resolved matrix indices: {resolved_matrix_indices}")

            if -1 in resolved_matrix_indices:
                missing = [
                    resolved_names[i]
                    for i, idx in enumerate(resolved_matrix_indices)
                    if idx == -1
                ]
                raise KeyError(f"Sites not found in travel matrix: {missing}")

            # Smart sorting
            if site_indices is not None:
                original_indices = site_indices
            else:
                # Derive indices from candidate_sites using the resolved names
                original_indices = (
                    self.candidate_sites.set_index(
                        self._candidate_sites_candidate_id_col
                    )
                    .loc[resolved_names]["canonical_site_index"]
                    .tolist()
                )

            # Zip (Original Index, Site Name, Matrix Column Position)
            combined = list(
                zip(original_indices, resolved_names, resolved_matrix_indices)
            )

            # Sort by the original positional index
            combined.sort(key=lambda x: x[0])

            # Unpack
            final_indices = [x[0] for x in combined]
            final_names = [x[1] for x in combined]
            final_matrix_cols = [x[2] for x in combined]

            # print(f"Final indices: {final_indices}")
            # print(f"Final names: {final_names}")
            # print(f"Final matrix cols: {final_matrix_cols}")

        except Exception as e:
            if isinstance(e, (IndexError, KeyError)):
                raise
            raise ValueError(f"Error mapping site names: {e}")

        # Filter and calculate
        try:
            # Facility filtering code modified from
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
            # We use .iloc because we now have guaranteed integer positions
            active_facilities = self.travel_and_demand_df.iloc[
                :, final_matrix_cols
            ].copy()

        except IndexError:
            # Copy-paste bug: this used to report `max_idx` twice (once as
            # the valid upper bound, then again mislabelled as "You
            # provided indices"), instead of the column positions that
            # were actually attempted.
            max_idx = self.travel_and_demand_df.shape[1] - 1
            raise IndexError(
                f"Index out of bounds. Your travel data has indices 0 to {max_idx}. "
                f"You provided indices: {final_matrix_cols}"
            )

        if not capacitated:
            # Assume travel to closest facility
            active_facilities["min_cost"] = active_facilities.min(axis=1)

            # Add column for the selected site (column name with minimum
            # cost). NaN for a demand location with no feasible journey to
            # any selected site (allow_missing=True travel matrices) --
            # _safe_idxmin sidesteps idxmin's ValueError on such all-NaN
            # rows rather than letting it propagate.
            active_facilities["selected_site"] = _safe_idxmin(
                active_facilities, active_facilities["min_cost"]
            )

            if threshold_for_coverage is None:
                active_facilities["within_threshold"] = np.nan
            else:
                active_facilities["within_threshold"] = active_facilities[
                    "min_cost"
                ].apply(lambda x: x < threshold_for_coverage)

            # Secondary travel matrices: same min/selected/within-threshold
            # columns as the primary matrix, suffixed `__<label>`. Selected
            # by *name* (final_names), not the positional final_matrix_cols
            # used for the primary matrix above -- those positions are
            # specific to travel_and_demand_df's column layout and are
            # meaningless for a secondary frame's own columns.
            for label, frame in self._secondary_travel_frames.items():
                sub = frame.loc[active_facilities.index, final_names]
                sub_min_cost = sub.min(axis=1)
                active_facilities[f"min_cost__{label}"] = sub_min_cost
                active_facilities[f"selected_site__{label}"] = _safe_idxmin(
                    sub, sub_min_cost
                )

                matrix_threshold = self.secondary_travel_matrices[label][
                    "threshold_for_coverage"
                ]
                if matrix_threshold is None:
                    matrix_threshold = threshold_for_coverage

                if matrix_threshold is None:
                    active_facilities[f"within_threshold__{label}"] = np.nan
                else:
                    active_facilities[f"within_threshold__{label}"] = (
                        active_facilities[f"min_cost__{label}"] < matrix_threshold
                    )

            # Secondary demand scenarios: merge in a `demand__<label>`
            # column while active_facilities' index is still the demand
            # location IDs (matching _secondary_demand_frames' index).
            # Adding it here -- rather than reindexing separately after the
            # merges below -- lets it ride through the same
            # reset_index()/merge sequence as every other computed column,
            # so it stays correctly row-aligned with evaluated_combination_df
            # without depending on merge order being preserved. This
            # alignment is what the equity-group breakdown in
            # site_solutions.py (`active_weights.loc[group.index]`) relies
            # on -- a separately-indexed Series would silently break that
            # lookup whenever equity data is also registered.
            for label, dser in self._secondary_demand_frames.items():
                active_facilities[f"demand__{label}"] = dser.loc[
                    active_facilities.index
                ]

            afi = active_facilities.index
            active_facilities = active_facilities.reset_index()

            # Re-add the demand data
            active_facilities = active_facilities.merge(
                self.demand_data,
                left_on=afi,
                right_on=self._demand_data_id_col,
                how="inner",
                suffixes=("", "_y"),
            )

            active_facilities = active_facilities.drop(
                active_facilities.filter(regex="_y$").columns, axis=1
            )

            # Re-add any additional data
            if self.additional_data is not None:
                for additional_dataset in self.additional_data:
                    # print(f"Right {additional_dataset['join_col']}")
                    # print(f"Left {afi}")

                    active_facilities = active_facilities.merge(
                        additional_dataset["data"],
                        left_on=afi,
                        right_on=additional_dataset["join_col"],
                        how="left",
                        suffixes=("", "_y"),
                    )
                    active_facilities = active_facilities.drop(
                        active_facilities.filter(regex="_y$").columns, axis=1
                    )
            # Add equity data to the dataframe if present
            if self.equity_data is not None:
                # print(
                #     f"self._joined_demand_travel_df_key_col: {self._joined_demand_travel_df_key_col}"
                # )
                # print(f"self._equity_data_common_col: {self._equity_data_common_col}")
                # print(f"active_facilities: {active_facilities.head(1)}")
                # print(f"self.equity_data: {self.equity_data.head(1)}")

                # how="left": demand points missing from the equity data
                # must keep their travel/cost rows (with NaN equity values)
                # rather than being dropped from the evaluation. An inner
                # join here silently shrank every metric -- max, weighted
                # averages, coverage -- whenever the equity data didn't
                # cover all demand locations, even when equity wasn't
                # being weighted.
                active_facilities = pd.merge(
                    active_facilities,
                    self.equity_data,
                    left_on=afi,
                    right_on=self._equity_data_common_col,
                    how="left",
                    suffixes=("", "_y"),
                )

                active_facilities = active_facilities.drop(
                    active_facilities.filter(regex="_y$").columns, axis=1
                )
        else:
            raise NotImplementedError(
                "Capacitated solving not yet supported. Please rerun with capacitated=False."
            )

        return EvaluatedCombination(
            objective,
            site_indices=final_indices,
            site_names=final_names,
            evaluated_combination_df=active_facilities,
            weights=weights,
            site_problem=self,
            coverage_threshold=threshold_for_coverage,
            baseline_costs=baseline_costs,
            meaningful_change_threshold=meaningful_change_threshold,
            beyond_thresholds=beyond_thresholds,
            unreachable_cost=unreachable_cost,
        )

    # MARK: evaluate_baseline()
    def evaluate_baseline(
        self,
        site_names=None,
        site_indices=None,
        objective: str = "p_median",
        weights=None,
        threshold_for_coverage=None,
        beyond_thresholds=None,
    ):
        """
        Evaluate the current ("do-nothing") network as a one-solution
        `SiteSolutionSet`, for use as a baseline with
        `SolutionComparator.population_impact_summary()` or
        `solve(baseline=...)`.

        A thin, ergonomic wrapper around
        `evaluate_single_solution_single_objective()` -- it wraps that
        single result in a one-row `solution_df` so the baseline can be
        passed around and compared using the same `SiteSolutionSet` API as
        any solved solution (`site_allocation_summary()`,
        `SolutionComparator`, etc.), without hand-building that DataFrame.

        Parameters
        ----------
        site_names : list of str, optional
        site_indices : list of int, optional
            The current network's sites. At most one of these may be
            given. If neither is given, defaults to the sites flagged via
            `add_sites(required_sites_col=...)` -- i.e. "the sites already
            required in every solution" -- which is the existing network
            in the common case of modelling "which *additional* sites
            should we open". Raises `ValueError` if neither is given and
            no `required_sites_col` is configured.
        objective : str, default "p_median"
            Forwarded to `evaluate_single_solution_single_objective()`.
        weights : dict, optional
            Forwarded to `evaluate_single_solution_single_objective()`.
        threshold_for_coverage : float or int, optional
            Forwarded to `evaluate_single_solution_single_objective()`.
        beyond_thresholds : float or sequence of float, optional
            Forwarded to `evaluate_single_solution_single_objective()` --
            reports the baseline's own "left behind" headcounts alongside
            its coverage.

        Returns
        -------
        SiteSolutionSet
            A one-row solution set wrapping the baseline evaluation.

        Raises
        ------
        ValueError
            If both `site_names` and `site_indices` are given (see
            `evaluate_single_solution_single_objective()`), or if neither
            is given and no `required_sites_col` was registered via
            `add_sites()`.

        Notes
        -----
        Unlike `evaluate_single_solution_single_objective()` called
        directly, this mirrors `solve()`'s auto-setup: if no demand data
        has been registered via `add_demand()`, demand from all regions is
        assumed equal (with the same `UserWarning` `solve()` raises),
        rather than failing with a merge error when
        `evaluate_single_solution_single_objective()` hits `demand_data is
        None`. This is what lets `evaluate_baseline()` be called with no
        `add_demand()` step first, matching `solve()`'s own ergonomics.
        """
        if self.demand_data is None:
            self._setup_equal_demand_df()
            if objective != "mclp":
                warn(
                    "No demand data was provided. Demand from all regions "
                    "has been assumed to be equal. If you wish to override "
                    "this, run .add_demand() to add your demand dataframe "
                    "before calling evaluate_baseline() again."
                )

        if site_names is None and site_indices is None:
            required = _get_required_site_indices(self)
            if not required:
                raise ValueError(
                    "evaluate_baseline() needs a baseline site set: either "
                    "pass site_names= or site_indices= naming the current "
                    "network directly, or register it via "
                    "add_sites(required_sites_col=...) so evaluate_baseline() "
                    "can default to it."
                )
            site_indices = required

        evaluated = self.evaluate_single_solution_single_objective(
            objective=objective,
            weights=weights,
            site_names=site_names,
            site_indices=site_indices,
            threshold_for_coverage=threshold_for_coverage,
            beyond_thresholds=beyond_thresholds,
        )
        metrics = evaluated.return_solution_metrics()

        return SiteSolutionSet(
            solution_df=pd.DataFrame([metrics]),
            site_problem=self,
            objectives=objective,
            n_sites=len(evaluated.site_indices),
        )

    # MARK: solve()
    def solve(
        self,
        p: int,
        objectives: str = "p_median",
        weights: dict = None,
        rank_on=None,
        capacitated=False,  # Not yet implemented
        search_strategy: Literal["brute-force", "greedy", "grasp"] = "brute-force",
        brute_force_ignore_limit=False,
        show_progress=True,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        n_jobs=1,
        max_value_cutoff=None,  # only used for hybrid
        threshold_for_coverage=None,  # used for filtering in mclp or lscp, used for scoring in others
        grasp_num_solutions=5,
        grasp_alpha=0.2,
        grasp_max_attempts="default",
        grasp_min_sites_different=1,
        grasp_local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        grasp_max_swap_count_local_search=10,
        random_seed=42,
        full_secondary_metrics=False,
        baseline=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
    ):
        """
        Solve the site location problem using the specified objective and strategy.

        This method validates the problem configuration, handles automatic setup of
        missing demand or site data, and dispatches the optimization task to the
        appropriate internal solver.

        Parameters
        ----------
        p : int
            The number of facilities to be located.
        objectives : str or list of str, default "p_median"
            The optimization objective(s). Currently, only single-objective
            optimization is supported; if a list is provided, only the first
            element is used. Supported: "p_median", "p_center", "mclp", etc.

            The objective sets both which metric is ranked by default and
            which constraints apply ("mclp" requires `threshold_for_coverage`;
            the "hybrid_*" models require `max_value_cutoff`). `rank_on`
            below overrides only the first of those, so the two compose:
            `objectives="hybrid_p_median", max_value_cutoff=60,
            rank_on="inter_tertile_ratio"` means "cap the worst journey at 60
            minutes, then pick the most equitable option that qualifies".

            Pass "custom" for no model constraints at all -- it requires
            `rank_on` and rejects `max_value_cutoff`. Use it when you'd
            rather the returned `SiteSolutionSet` not name a textbook model
            the run didn't actually perform.
        weights: dict
            Only used with p_median.
            A dictionary of weights. Recognized keys are "demand", "equity",
            "cost" (requires `add_sites(cost_col=...)` to have been called),
            and any label registered via `add_additional_data()`. Not
            supported for "p_center", which ranks solely by worst-case
            travel time.

            These are *row-level* weights over demand regions: they change
            how `weighted_average` is computed, not which metric is ranked.
            To rank on a different metric, use `rank_on`, not `weights`.
            ("cost" is the exception -- a per-combination value blended in
            after the fact, which is why it forces brute-force pruning to
            materialise every combination first.)
        rank_on : str or lokigi.multiobjective.Metric, optional
            Rank and prune on this metric instead of the one implied by
            `objectives`. Any scalar column `solve()` computes is valid --
            `"inter_tertile_ratio"`, `"90th_percentile"`,
            `"demand_beyond_threshold_45"`, `"proportion_demand_improved"`,
            a `"<metric>__<label>"` secondary-matrix column, and so on. Call
            `describe_solution_columns()` on any previous result to list
            them.

            Unlike re-sorting a finished `SiteSolutionSet`, this drives the
            search itself, so it also decides which combinations survive
            `brute_force_keep_best_n` or GRASP's pool -- re-sorting can only
            reorder candidates that were already kept for a different
            metric.

            A bare string takes its direction from the usual convention
            (coverage/improvement metrics are higher-is-better, travel costs
            lower-is-better). Pass a `Metric` to state it explicitly, which
            is the only way to express a metric that is best at neither
            extreme::

                from lokigi.multiobjective import Metric

                problem.solve(
                    p=3,
                    rank_on=Metric(
                        "inter_tertile_ratio",
                        direction="closest_to_target",
                        target=1.0,   # 1.0 = equal travel across equity bands
                    ),
                )

            Validated against a representative combination before the search
            starts, so naming a column that doesn't exist, holds one value
            per equity band, or is NaN because its precondition wasn't met
            (e.g. `inter_tertile_ratio` without `add_equity_data()`) fails
            immediately rather than after a full solve.
        capacitated : bool, default False
            Whether to enforce site capacity constraints.
            *Note: Currently not implemented.*
        search_strategy : {"brute-force", "greedy", "grasp"}, default "brute-force"
            The algorithm used to find the solution:
            - "brute-force": Exhaustively checks all combinations (if p is small).
            - "greedy": Iteratively adds the best performing site.
            - "grasp": Greedy Randomized Adaptive Search Procedure.
        brute_force_ignore_limit : bool, default False
            (Brute Force only) If True, allows brute-force searching even if the number of
            combinations is extremely high.
        show_progress : bool, default True
            If True, displays a progress bar during the optimization process.
        brute_force_keep_best_n / brute_force_keep_worst_n : int, optional
            (Brute Force only) The number of top or bottom results to retain during a
            brute-force search. Normally this prunes combinations on the fly
            to bound memory use. If `weights` includes a positive "cost"
            weight, that streaming prune is skipped: every combination is
            evaluated and held in memory so cost can be blended in over the
            full batch before pruning to N, otherwise a combination that
            only looks good once cost is considered could be discarded
            before cost is ever factored in. A UserWarning is raised when
            this fallback is triggered.
        n_jobs : int, default 1
            (Brute Force only) Number of worker processes to evaluate
            combinations in parallel. 1 (the default) evaluates every
            combination in the current process and is always
            byte-for-byte identical to prior (pre-parallel) behaviour,
            including which combination brute_force_keep_best_n /
            brute_force_keep_worst_n keeps on an exact score tie. -1 uses
            all available CPU cores. With n_jobs != 1, results are always
            correctly ranked and bounded to the requested count, but if
            more combinations tie exactly on score than keep_best_n /
            keep_worst_n allows, which specific tied combination is kept
            can differ from a serial run (their scores are still
            identical either way). Exact ties are rare with real-valued
            travel costs. The first call with a given n_jobs value in a
            process (or any call after switching to a different n_jobs
            value) pays a one-time worker-pool startup cost -- each worker
            process has to import pandas/numpy/etc. from scratch, which on
            Windows can take several seconds regardless of workload size.
            Subsequent calls with the same n_jobs reuse the already-running
            pool and are fast. For small combination counts that finish in
            a second or two serially, that startup cost can make a single
            one-off parallel call look slower than n_jobs=1 -- this is a
            fixed process-spawn cost, not a sign that parallelism itself is
            ineffective (repeated calls, or larger workloads, amortize it
            away).
        max_value_cutoff : float, optional
            The maximum allowable travel cost. Only applicable for hybrid
            objective models. All search strategies honour it: brute-force
            discards every combination whose worst-case travel exceeds it,
            greedy applies it when choosing the final site (raising a
            ValueError if no feasible completion exists), and GRASP rejects
            candidate solutions that violate it.
        threshold_for_coverage : float, optional
            The distance or time threshold. Used as a hard filter for MCLP
            objectives or as a scoring metric for others.

            Coverage is measured as the proportion of *demand* within the
            threshold, weighted by the demand registered via `add_demand()`
            (all regions weigh equally if it was never called). The `mclp`
            objective therefore maximises covered demand, matching the
            textbook Maximal Covering Location Problem. The unweighted share
            of regions is still reported, as
            `proportion_regions_within_coverage_threshold`.
        grasp_num_solutions : int, default 5
            (GRASP only) The number of high-quality solutions to generate.
        grasp_alpha : float, default 0.2
            (GRASP only) The selection restriction parameter (0 is fully
            greedy, 1 is fully random).
        grasp_max_attempts : int or "default", default "default"
            (GRASP only) Maximum iterations to find a valid solution.
        grasp_min_sites_different : int, default 1
            (GRASP only) Minimum number of sites that must differ between
            generated solutions. Useful for generating a more diverse
            solution pool, though you may need to increase the max_attempts
            at the same time.
        grasp_local_search_chance : float, default 0.8
            (GRASP only) The probability (0.0 to 1.0) of performing a local
            search to improve a found solution.
        grasp_max_swap_count_local_search : int, default 10
            (GRASP only) Maximum number of site swaps allowed during the
            local search phase.
        random_seed : int, default 42
            (GRASP only) Seed for reproducibility in randomized strategies like GRASP.
        full_secondary_metrics : bool, default False
            If False (the default), each registered secondary travel matrix
            (see `add_secondary_travel_matrix()`) contributes only its core
            five metrics plus the float-valued equity aggregations to
            `solution_df`. If True, every registered secondary matrix also
            contributes its dict-valued equity breakdowns (e.g.
            `weighted_by_equity_group__<label>`) and description strings,
            matching what the primary matrix already always returns
            unsuffixed. This has no effect if no secondary matrices are
            registered, and costs nothing extra to compute -- the values are
            already computed either way, this only controls which of them
            are included in the returned table.
        baseline : None, True, or SiteSolutionSet, default None
            Compares every solution against a baseline "do-nothing"
            network, adding `demand_improved`/`demand_worsened`/
            `demand_unchanged`, `regions_improved`/`regions_worsened`/
            `regions_unchanged`, `mean_reduction_among_improved`,
            `mean_increase_among_worsened`, `max_reduction` and
            `max_increase` to every row of `solution_df` -- how many
            people's journey actually changed relative to the baseline,
            and by how much, rather than only the region-wide
            `weighted_average` shift (which dilutes a large local effect
            across everyone unaffected by it). See
            `EvaluatedCombination.return_solution_metrics`'s point 7, and
            `SolutionComparator.population_impact_summary()` for the
            equivalent baseline-vs-candidate comparison outside `solve()`.

            - `None` (the default): off. `solution_df`'s column set is
              byte-for-byte identical to before this parameter existed.
            - `True`: build the baseline from the sites flagged via
              `add_sites(required_sites_col=...)`, inheriting this call's
              `objectives`/`weights`/`threshold_for_coverage` (see
              `evaluate_baseline()`). Raises `ValueError` if no
              `required_sites_col` is configured.
            - A `SiteSolutionSet` containing exactly one solution
              (typically from `evaluate_baseline()`): used directly, so a
              baseline built with different objective/weights/threshold
              settings than this `solve()` call can be supplied
              explicitly.

            The baseline itself is evaluated once per `solve()` call, not
            once per enumerated combination -- negligible added cost
            regardless of search strategy.
        meaningful_change_threshold : float, default 0.0
            Only used when `baseline` is given. A region's travel cost
            must move by strictly more than
            `max(meaningful_change_threshold, 1e-9)` to count as improved
            or worsened; anything smaller (including floating-point noise
            at the default 0.0) is `unchanged`.
        beyond_thresholds : float or sequence of float, optional
            One or more "left behind" travel-cost thresholds, added to
            every row of `solution_df` as `demand_beyond_threshold_<t>` /
            `regions_beyond_threshold_<t>` -- how many people/regions have
            a travel cost beyond `t`, for each `t`. Distinct from
            `threshold_for_coverage`: "covered" (good) and "beyond" (bad)
            cross the threshold in opposite directions, and this parameter
            accepts more than one value at once (`threshold_for_coverage`
            does not). `None` (the default): off, `solution_df`'s column
            set is unchanged. See
            `EvaluatedCombination.return_solution_metrics`'s point 8.
        unreachable_cost : float, optional
            Required whenever the primary travel matrix was registered
            with `add_travel_matrix(allow_missing=True)` and actually
            contains a missing (NaN) travel cost, for every objective
            except `"mclp"` -- see the `NotImplementedError` this raises
            when it's needed but missing for the full explanation.

            A finite cost substituted for every unreachable pair, used
            ONLY to rank/prune combinations during search (and to enforce
            `max_value_cutoff`) -- never in `solution_df`'s reported
            `weighted_average`/`unweighted_average`/`90th_percentile`/
            `max`, and never in any plot, both of which stay honest,
            reachable-only figures throughout. The substituted view is
            still available for inspection as
            `weighted_average_for_ranking`/`unweighted_average_for_ranking`/
            `max_for_ranking`. Choose a value clearly worse than any real
            travel cost you'd consider acceptable (e.g. several times your
            longest reasonable journey) -- too small a value under-
            penalises stranding demand relative to a genuinely long but
            reachable journey; the two are not the same failure.

            `"mclp"` never needs this: its coverage-proportion ranking
            already treats an unreachable pair as "not covered" --
            correctly bad, with no equivalent silent-reward failure mode
            -- so it accepts a matrix with missing values regardless of
            whether `unreachable_cost` is set.

        Returns
        -------
        SiteSolutionSet
            An object containing the optimal sites, objective score, and
            detailed assignment data for each provided solution.

        Raises
        ------
        ValueError
            If `capacitated` is True, if the travel matrix is missing, if an
            unsupported objective/strategy is provided, or if `max_value_cutoff`
            is used with an incompatible objective.

        Raises
        -----
        UserWarning
            If multi-objective lists are provided (only the first is taken).
            If demand or site data is missing and must be auto-generated.

        Notes
        -----
        If `demand_data` or `candidate_sites` have not been explicitly added
        prior to calling `.solve()`, the method will automatically initialize
        them based on the travel matrix.
        """
        # Error early for common errors
        if search_strategy not in ["brute-force", "greedy", "grasp"]:
            raise ValueError(
                f"Unsupported search strategy ({search_strategy}) passed. Only 'brute-force', 'greedy' and 'grasp' are currently supported."
            )

        objective = objectives if isinstance(objectives, str) else objectives[0]

        if objective in PLANNED_OBJECTIVES:
            raise NotImplementedError(
                f"The '{objective}' objective is planned but not yet implemented. "
                f"Currently supported objectives are: {SUPPORTED_OBJECTIVES}."
            )

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"Unsupported objective ({objective}) passed.")

        if objective == "mclp" and threshold_for_coverage is None:
            raise ValueError(
                "The 'mclp' objective requires a threshold_for_coverage to be "
                "provided (the maximum travel time/distance for a demand point "
                "to count as covered)."
            )

        # 'custom' has no ranking column of its own -- it exists purely to
        # say "rank on what I name, with no model constraints", so without
        # rank_on there is nothing at all to optimise.
        if objective == "custom" and rank_on is None:
            raise ValueError(
                "The 'custom' objective requires rank_on=<column name or "
                "Metric> -- it applies no model of its own, so the metric to "
                "rank on has to come from you. Either pass rank_on, or use a "
                "named objective "
                f"({', '.join(o for o in SUPPORTED_OBJECTIVES if o != 'custom')})."
            )

        # Resolve what will actually be ranked. Done before the remaining
        # validation because several checks below depend on the resolved
        # column rather than on the objective.
        scorer, rank_on_warning = _resolve_ranking_metric(
            objective=objective, rank_on=rank_on, unreachable_cost=unreachable_cost
        )
        if rank_on_warning is not None:
            warn(rank_on_warning, UserWarning, stacklevel=2)

        # Error early if trying to use weights with unsupported or inadvisable problem types.
        # p_center ranks solely by worst-case travel time (max), so a weights
        # dict would have no effect on the outcome -- reject it explicitly
        # rather than silently ignoring it. Only applies when the caller
        # actually passed something; the implicit demand-only default is fine.
        if objective == "p_center" and weights is not None:
            raise ValueError(
                "Custom weights are not supported for the 'p_center' objective, "
                "since it ranks by worst-case travel time (max) rather than a "
                "weighted average. Please rerun without the 'weights' argument."
            )

        # Handle weights
        if weights is None:
            # Fall back to legacy behavior
            weights = {"demand": 1.0}

        if not isinstance(weights, dict):
            raise TypeError(
                f"Expected 'weights' to be a dict, got {type(weights).__name__}."
            )

        if any(w < 0 for w in weights.values()):
            raise ValueError("Weights cannot be negative.")

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("The sum of the weights must be greater than zero.")

        # Normalise weights to ensure they sum to exactly 1.0
        # This safely handles {"demand": 80, "equity": 20} -> {"demand": 0.8, "equity": 0.2}
        #
        # "demand", "equity", and "cost" are canonicalised to lowercase here
        # because every downstream consumer (site_solutions.py,
        # site_solvers.py, utils._apply_cost_weighting) looks them up via
        # exact-case comparisons/lookups, even though the missing-key
        # validation below accepts all three case-insensitively -- without
        # this, a differently-cased key like "Demand" or "Equity" would pass
        # validation but then silently fail to match any known column,
        # leaving the compound row weights all zero and crashing
        # np.average with "Weights sum to zero". Additional-data labels are
        # user-defined exact strings (not built-in keywords) and are
        # deliberately left untouched -- they are matched case-sensitively
        # on both sides (here and in site_solutions.py).
        _canonical_weight_keys = {"demand", "equity", "cost"}
        normalised_weights = {
            (col.lower() if col.lower() in _canonical_weight_keys else col): float(
                weight
            )
            / total_weight
            for col, weight in weights.items()
        }

        if capacitated:
            raise ValueError(
                "Capacitated modelling not yet supported. Please rerun with `capacitated=False`."
            )

        # Check minimum required information is provided
        # If travel matrix (or any cost matrix) is not provided, cannot continue
        if self.travel_matrix is None:
            raise ValueError(
                "No travel matrix or other cost matrix has been provided. Please add this using the .add_travel_matrix() method before running .solve() again."
            )

        # A primary travel matrix registered with allow_missing=True can
        # legitimately hold NaN cost values (evaluate_single_solution_
        # single_objective() already handles them correctly -- see
        # EvaluatedCombination._compute_travel_metrics). solve()'s
        # ranking/pruning is a different problem: computing
        # weighted_average etc. over reachable rows only, as that method
        # does, would silently reward a combination for stranding more
        # demand, since the excluded rows simply vanish from the average
        # rather than counting against it. unreachable_cost fixes that by
        # substituting a caller-chosen cost for every unreachable pair,
        # used only for ranking/pruning (see _get_ranking_by_objective).
        #
        # Required exactly when the column being ranked is one of the
        # reachable-only aggregates in _FOR_RANKING_BASE_COLUMNS, since
        # those are the ones carrying that silent-reward failure mode. Any
        # other ranking column already handles unreachability sensibly:
        # a coverage proportion counts an unreachable pair as "not covered"
        # (correctly bad), a beyond-threshold count as "beyond it", and so
        # on. Keyed off the resolved column rather than off the objective
        # (which is how this read before `rank_on` existed) so it stays
        # right for a custom ranking metric -- for the built-in objectives
        # the two are equivalent, since the three aggregates cover every
        # objective except mclp.
        if scorer.column in _FOR_RANKING_BASE_COLUMNS and unreachable_cost is None:
            primary_cost_cols = self.travel_matrix.columns.drop(
                self._travel_matrix_source_col
            )
            if self.travel_matrix[primary_cost_cols].isna().any().any():
                raise NotImplementedError(
                    f"solve() requires unreachable_cost=<a number> when "
                    f"ranking on '{scorer.column}' and the primary travel "
                    "matrix (registered with allow_missing=True) actually "
                    "contains missing (NaN) values -- optimising/ranking "
                    "over partially-unreachable demand isn't safe without "
                    "an explicit cost for what an unreachable pair should "
                    "count as, or a combination that strands more demand "
                    "would be silently preferred simply because the "
                    "excluded rows vanish from its average rather than "
                    "counting against it. unreachable_cost is used only to "
                    "rank/prune combinations during search -- it never "
                    "appears in solution_df's reported weighted_average/"
                    "unweighted_average/90th_percentile/max (those stay "
                    "honest, reachable-only figures) or in any plot. "
                    "Alternatively, evaluate individual combinations via "
                    "evaluate_single_solution_single_objective(), whose "
                    "reported metrics already correctly treat a missing "
                    "travel cost as \"no reachable site\" without needing "
                    "unreachable_cost at all. Secondary travel matrices "
                    "(add_secondary_travel_matrix()) are unaffected -- "
                    "they never drive optimisation, only reporting."
                )

        # If demand data not present,a ssume equal demand
        if self.demand_data is None:
            self._setup_equal_demand_df()
            # Compare against `objective` (the already-resolved single
            # string), not the raw `objectives` parameter -- when the
            # caller passes a list (e.g. objectives=["mclp"]), a list is
            # never equal to the string "mclp", so this warning fired even
            # for the exempted mclp objective.
            if objective != "mclp":
                warn(
                    "No demand data was provided. Demand from all regions has been assumed to be equal."
                    "If you wish to override this, run .add_demand() to add your site dataframe before running .solve() again."
                    "You can use the .show_demand_format() to see the expected format beforehand."
                )

        # If candidate sites not provided, make assumption from columns of travel/cost matrix
        if self.candidate_sites is None:
            self._setup_sites_df_from_travel_matrix()
            warn(
                "No candidate site dataframe was given."
                f"\nSites names have been taken from the columns of your travel matrix: {', '.join(self.candidate_sites[self._candidate_sites_candidate_id_col].to_list())}."
                "\nIf you wish to override this, run .add_sites() to add your site dataframe before running .solve() again."
                "\nYou can use the .show_sites_format() to see the expected format beforehand."
            )

        self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
        self._build_secondary_travel_frames()
        self._build_secondary_demand_frames()

        # Data Existence Check
        # Ensure the user didn't typo a name in their weights dictionary
        # This has to happen after any auto demand matrix is created
        # else demand will not exist in those cases
        missing_cols = []
        for col in weights.keys():
            col_lower = col.lower()

            if col_lower == "demand":
                if self.demand_data is None:
                    missing_cols.append(col)

            elif col_lower == "equity":
                if self.equity_data is None:
                    missing_cols.append(col)

            elif col_lower == "cost":
                if getattr(self, "_candidate_sites_cost_col", None) is None:
                    missing_cols.append(col)

            elif col in self.secondary_demand_matrices:
                # Valid weight key -- a secondary demand scenario's column
                # is resolved by EvaluatedCombination's compound_weights
                # builder, exactly like an additional-data label.
                pass

            elif col not in (self._additional_data_labels or []):
                missing_cols.append(col)

        if missing_cols:
            raise KeyError(
                f"The following weight keys are missing from the problem data: {missing_cols}"
            )

        # Equity coverage check. The per-solution equity merge is a left
        # join, so demand locations missing from the equity data keep their
        # travel metrics but carry NaN equity values. That is fatal for
        # equity weighting (the row weights would be NaN) and silently
        # excludes those locations from equity-band breakdowns, so surface
        # it here, once, before any solving starts.
        if self.equity_data is not None:
            demand_ids = self.travel_and_demand_df.index
            equity_ids = self.equity_data[self._equity_data_common_col]
            missing_equity_ids = demand_ids.difference(equity_ids)

            if len(missing_equity_ids) > 0:
                id_summary = (
                    f"{len(missing_equity_ids)} of {len(demand_ids)} demand "
                    f"location(s) have no matching row in the equity data "
                    f"(e.g. {list(missing_equity_ids[:5])})"
                )
                if any(key.lower() == "equity" for key in weights):
                    raise ValueError(
                        f"Cannot weight by equity: {id_summary}. Equity row "
                        "weights cannot be computed for these locations. "
                        "Please provide equity data covering every demand "
                        "location, or remove 'equity' from the weights dict."
                    )
                warn(
                    f"{id_summary}. These locations are still included in "
                    "all travel/cost metrics, but will be excluded from "
                    "equity-band breakdowns (e.g. coverage or averages per "
                    "equity group)."
                )

        if isinstance(objectives, list) and len(objectives) > 1:
            warn(
                "Multi-objective optimization is coming in a future release."
                f"For now, just your first objective {objectives[0]} has been taken."
            )

        # Resolved once here, not once per enumerated combination -- see
        # solve()'s `baseline` parameter docstring above.
        baseline_costs = self._resolve_baseline_costs(
            baseline, objective, normalised_weights, threshold_for_coverage
        )

        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model variant ({objective}) that doesn't support it."
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
            )

        if max_value_cutoff is None and objective == "hybrid_p_median":
            raise ValueError(
                "hybrid_p_median requires a max_value_cutoff (the maximum allowable "
                "travel cost every demand point must be guaranteed) -- without one, "
                "it has no 'safety net' constraint to apply. Please either provide "
                "max_value_cutoff, or use objective='p_median' if you don't need "
                "that guarantee."
            )

        if max_value_cutoff is None and objective == "hybrid_simple_p_median":
            raise ValueError(
                "hybrid_simple_p_median requires a max_value_cutoff (the maximum "
                "allowable travel cost every demand point must be guaranteed) -- "
                "without one, it has no 'safety net' constraint to apply. Please "
                "either provide max_value_cutoff, or use objective='simple_p_median' "
                "if you don't need that guarantee."
            )

        # Catch a bad rank_on now, against one representative combination,
        # rather than after a full search has run and produced a
        # meaninglessly-ordered result. Costs a single extra evaluation
        # (negligible beside the thousands the search itself does), and only
        # when rank_on was actually passed. Every argument that decides
        # which columns exist has to be forwarded, or the probe would report
        # a column missing that the real run would have had.
        if rank_on is not None:
            self._validate_rank_on(
                scorer=scorer,
                p=p,
                objective=objective,
                weights=normalised_weights,
                threshold_for_coverage=threshold_for_coverage,
                full_secondary_metrics=full_secondary_metrics,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            )

        if objective in [
            "p_median",
            "p_center",
            "simple_p_median",
            "hybrid_p_median",
            "hybrid_simple_p_median",
            "mclp",
            "custom",
        ]:
            return self._solve_pmedian_pcenter_mclp_problem(
                p,
                search_strategy=search_strategy,
                objective=objective,
                solve_scorer=scorer,
                weights=normalised_weights,
                brute_force_ignore_limit=brute_force_ignore_limit,
                show_progress=show_progress,
                brute_force_keep_best_n=brute_force_keep_best_n,
                brute_force_keep_worst_n=brute_force_keep_worst_n,
                n_jobs=n_jobs,
                max_value_cutoff=max_value_cutoff,
                grasp_num_solutions=grasp_num_solutions,
                grasp_alpha=grasp_alpha,
                grasp_max_attempts=grasp_max_attempts,
                grasp_min_sites_different=grasp_min_sites_different,
                threshold_for_coverage=threshold_for_coverage,
                random_seed=random_seed,
                grasp_local_search_chance=grasp_local_search_chance,  # Chance that local searching will happen to improve found solution
                grasp_max_swap_count_local_search=grasp_max_swap_count_local_search,
                full_secondary_metrics=full_secondary_metrics,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            )
        else:
            raise ValueError(f"Unknown objective '{objective}'.")

    # MARK: _validate_rank_on
    def _validate_rank_on(
        self,
        scorer,
        p,
        objective,
        weights,
        threshold_for_coverage,
        full_secondary_metrics,
        baseline_costs,
        meaningful_change_threshold,
        beyond_thresholds,
        unreachable_cost,
    ):
        """
        Check `rank_on` names something actually rankable, before searching.

        Works by evaluating one representative combination and inspecting
        the metrics dict it produces, rather than by checking the requested
        name against a hardcoded list. Which columns exist is genuinely
        intricate -- the population-impact family only appears with a
        `baseline`, `<metric>__<label>` columns depend on which secondary
        matrices are registered and on `full_secondary_metrics`,
        `demand_beyond_threshold_<t>` on `beyond_thresholds` -- and a list
        would drift out of step with `return_solution_metrics()` the first
        time a metric was added. Asking a real result is always current.

        A NaN is an error rather than a warning because it means the
        ranking column carries no information at all, so every solution
        would be tied and `solution_rank` would be meaningless -- silently
        returning an arbitrary ordering is the worst of the options. The
        preconditions that produce one (equity data registered, a coverage
        threshold given, a baseline supplied) depend on the problem's
        registered data rather than on which sites a combination picks, so
        one probe answers for all of them.
        """
        probe_indices = list(_get_required_site_indices(self))
        for index in range(self.total_n_sites):
            if len(probe_indices) >= p:
                break
            if index not in probe_indices:
                probe_indices.append(index)

        metrics = self.evaluate_single_solution_single_objective(
            site_indices=probe_indices,
            objective=objective,
            threshold_for_coverage=threshold_for_coverage,
            weights=weights,
            baseline_costs=baseline_costs,
            meaningful_change_threshold=meaningful_change_threshold,
            beyond_thresholds=beyond_thresholds,
            unreachable_cost=unreachable_cost,
        ).return_solution_metrics(full_secondary_metrics=full_secondary_metrics)

        column = scorer.column

        if column not in metrics:
            rankable = sorted(
                name
                for name, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
            )
            raise KeyError(
                f"rank_on='{column}' is not a metric this problem computes. "
                f"Rankable columns for this problem are: {', '.join(rankable)}. "
                "Call describe_solution_columns() on a previous result for "
                "these grouped and explained. Note that some columns only "
                "appear once their input is registered -- the population-"
                "impact metrics need solve(baseline=...), "
                "demand_beyond_threshold_<t> needs beyond_thresholds=, and "
                "'<metric>__<label>' needs the matching secondary travel "
                "matrix or demand scenario."
            )

        value = metrics[column]

        if isinstance(value, dict):
            raise ValueError(
                f"rank_on='{column}' holds one value per equity band (a "
                f"dict of {len(value)} entries), so there is no single "
                "number to rank solutions by. Rank on one of the scalar "
                "summaries derived from the bands instead -- "
                "'gap_absolute_weighted' (worst band minus best), "
                "'gap_relative_weighted' (worst band over best), or "
                "'inter_tertile_ratio' (most- over least-disadvantaged "
                "third) -- or pick a single band out of this column "
                "yourself after solving."
            )

        # None and NaN both mean "this metric wasn't measured", and which
        # one you get is an implementation detail of the metric (equity
        # summaries come back as None with no equity data registered, the
        # coverage family as NaN with no threshold). Both get the
        # precondition message, which is the useful one -- checked before
        # the type guard below, or a None would be reported as a mere type
        # error and send the reader looking for the wrong problem.
        if value is None or (
            isinstance(value, (int, float, np.integer, np.floating))
            and pd.isna(value)
        ):
            hints = {
                "inter_tertile": "register equity data via add_equity_data()",
                "equity": "register equity data via add_equity_data()",
                "gap_": "register equity data via add_equity_data()",
                "coverage": "pass threshold_for_coverage= to solve()",
                "within_coverage_threshold": "pass threshold_for_coverage= to solve()",
                "improved": "pass baseline= to solve()",
                "worsened": "pass baseline= to solve()",
                "reduction": "pass baseline= to solve()",
                "increase": "pass baseline= to solve()",
            }
            hint = next(
                (advice for key, advice in hints.items() if key in column),
                "check that whatever this metric is derived from is registered "
                "on the problem",
            )
            raise ValueError(
                f"rank_on='{column}' has no value for a representative "
                "combination, so every solution would tie and the ranking "
                f"would be meaningless. To rank on it, {hint}."
            )

        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
            value, bool
        ):
            raise ValueError(
                f"rank_on='{column}' is a {type(value).__name__}, not a "
                "number, so solutions cannot be ordered by it."
            )

    # MARK: _resolve_baseline_costs
    def _resolve_baseline_costs(self, baseline, objective, weights, threshold_for_coverage):
        """
        Resolve `solve()`'s `baseline` argument into a dict mapping cost-
        column name ("min_cost", or "min_cost__<label>" for a registered
        secondary travel matrix) to a baseline `pd.Series` indexed by
        demand-location ID, or `None` if no baseline was requested. See
        `solve()`'s `baseline` parameter for the accepted forms.

        Also carries the baseline's `site_names` under the reserved
        `_BASELINE_SITE_NAMES_KEY`, for `EvaluatedCombination` to compute
        `sites_closed_vs_baseline`/`sites_added_vs_baseline` -- see that
        constant's own docstring for why this rides along in the same dict
        rather than as a separate parameter.
        """
        if baseline is None:
            return None

        if baseline is True:
            baseline_set = self.evaluate_baseline(
                objective=objective,
                weights=weights,
                threshold_for_coverage=threshold_for_coverage,
            )
        elif isinstance(baseline, SiteSolutionSet):
            baseline_set = baseline
        else:
            raise TypeError(
                "solve(baseline=...) must be None, True, or a "
                "SiteSolutionSet (typically from evaluate_baseline()) -- "
                f"got {type(baseline).__name__}."
            )

        if len(baseline_set.solution_df) != 1:
            raise ValueError(
                "solve(baseline=...) requires a SiteSolutionSet containing "
                f"exactly one solution; got {len(baseline_set.solution_df)}. "
                "Build one with evaluate_baseline(), or select a single row "
                "before passing it in."
            )

        problem_df = baseline_set.solution_df.iloc[0]["problem_df"]
        id_col = self._demand_data_id_col
        if id_col not in problem_df.columns:
            raise ValueError(
                "solve(baseline=...) could not find demand-location ID "
                f"column '{id_col}' in the baseline's problem_df. The "
                "baseline must have been evaluated against this same "
                "problem."
            )
        indexed = problem_df.set_index(id_col)

        baseline_costs = {"min_cost": indexed["min_cost"]}
        for label in getattr(self, "secondary_travel_matrices", {}):
            col = f"min_cost__{label}"
            if col in indexed.columns:
                baseline_costs[col] = indexed[col]

        baseline_costs[_BASELINE_SITE_NAMES_KEY] = baseline_set.solution_df.iloc[0][
            "site_names"
        ]

        return baseline_costs

    # MARK: solve pmed pcen mclp
    def _solve_pmedian_pcenter_mclp_problem(
        self,
        p: int,
        objective="p_median",
        weights=None,
        search_strategy="brute-force",
        show_progress=False,
        brute_force_ignore_limit=False,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        n_jobs=1,
        max_value_cutoff=None,
        threshold_for_coverage=None,  # only used for mclp
        grasp_num_solutions=5,
        grasp_alpha=0.2,
        grasp_max_attempts="default",
        grasp_min_sites_different=1,
        grasp_local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        grasp_max_swap_count_local_search=10,
        random_seed=42,
        full_secondary_metrics=False,
        baseline_costs=None,
        meaningful_change_threshold=0.0,
        beyond_thresholds=None,
        unreachable_cost=None,
        solve_scorer=None,
    ):
        """
        Internal dispatcher for solving location-allocation problems.

        This method routes the problem to the appropriate search algorithm
        (Brute Force, Greedy, or GRASP) based on the specified strategy and
        objective. It handles ranking logic and result sorting before
        encapsulating outputs into a SiteSolutionSet.

        Parameters
        ----------
        p : int
            The number of facilities to be located.
        objective : str, default "p_median"
            The name of the objective function to optimize. Supported values
            typically include "p_median", "p_center", and "mclp".
        search_strategy : {"brute-force", "greedy", "grasp"}, default "brute-force"
            The search algorithm to apply.
        show_progress : bool, default False
            If True, displays a progress bar during computation.
        brute_force_ignore_limit : bool, default False
            (Brute Force) If True, bypasses safety checks on the total number of
            combinations for exhaustive searches.
        brute_force_keep_best_n : int, optional
            (Brute Force) The number of top-performing combinations to retain in
            brute-force results. If `weights` includes a positive "cost"
            weight, pruning falls back to materialising every combination
            first so cost can be blended in over the full batch before
            pruning to N (see `_brute_force`); a UserWarning is raised when
            this happens.
        brute_force_keep_worst_n : int, optional
            (Brute Force) The number of lowest-performing combinations to retain in
            brute-force results. Same cost-weighting fallback as
            `brute_force_keep_best_n` applies.
        n_jobs : int, default 1
            (Brute Force) Number of worker processes to evaluate combinations
            in parallel. 1 runs serially; -1 uses all available CPU cores.
            See `solve`'s docstring for the one-time worker-pool startup
            cost paid on the first call with a given n_jobs value.
        max_value_cutoff : float, optional
            The maximum allowable travel cost, used only for hybrid
            objective models.
        threshold_for_coverage : float, optional
            The maximum distance/time for a demand point to be considered
            'covered'. Required for "mclp" objectives.
        grasp_num_solutions : int, default 5
            (GRASP) Number of candidate solutions to generate.
        grasp_alpha : float, default 0.2
            (GRASP) Threshold for the Restricted Candidate List (RCL).
        grasp_max_attempts : int or "default", default "default"
            (GRASP) Maximum number of iterations to find distinct solutions.
        grasp_min_sites_different : int, default 1
            (GRASP) Minimum site difference required between solutions
            in the set.
        grasp_local_search_chance : float, default 0.8
            (GRASP) Probability of applying a local search (2-opt)
            improvement phase.
        grasp_max_swap_count_local_search : int, default 10
            (GRASP) Maximum number of facility swaps per local search attempt.
        random_seed : int, default 42
            (GRASP) Seed for random number generation to ensure reproducibility.

        Returns
        -------
        SiteSolutionSet
            A collection of solutions found, sorted by the primary objective
            ranking and weighted average costs.

        Raises
        ------
        ValueError
            If an unsupported objective or search strategy is provided.

        Notes
        -----
        `solve_scorer` (the `Metric` resolved by `_resolve_ranking_metric`)
        determines both the primary sorting column and its direction, and is
        threaded into whichever search strategy runs so all of them agree.
        Results are then sorted on `Metric.normalise()`'d values, where lower
        is always better -- so a higher-is-better column like mclp's coverage
        proportion, and one that is best at neither extreme like an
        inter-tertile ratio targeting 1.0, both sort correctly under a single
        ascending sort.
        """

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                "Unsupported objective passed to _solve_pmedian_pcenter_mclp_problem. Please contact a developer."
            )

        # One resolved Metric drives every comparison from here on -- the
        # brute-force heap, the greedy per-step sort, GRASP's construction
        # and local search, and the final cross-strategy sort below. Each of
        # those used to decide direction for itself with `objective ==
        # "mclp"`, which no `rank_on` outside that binary could satisfy.
        scorer = solve_scorer
        if scorer is None:
            scorer, _ = _resolve_ranking_metric(
                objective=objective, rank_on=None, unreachable_cost=unreachable_cost
            )
        ranking = scorer.column

        if objective in ["hybrid_p_median", "hybrid_simple_p_median"]:
            max_value_cutoff = max_value_cutoff
        else:
            max_value_cutoff = None

        if search_strategy not in ["brute-force", "greedy", "grasp"]:
            raise ValueError(f"Approach {search_strategy} not yet supported.")
        if search_strategy == "brute-force":
            outputs = self._brute_force(
                p=p,
                objectives=objective,
                weights=weights,
                brute_force_ignore_limit=brute_force_ignore_limit,
                show_progress=show_progress,
                brute_force_keep_best_n=brute_force_keep_best_n,
                brute_force_keep_worst_n=brute_force_keep_worst_n,
                scorer=scorer,
                max_value_cutoff=max_value_cutoff,
                threshold_for_coverage=threshold_for_coverage,
                n_jobs=n_jobs,
                full_secondary_metrics=full_secondary_metrics,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            )

        if search_strategy == "greedy":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._greedy(
                p=p,
                weights=weights,
                objectives=objective,
                scorer=scorer,
                show_progress=show_progress,
                threshold_for_coverage=threshold_for_coverage,
                max_value_cutoff=max_value_cutoff,
                full_secondary_metrics=full_secondary_metrics,
                baseline_costs=baseline_costs,
                meaningful_change_threshold=meaningful_change_threshold,
                beyond_thresholds=beyond_thresholds,
                unreachable_cost=unreachable_cost,
            )

        if search_strategy == "grasp":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._grasp(
                p=p,
                objectives=objective,
                weights=weights,
                scorer=scorer,
                threshold_for_coverage=threshold_for_coverage,
                num_solutions=grasp_num_solutions,
                alpha=grasp_alpha,
                max_attempts=grasp_max_attempts,
                show_progress=show_progress,
                random_seed=random_seed,
                min_sites_different=grasp_min_sites_different,
                local_search_chance=grasp_local_search_chance,  # Chance that local searching will happen to improve found solution
                max_swap_count_local_search=grasp_max_swap_count_local_search,
                max_value_cutoff=max_value_cutoff,
                full_secondary_metrics=full_secondary_metrics,
                baseline_costs=baseline_costs,
                beyond_thresholds=beyond_thresholds,
                meaningful_change_threshold=meaningful_change_threshold,
                unreachable_cost=unreachable_cost,
            )

        # An empty result set would otherwise crash further down with a
        # cryptic KeyError when ranking columns are missing from the empty
        # DataFrame -- most commonly caused by a max_value_cutoff strict
        # enough to rule out every combination.
        if len(outputs) == 0:
            cutoff_note = (
                f" with max_value_cutoff={max_value_cutoff}"
                if max_value_cutoff is not None
                else ""
            )
            raise ValueError(
                f"No feasible solutions were found for objective '{objective}' "
                f"using search_strategy='{search_strategy}'{cutoff_note}. "
                "Try relaxing max_value_cutoff, checking that p does not "
                "exceed the number of candidate sites, or using a different "
                "search strategy."
            )

        outputs_df = pd.DataFrame(outputs)

        score_col = ranking
        if weights and weights.get("cost", 0) > 0:
            outputs_df, score_col = _apply_cost_weighting(
                outputs_df,
                ranking_col=ranking,
                weights=weights,
                scorer=scorer,
            )

        # _apply_cost_weighting only blends onto its lower-is-better
        # "composite_score" scale when it actually has usable cost data to
        # blend; it no-ops (returns score_col == ranking unchanged) when
        # e.g. every candidate's total_cost is NaN. Whenever it hasn't
        # blended, put the raw ranking column onto that same scale via the
        # scorer, so the rank below is unconditionally ascending. Deciding
        # the direction from the objective instead, as this used to, ranked
        # the WORST combination first whenever cost weighting silently
        # no-op'd on a higher-is-better objective -- and could not express a
        # `closest_to_target` rank_on at all. This is the final
        # cross-strategy sort, so it affects all three alike.
        if score_col == ranking:
            outputs_df, score_col = _add_rank_score_column(outputs_df, scorer)

        solution_df = _add_rank_column(
            outputs_df,
            score_col=score_col,
            tiebreaker_col="weighted_average",
            ascending=[True, True],
        ).drop(columns=_RANK_SCORE_COL, errors="ignore")

        return SiteSolutionSet(
            solution_df=solution_df,
            site_problem=self,
            objectives=objective,
            n_sites=p,
            ranking_metric=scorer,
        )

    def evaluate_n_sites(self, min_sites, max_sites):
        raise NotImplementedError(
            "This method is not yet available, but is on the roadmap for future."
            "Please see examples in docs for how to do this manually."
        )

    def describe_models(self, available_only=True):
        """
        Prints a menu of available optimization strategies for healthcare.

        Parameters
        ----------
        available_only : bool
            Whether to limit the printout to only the models that are currently
            supported by the library rather than all methods
        """
        if available_only:
            print("=== Supported Healthcare Location Models ===")
        else:
            print("=== Healthcare Location Models ===")
        for key, info in SOLVER_DEFINITIONS.items():
            if available_only and not info["status"] == "Supported":
                continue

            print(f"\nID: {key}")
            print(f"Name: {info['name']}")
            print(f"Goal: {info['goal']}")
            print(f"When to use: {info['healthcare_context']}")
            print(f"Main Trade-off: {info['trade_off']}")
            if not available_only:
                print(f"Status: {info['status']}")
        print('\nTo run a model, use: prob.solve(p=3, objectives="p_median") or similar.')

    def copy(self):
        return copy.deepcopy(self)
