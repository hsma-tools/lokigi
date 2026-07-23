from lokigi.utils import (
    SOLVER_DEFINITIONS,
    SUPPORTED_OBJECTIVES,
    PLANNED_OBJECTIVES,
    _get_ranking_by_objective,
    _add_rank_column,
    _apply_cost_weighting,
)

from lokigi.site_solutions import EvaluatedCombination, SiteSolutionSet

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

        try:
            # If we haven't come via the solve method, we will need to make the joined
            # demand and travel dataframe. Otherwise, that's handled once in that method.
            # Otherwise
            if self.travel_and_demand_df is None:
                self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
                self._build_secondary_travel_frames()
                self._build_secondary_demand_frames()

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

            # Add column for the selected site (column name with minimum cost)
            active_facilities["selected_site"] = active_facilities.idxmin(axis=1)

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
                active_facilities[f"min_cost__{label}"] = sub.min(axis=1)
                active_facilities[f"selected_site__{label}"] = sub.idxmin(axis=1)

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
            # without depending on merge order being preserved.
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
        )

    # MARK: solve()
    def solve(
        self,
        p: int,
        objectives: str = "p_median",
        weights: dict = None,
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
        weights: dict
            Only used with p_median.
            A dictionary of weights. Recognized keys are "demand", "equity",
            "cost" (requires `add_sites(cost_col=...)` to have been called),
            and any label registered via `add_additional_data()`. Not
            supported for "p_center", which ranks solely by worst-case
            travel time.
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

        if objective in [
            "p_median",
            "p_center",
            "simple_p_median",
            "hybrid_p_median",
            "hybrid_simple_p_median",
            "mclp",
        ]:
            return self._solve_pmedian_pcenter_mclp_problem(
                p,
                search_strategy=search_strategy,
                objective=objective,
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
            )
        else:
            raise ValueError(f"Unknown objective '{objective}'.")

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
        The method uses `_get_ranking_by_objective` to determine the primary
        sorting column. For "mclp", results are sorted in descending order of
        coverage (higher is better), while for other objectives, results are
        sorted in ascending order of cost (lower is better).
        """

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                "Unsupported objective passed to _solve_pmedian_pcenter_mclp_problem. Please contact a developer."
            )

        ranking = _get_ranking_by_objective(objective=objective)

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
                rank_best_n_on=ranking,
                max_value_cutoff=max_value_cutoff,
                threshold_for_coverage=threshold_for_coverage,
                n_jobs=n_jobs,
                full_secondary_metrics=full_secondary_metrics,
            )

        if search_strategy == "greedy":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._greedy(
                p=p,
                weights=weights,
                objectives=objective,
                show_progress=show_progress,
                threshold_for_coverage=threshold_for_coverage,
                max_value_cutoff=max_value_cutoff,
                full_secondary_metrics=full_secondary_metrics,
            )

        if search_strategy == "grasp":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._grasp(
                p=p,
                objectives=objective,
                weights=weights,
                threshold_for_coverage=threshold_for_coverage,
                num_solutions=grasp_num_solutions,
                alpha=grasp_alpha,
                max_attempts=grasp_max_attempts,
                show_progress=show_progress,
                random_seed=random_seed,
                min_sites_different=grasp_min_sites_different,
                is_minimization=objective != "mclp",
                local_search_chance=grasp_local_search_chance,  # Chance that local searching will happen to improve found solution
                max_swap_count_local_search=grasp_max_swap_count_local_search,
                max_value_cutoff=max_value_cutoff,
                full_secondary_metrics=full_secondary_metrics,
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

        higher_is_better = objective == "mclp"
        outputs_df = pd.DataFrame(outputs)

        if weights and weights.get("cost", 0) > 0:
            outputs_df, score_col = _apply_cost_weighting(
                outputs_df,
                ranking_col=ranking,
                weights=weights,
                higher_is_better=higher_is_better,
            )
        else:
            score_col = ranking

        # _apply_cost_weighting only blends onto its lower-is-better
        # "composite_score" scale when it actually has usable cost data to
        # blend; it no-ops (returns score_col == ranking unchanged) when
        # e.g. every candidate's total_cost is NaN. Assuming "cost weight
        # requested" implies "blended, therefore ascending" was wrong: for
        # mclp (higher-is-better coverage proportion) whose cost weighting
        # silently no-ops, sorting ascending on the raw ranking column
        # ranks the WORST combination first. This is the final cross-
        # strategy sort applied after brute-force/greedy/grasp all
        # return, so it affects every search strategy alike.
        blended = score_col != ranking
        score_ascending = True if blended else not higher_is_better

        solution_df = _add_rank_column(
            outputs_df,
            score_col=score_col,
            tiebreaker_col="weighted_average",
            ascending=[score_ascending, True],
        )

        return SiteSolutionSet(
            solution_df=solution_df,
            site_problem=self,
            objectives=objective,
            n_sites=p,
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
        print("\nTo run a model, use: prob.solve_pmedian(p=3) or similar.")

    def copy(self):
        return copy.deepcopy(self)
