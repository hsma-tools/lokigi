from lokigi.utils import (
    SOLVER_DEFINITIONS,
    SUPPORTED_OBJECTIVES,
    _get_ranking_by_objective,
    _add_rank_column,
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
from .mixins.site_eda import SiteEDAMixin
import copy
from lokigi.problem import _Problem


class SiteProblem(
    _Problem, SiteAttributeMixin, BruteForceMixin, GreedyMixin, GraspMixin, SiteEDAMixin
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
            lower than this value are flagged as 'covered'.

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

        # Ensure exactly one argument is provided out of site_names and site_indices
        if (site_names is None and site_indices is None) or (
            site_names and site_indices
        ):
            raise ValueError(
                "Please provide either 'site_names' or 'site_indices', but not both. "
                "This helps prevent 'off-by-one' errors with numeric site IDs."
            )

        # Ensure travel data is ready
        if self.travel_and_demand_df is None:
            self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)

        try:
            # We need to make sure that we use IDs and names completely consistently throughout.
            # 1. Resolve site_indices to actual Site IDs (names)
            if site_indices is not None:
                # Use .iloc to get the actual ID/Name from the master site list
                resolved_names = self.candidate_sites[
                    self.candidate_sites["canonical_site_index"].isin(site_indices)
                ][self._candidate_sites_candidate_id_col].tolist()
                # print(f"Site indices provided. Resolved names: {resolved_names}")

                if not resolved_names:
                    raise IndexError(
                        f"Indices {site_indices} not found in candidate sites."
                    )
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

            # 3. SMART SORTING:
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
            max_idx = self.travel_and_demand_df.shape[1] - 1
            raise IndexError(
                f"Index out of bounds. Your travel data has indices 0 to {max_idx}. "
                f"You provided indices: {max_idx}"
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

            afi = active_facilities.index
            active_facilities = active_facilities.reset_index()

            # Re-add the demand data
            active_facilities = active_facilities.merge(
                self.demand_data,
                left_on=afi,
                right_on=self._demand_data_id_col,
                how="inner",
            )

        else:
            raise NotImplementedError(
                "Capacitated solving not yet supported. Please rerun with capacitated=False."
            )

        if self.equity_data is not None:
            active_facilities = pd.merge(
                active_facilities,
                self.equity_data,
                left_on=self._joined_demand_travel_df_key_col,
                right_on=self._equity_data_common_col,
            )

        return EvaluatedCombination(
            objective,
            site_indices=final_indices,
            site_names=final_names,
            evaluated_combination_df=active_facilities,
            site_problem=self,
            coverage_threshold=threshold_for_coverage,
        )

    def solve(
        self,
        p: int,
        objectives: str = "p_median",
        capacitated=False,  # Not yet implemented
        search_strategy: Literal["brute-force", "greedy", "grasp"] = "brute-force",
        brute_force_ignore_limit=False,
        show_progress=True,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        max_value_cutoff=None,  # only used for hybrid
        threshold_for_coverage=None,  # used for filtering in mclp or lscp, used for scoring in others
        grasp_num_solutions=5,
        grasp_alpha=0.2,
        grasp_max_attempts="default",
        grasp_min_sites_different=1,
        grasp_local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        grasp_max_swap_count_local_search=10,
        random_seed=42,
        **kwargs,
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
            brute-force search.
        max_value_cutoff : float, optional
            The maximum allowable travel cost. Only applicable for hybrid
            objective models.
        threshold_for_coverage : float, optional
            The distance or time threshold. Used as a hard filter for MCLP
            objectives or as a scoring metric for others.
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
        **kwargs : dict
            Additional arguments passed to the internal solver.

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

        if capacitated:
            raise ValueError(
                "Capacitated modelling not yet supported. Please rerun with `capacitated=False`."
            )

        # Check minimum required information is provided
        if self.travel_matrix is None:
            raise ValueError(
                "No travel matrix or other cost matrix has been provided. Please add this using the .add_travel_matrix() method before running .solve() again."
            )

        if self.demand_data is None:
            self._setup_equal_demand_df()
            if objectives != "mclp":
                warn(
                    "No demand data was provided. Demand from all regions has been assumed to be equal."
                    "If you wish to override this, run .add_demand() to add your site dataframe before running .solve() again."
                    "You can use the .show_demand_format() to see the expected format beforehand."
                )

        if self.candidate_sites is None:
            self._setup_sites_df_from_travel_matrix()
            warn(
                "No candidate site dataframe was given."
                f"\nSites names have been taken from the columns of your travel matrix: {', '.join(self.candidate_sites[self._candidate_sites_candidate_id_col].to_list())}."
                "\nIf you wish to override this, run .add_sites() to add your site dataframe before running .solve() again."
                "\nYou can use the .show_sites_format() to see the expected format beforehand."
            )

        if isinstance(objectives, list) and len(objectives) > 1:
            warn(
                "Multi-objective optimization is coming in a future release."
                f"For now, just your first objective {objectives[0]} has been taken."
            )

        objective = objectives if isinstance(objectives, str) else objectives[0]

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(f"Unsupported objective ({objective}) passed.")

        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model objective ({objective} that doesn't support it.)"
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
            )

        if search_strategy not in ["brute-force", "greedy", "grasp"]:
            raise ValueError(
                f"Unsupported search strategy ({search_strategy}) passed. Only 'brute-force', 'greedy' and 'grasp' are currently supported."
            )

        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model variant ({objective}) that doesn't support it."
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
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
                brute_force_ignore_limit=brute_force_ignore_limit,
                show_progress=show_progress,
                brute_force_keep_best_n=brute_force_keep_best_n,
                brute_force_keep_worst_n=brute_force_keep_worst_n,
                max_value_cutoff=max_value_cutoff,
                grasp_num_solutions=grasp_num_solutions,
                grasp_alpha=grasp_alpha,
                grasp_max_attempts=grasp_max_attempts,
                grasp_min_sites_different=grasp_min_sites_different,
                threshold_for_coverage=threshold_for_coverage,
                random_seed=random_seed,
                grasp_local_search_chance=grasp_local_search_chance,  # Chance that local searching will happen to improve found solution
                grasp_max_swap_count_local_search=grasp_max_swap_count_local_search,
            )
        else:
            raise ValueError(f"Unknown objective '{objective}'.")

    def _solve_pmedian_pcenter_mclp_problem(
        self,
        p: int,
        objective="p_median",
        search_strategy="brute-force",
        show_progress=False,
        brute_force_ignore_limit=False,
        brute_force_keep_best_n=None,
        brute_force_keep_worst_n=None,
        max_value_cutoff=None,
        threshold_for_coverage=None,  # only used for mclp
        grasp_num_solutions=5,
        grasp_alpha=0.2,
        grasp_max_attempts="default",
        grasp_min_sites_different=1,
        grasp_local_search_chance=0.8,  # Chance that local searching will happen to improve found solution
        grasp_max_swap_count_local_search=10,
        random_seed=42,
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
            brute-force results.
        brute_force_keep_worst_n : int, optional
            (Brute Force) The number of lowest-performing combinations to retain in
            brute-force results.
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
                brute_force_ignore_limit=brute_force_ignore_limit,
                show_progress=show_progress,
                brute_force_keep_best_n=brute_force_keep_best_n,
                brute_force_keep_worst_n=brute_force_keep_worst_n,
                rank_best_n_on=ranking,
                max_value_cutoff=max_value_cutoff,
                threshold_for_coverage=threshold_for_coverage,
            )

        if search_strategy == "greedy":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._greedy(
                p=p,
                objectives=objective,
                show_progress=show_progress,
                threshold_for_coverage=threshold_for_coverage,
            )

        if search_strategy == "grasp":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._grasp(
                p=p,
                objectives=objective,
                threshold_for_coverage=threshold_for_coverage,
                num_solutions=grasp_num_solutions,
                alpha=grasp_alpha,
                max_attempts=grasp_max_attempts,
                show_progress=show_progress,
                random_seed=random_seed,
                min_sites_different=grasp_min_sites_different,
                local_search_chance=grasp_local_search_chance,  # Chance that local searching will happen to improve found solution
                max_swap_count_local_search=grasp_max_swap_count_local_search,
            )

        if objective != "mclp":
            solution_df = _add_rank_column(
                pd.DataFrame(outputs),
                score_col=ranking,
                tiebreaker_col="weighted_average",
                ascending=[True, True],
            )
        else:
            solution_df = _add_rank_column(
                pd.DataFrame(outputs),
                score_col=ranking,
                tiebreaker_col="weighted_average",
                ascending=[False, True],
            )

        return SiteSolutionSet(
            solution_df=solution_df,
            site_problem=self,
            objectives=objective,
            n_sites=p,
        )

    def evaluate_n_sites(self, min_sites, max_sites):
        pass

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
