from lokigi.utils import (
    SOLVER_DEFINITIONS,
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
    GEOPANDAS_EXTS,
    _generate_all_combinations,
    _check_crs_match_pref,
    _convert_crs,
    SUPPORTED_OBJECTIVES,
    _get_ranking_by_objective,
)

from lokigi.site_solutions import EvaluatedCombination, SiteSolutionSet

# Data manipulation imports
import pandas as pd
import geopandas

# Plotting imports
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt

# Other imports
from warnings import warn
import numpy as np
from typing import Literal
import heapq
from tqdm.auto import tqdm

# Warn if brute force will be slow
BRUTE_FORCE_WARN_THRESHOLD = 75_000
BRUTE_FORCE_LIMIT = 500_000


class SiteProblem:
    """
    SiteProblem.solve_pmedian.__doc__ = f'''
    {SOLVER_DEFINITIONS['p_median']['goal']}

    Healthcare Context:
    {SOLVER_DEFINITIONS['p_median']['healthcare_context']}

    Trade-offs:
    {SOLVER_DEFINITIONS['p_median']['trade_off']}
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

        self.travel_matrix = None  # Travel time/distance matrix
        self._travel_matrix_type = None
        self._travel_matrix_source_col = None
        self._travel_matrix_unit = None

        self.region_geometry_layer = None
        self._region_geometry_layer_type = None
        self._region_geometry_layer_common_col = None

        # self.baseline_sites = None  # Current existing clinics
        # self._baseline_sites_type = None

        self.travel_and_demand_df = None

        if debug_mode:
            self._verbose = True
        else:
            self._verbose = False

    @staticmethod
    def show_demand_format():
        """Prints the expected structure for the demand DataFrame."""
        print("\n--- Expected Demand DataFrame Format ---")
        print("Note: Each row represents a unique demand location (e.g., LSOA).")
        print(f"{'site_id_col':<15} | {'demand_col':<10}")
        print("-" * 30)
        print(f"{'LSOA 1':<15} | {'25':<10}")
        print(f"{'LSOA 2':<15} | {'15':<10}")
        print(f"{'...':<15} | {'...':<10}")
        print("----------------------------------------\n")

    @staticmethod
    def show_travel_format():
        """Prints the expected structure for the travel/cost matrix DataFrame."""
        print("\n--- Expected Travel/Cost DataFrame Format ---")
        print("Note: Rows are sources, columns are destinations.")
        print(f"{'source_id':<15} | {'dest_1':<15} | {'dest_2':<15}")
        print("-" * 50)
        print(f"{'source_1':<15} | {'22.6':<15} | {'16.3':<15}")
        print(f"{'source_2':<15} | {'15.1':<15} | {'17.1':<15}")
        print(f"{'...':<15} | {'...':<15} | {'...':<15}")
        print("--------------------------------------------\n")
        print("For example, if using LSOAs, your dataframe might look like this:")
        print(f"{'source_id':<15} | {'E01000259':<15} | {'E01000314':<15}")
        print("-" * 50)
        print(f"{'Brighton and Hove 027E':<15} | {'22.6':<15} | {'16.3':<15}")
        print(f"{'Brighton and Hove 005C':<15} | {'15.1':<15} | {'17.1':<15}")
        print(f"{'...':<15} | {'...':<15} | {'...':<15}")
        print("--------------------------------------------\n")
        print("Or if you've defined your site names, it might look like this:")
        print(f"{'source_id':<15} | {'Site 1':<15} | {'Site 1':<15}")
        print("-" * 50)
        print(f"{'Brighton and Hove 027E':<15} | {'22.6':<15} | {'16.3':<15}")
        print(f"{'Brighton and Hove 005C':<15} | {'15.1':<15} | {'17.1':<15}")
        print(f"{'...':<15} | {'...':<15} | {'...':<15}")
        print("--------------------------------------------\n")

    def add_demand(self, demand_df, demand_col, location_id_col, skip_cols=None):
        """
        Adds a dataframe or geodataframe containing the demand observed
        to the SiteProblem object.

        df: Pandas DataFrame or Geopandas Geodataframe
        Validates CRS and aligns patient data."""
        loaded_df, df_type = _load_spatial_or_tabular_data(
            demand_df, skip_cols=skip_cols
        )

        _validate_columns(
            df=loaded_df,
            col_names=[
                demand_col,
                location_id_col,
            ],
            msg_template=(
                "It looks like your demand data is missing these columns: {missing}. "
                "We found these instead: {available}. Please double-check the column names "
                "you are passing to the .add_demand() method."
            ),
        )

        self.demand_data = loaded_df
        self._demand_data_type = df_type
        self._demand_data_demand_col = demand_col
        self._demand_data_id_col = location_id_col

    def show_demand(self):
        return self.demand_data

    def add_region_geometry_layer(self, region_geometry_df, common_col):

        loaded_df, df_type = _load_spatial_or_tabular_data(region_geometry_df)
        if df_type != "geopandas":
            raise TypeError(
                "Please pass in a created geodataframe or the path to a source of geographic data."
                "If passing a path to geographic data as a string, paths with extensions"
                f"{GEOPANDAS_EXTS} will be automatically read in as geopandas dataframes."
            )

        if not _check_crs_match_pref(loaded_df, self.preferred_crs):
            loaded_df = _convert_crs(loaded_df, self.preferred_crs)

        self.region_geometry_layer = loaded_df
        self._region_geometry_layer_type = df_type
        self._region_geometry_layer_common_col = common_col

    def show_region_geometry_layer(self):
        return self.region_geometry_layer

    def plot_region_geometry_layer(
        self, interactive=False, plot_demand=False, **kwargs
    ):
        if self.region_geometry_layer is None:
            raise ValueError(
                "No region geometry layer has been initialised."
                "Please run `.add_region_geometry_layer()` first."
            )
        if plot_demand and self.demand_data is None:
            raise ValueError(
                "Cannot plot demand when no demand data is present."
                "Please run `.add_demand()` first or change the `plot_demand` parameter to False."
            )

        if plot_demand:
            plotting_df = self.region_geometry_layer.merge(
                self.demand_data,
                left_on=self._region_geometry_layer_common_col,
                right_on=self._demand_data_id_col,
            )

            if interactive:
                m = plotting_df.explore(
                    column=self._demand_data_demand_col,  # make choropleth based on demand col
                    tooltip=self._demand_data_demand_col,  # show demand col value in tooltip (on hover)
                    popup=True,  # show all values in popup (on click)
                    cmap="Blues",  # use "Blues" matplotlib colormap
                    style_kwds=dict(color="black"),
                    tiles="CartoDB positron",
                    **kwargs,
                )

                return m
            else:
                fig = plotting_df.plot(
                    column=self._demand_data_demand_col, legend=True, **kwargs
                )

                return fig

        if interactive:
            m = self.region_geometry_layer.explore(tiles="CartoDB positron", **kwargs)
            return m
        else:
            fig = self.region_geometry_layer.plot(**kwargs)
            return fig

    def add_sites(
        self,
        candidate_site_df,
        candidate_id_col,
        required_sites_col=None,
        geometry_col="geometry",
        vertical_geometry_col="lat",
        horizontal_geometry_col="long",
        crs=None,
        capacity_col=None,
        skip_cols=None,
    ):
        """Validates CRS and identifies potential new sites."""
        loaded_df, df_type = _load_spatial_or_tabular_data(
            candidate_site_df, skip_cols=skip_cols
        )

        col_list = [candidate_id_col]
        if capacity_col is not None:
            col_list.extend([capacity_col])

        if df_type == "geopandas":
            col_list.extend([geometry_col])
            _validate_columns(
                df=loaded_df,
                col_names=col_list,
                msg_template=(
                    "It looks like your candidate site data is missing these columns: {missing}. "
                    "We found these instead: {available}. Please double-check the column names you are "
                    "passing in to the .add_candidates() method and try running this method again."
                ),
            )
        else:
            col_list.extend([horizontal_geometry_col, vertical_geometry_col])
            _validate_columns(
                df=loaded_df,
                col_names=col_list,
                msg_template=(
                    "It looks like your candidate site data is missing these columns: {missing}. "
                    "We found these instead: {available}. Please double-check the column names you are "
                    "passing in to the .add_candidates() method and try running this method again."
                ),
            )

        if df_type != "geopandas":
            # If CRS is not provided, make a good guess
            if crs is None:
                crs = _guess_crs(
                    loaded_df,
                    horizontal_geometry_col,
                    vertical_geometry_col,
                    verbose=self._verbose,
                )

                if self.preferred_crs is None:
                    self.preferred_crs = crs

            loaded_df = geopandas.GeoDataFrame(
                data=loaded_df,
                geometry=geopandas.points_from_xy(
                    loaded_df[horizontal_geometry_col], loaded_df[vertical_geometry_col]
                ),
                crs=crs,
            )

        if not _check_crs_match_pref(loaded_df, self.preferred_crs):
            loaded_df = _convert_crs(loaded_df, target_crs=self.preferred_crs)

        loaded_df = loaded_df.reset_index(drop=False, names="index")

        self.candidate_sites = loaded_df
        self._candidate_sites_type = df_type
        self._candidate_sites_candidate_id_col = candidate_id_col
        self._candidate_sites_geometry_col = geometry_col
        self._candidate_sites_capacity_col = capacity_col
        self._candidate_sites_required_sites_col = required_sites_col
        self.total_n_sites = len(self.candidate_sites)

    def show_sites(self):
        return self.candidate_sites

    def plot_sites(self, add_basemap=True, show_labels=True, interactive=False):
        """
        Adds a quick plot
        """
        if not interactive or self._candidate_sites_type == "pandas":
            ax = self.candidate_sites.plot()

            if show_labels:
                texts = []
                for x, y, label in zip(
                    self.candidate_sites.geometry.x,
                    self.candidate_sites.geometry.y,
                    self.candidate_sites[self._candidate_sites_candidate_id_col],
                ):
                    wrapped_label = textwrap.fill(label, 15).title()
                    texts.append(plt.text(x, y, wrapped_label))

                adjust_text(texts, force_explode=(0.05, 0.05))

            if add_basemap:
                cx.add_basemap(ax, crs=self.candidate_sites.crs.to_string())
        else:
            m = self.candidate_sites.explore()
            return m

    # def add_baseline(self, gdf, id_col):
    #     """Loads 'status quo' sites to calculate the starting benchmark."""
    #     pass

    def add_travel_matrix(
        self,
        travel_matrix_df,
        source_col,
        skip_cols=None,
        unit=None,
        from_unit=None,
        to_unit=None,
    ):
        """Ensures the matrix indices match the demand/candidate IDs."""
        loaded_df, df_type = _load_spatial_or_tabular_data(
            travel_matrix_df, skip_cols=skip_cols
        )

        _validate_columns(
            df=loaded_df,
            col_names=[source_col],
            msg_template=(
                "It looks like your travel matrix data is missing these columns: {missing}. "
                "We found these instead: {available}. Please double-check the column names "
                "you are passing to the .add_travel_matrix() method."
            ),
        )

        conversion = {
            ("seconds", "minutes"): 1 / 60,
            ("seconds", "hours"): 1 / 3600,
            ("minutes", "seconds"): 60,
            ("minutes", "hours"): 1 / 60,
            ("hours", "seconds"): 3600,
            ("hours", "minutes"): 60,
        }

        if from_unit is not None and to_unit is not None:
            factor = conversion[(from_unit, to_unit)]
            num_cols = loaded_df.select_dtypes(include="number").columns
            loaded_df.loc[:, num_cols] *= factor
            self._travel_matrix_unit = to_unit
        elif from_unit is None and to_unit is not None:
            self._travel_matrix_unit = to_unit
        elif from_unit is not None and to_unit is None:
            self._travel_matrix_unit = from_unit
        else:
            self._travel_matrix_unit = unit

        self.travel_matrix = loaded_df
        self._travel_matrix_source_col = source_col

    def show_travel_matrix(self):
        return self.travel_matrix

    def _create_joined_demand_travel_df(self, index_col):
        # If one is a geopandas dataframe, put that first in the merge call so that the
        # output object will also be a geodataframe
        if self._demand_data_type == "geopandas":
            self.travel_and_demand_df = pd.merge(
                self.demand_data,
                self.travel_matrix,
                left_on=self._demand_data_id_col,
                right_on=self._travel_matrix_source_col,
                how="inner",
            ).set_index(index_col)

        else:
            self.travel_and_demand_df = pd.merge(
                self.travel_matrix,
                self.demand_data,
                left_on=self._travel_matrix_source_col,
                right_on=self._demand_data_id_col,
                how="inner",
            ).set_index(index_col)

            self.travel_and_demand_df = self.travel_and_demand_df

        if len(self.travel_and_demand_df) == 0:
            raise KeyError(
                "Warning: merging the travel matrix and demand data has failed."
                f"This may be because there are no common values found in the {self._travel_matrix_source_col}"
                f"(sample values: {self.travel_matrix.head(5)[self._travel_matrix_source_col]})"
                f"column in the travel dataframe and the {self._demand_data_id_col} column in the"
                f"demand dataframe (sample values: {self.demand_data.head(5)[self._demand_data_id_col]})"
            )

    def evaluate_single_solution_single_objective(
        self,
        objective: str = "p_median",
        site_names=None,
        site_indices=None,
        capacitated=False,
        threshold_for_coverage=None,
    ):
        """
        Evaluates a solution. User must provide either site_names OR site_indices.
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

        # Resolve to indices based on the chosen input
        if site_names:
            try:
                # Use .get_indexer to find positions of a list of labels
                # This returns integer positions for the provided names
                resolved_indices = self.travel_and_demand_df.columns.get_indexer(
                    site_names
                )

                # Check if any names were not found (get_indexer returns -1 for missing)
                if -1 in resolved_indices:
                    missing = [
                        site_names[i]
                        for i, idx in enumerate(resolved_indices)
                        if idx == -1
                    ]
                    raise KeyError(
                        f"The following site names were not found: {missing}"
                    )
            except Exception as e:
                raise ValueError(f"Error mapping site names: {e}")

        else:
            # User provided site_indices directly
            resolved_indices = site_indices

        # Filter and calculate
        try:
            # We use .iloc because we now have guaranteed integer positions
            active_facilities = self.travel_and_demand_df.iloc[
                :, resolved_indices
            ].copy()
        except IndexError:
            max_idx = self.travel_and_demand_df.shape[1] - 1
            raise IndexError(
                f"Index out of bounds. Your travel data has indices 0 to {max_idx}. "
                f"You provided indices: {site_indices}"
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
            raise ValueError(
                "Capacitated solving not yet supported. Please rerun with capacitated=False."
            )

        return EvaluatedCombination(
            objective,
            site_indices=resolved_indices,
            site_names=site_names,
            evaluated_combination_df=active_facilities,
            site_problem=self,
            coverage_threshold=threshold_for_coverage,
        )

    def _setup_equal_demand_df(self):
        demand_data_temp = pd.DataFrame(
            self.travel_matrix[self._travel_matrix_source_col],
            columns=[self._travel_matrix_source_col],
        )
        demand_data_temp["n"] = 1

        self.demand_data = demand_data_temp
        self._demand_data_type = "pandas"
        self._demand_data_id_col = self._travel_matrix_source_col
        self._demand_data_demand_col = "n"

    def _setup_sites_df_from_travel_matrix(self):
        sites_df_temp = pd.DataFrame(
            self.travel_matrix.columns.T.drop(self._demand_data_id_col),
            columns=["site"],
        )

        sites_df_temp = sites_df_temp.reset_index(drop=False, names="index")

        self.candidate_sites = sites_df_temp
        self._candidate_sites_type = "pandas"
        self._candidate_sites_candidate_id_col = "site"
        self._candidate_sites_vertical_col = None
        self._candidate_sites_horizontal_col = None
        self._candidate_sites_capacity_col = None
        self.total_n_sites = len(self.candidate_sites)

    def solve(
        self,
        p: int,
        objectives: str = "p_median",
        capacitated=False,  # Not yet implemented
        search_strategy: Literal["brute-force", "greedy"] = "brute-force",
        ignore_brute_force_limit=False,
        show_brute_force_progress=False,
        keep_best_n=None,
        keep_worst_n=None,
        max_value_cutoff=None,  # only used for hybrid
        threshold_for_coverage=None,  # only used for mclp or lscp
        **kwargs,
    ):

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

        # if search_strategy not in ["brute-force", "greedy", "local"]:
        #     raise ValueError(
        #         f"Unsupported search strategy ({search_strategy}) passed. Please choose from 'brute-force', 'greedy', or 'local'."
        #     )
        if search_strategy not in ["brute-force", "greedy"]:
            raise ValueError(
                f"Unsupported search strategy ({search_strategy}) passed. Only 'brute-force' is currently supported."
            )

        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model variant ({objective}) that doesn't support it."
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
            )

        if objective in ["p_median", "p_center", "simple_p_median"]:
            return self._solve_pmedian_pcenter_mclp_problem(
                p,
                search_strategy=search_strategy,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                objective=objective,
                threshold_for_coverage=threshold_for_coverage,
            )
        elif objective in ["hybrid_p_median", "hybrid_simple_p_median"]:
            return self._solve_pmedian_pcenter_mclp_problem(
                p,
                search_strategy=search_strategy,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                objective=objective,
                max_value_cutoff=max_value_cutoff,
                threshold_for_coverage=threshold_for_coverage,
            )
        elif objective in ["mclp"]:
            return self._solve_pmedian_pcenter_mclp_problem(
                p,
                search_strategy=search_strategy,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                objective=objective,
                threshold_for_coverage=threshold_for_coverage,
            )
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Currently supported: 'p_median'."
            )

    def _brute_force(
        self,
        p: int,
        objectives,
        ignore_brute_force_limit: bool = False,
        show_brute_force_progress: bool = False,
        keep_best_n=None,
        keep_worst_n=None,
        rank_best_n_on="weighted_average",
        max_value_cutoff=None,
        threshold_for_coverage=None,
    ):

        if keep_best_n is not None:
            top_n_heap = []  # To store the smallest scores (best)
        if keep_worst_n is not None:
            bottom_n_heap = []  # To store the largest scores (worst)

        possible_combinations = _generate_all_combinations(
            n_facilities=self.total_n_sites, p=p, site_problem=self
        )

        if len(possible_combinations) > BRUTE_FORCE_LIMIT:
            if not ignore_brute_force_limit:
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

        if show_brute_force_progress:
            possible_combinations = tqdm(possible_combinations)

        for possible_solution in possible_combinations:
            if keep_best_n is None and keep_worst_n is None:
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
                metrics = self.evaluate_single_solution(
                    site_indices=possible_solution, objectives=objectives
                ).return_solution_metrics()

                score = metrics[rank_best_n_on]
                max_value = metrics["max"]

                if max_value_cutoff is None or (
                    max_value_cutoff is not None and max_value <= max_value_cutoff
                ):
                    if keep_best_n is not None:
                        if len(top_n_heap) < keep_best_n and max_value <= max:
                            heapq.heappush(top_n_heap, (-score, metrics))
                        elif -score > top_n_heap[0][0]:
                            heapq.heapreplace(top_n_heap, (-score, metrics))

                    # --- Logic for Bottom N (Largest Scores) ---
                    # Standard Min-Heap to keep the largest values
                    if keep_worst_n is not None:
                        if len(bottom_n_heap) < keep_best_n:
                            heapq.heappush(bottom_n_heap, (score, metrics))
                        elif score > bottom_n_heap[0][0]:
                            heapq.heapreplace(bottom_n_heap, (score, metrics))

        if keep_best_n is None and keep_worst_n is None:
            return outputs
        else:
            # Reconstruct the 'outputs' list
            # Extract dictionaries from heaps and sort them
            if keep_best_n is not None:
                best_list = [
                    item[1]
                    for item in sorted(top_n_heap, key=lambda x: x[0], reverse=True)
                ]
            if keep_worst_n is not None:
                worst_list = [
                    item[1] for item in sorted(bottom_n_heap, key=lambda x: x[0])
                ]

            if keep_best_n is not None and keep_worst_n is None:
                return best_list
            elif keep_worst_n is not None and keep_best_n is None:
                return worst_list
            else:
                return best_list + worst_list

    def _solve_pmedian_pcenter_mclp_problem(
        self,
        p: int,
        objective="p_median",
        search_strategy="brute-force",
        ignore_brute_force_limit=False,
        show_brute_force_progress=False,
        keep_best_n=None,
        keep_worst_n=None,
        max_value_cutoff=None,
        threshold_for_coverage=None,  # only used for mclp
    ):

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                "Unsupported objective passed to _solve_pmedian_pcenter_mclp_problem. Please contact a developer."
            )

        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model objective ({objective} that doesn't support it.)"
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
            )

        ranking = _get_ranking_by_objective(objective=objective)

        if objective in ["hybrid_p_median", "hybrid_simple_p_median"]:
            max_value_cutoff = max_value_cutoff
        else:
            max_value_cutoff = None

        if search_strategy not in ["brute-force", "greedy"]:
            raise ValueError(f"Approach {search_strategy} not yet supported.")
        if search_strategy == "brute-force":
            outputs = self._brute_force(
                p=p,
                objectives=objective,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                rank_best_n_on=ranking,
                max_value_cutoff=max_value_cutoff,
                threshold_for_coverage=threshold_for_coverage,
            )

            if objective != "mclp":
                return SiteSolutionSet(
                    solution_df=pd.DataFrame(outputs).sort_values(
                        [ranking, "weighted_average"]
                    ),
                    site_problem=self,
                    objectives=objective,
                    n_sites=p,
                )

            else:
                return SiteSolutionSet(
                    solution_df=pd.DataFrame(outputs).sort_values(
                        [ranking, "weighted_average"], ascending=[False, True]
                    ),
                    site_problem=self,
                    objectives=objective,
                    n_sites=p,
                )

        if search_strategy == "greedy":
            # Note that coverage threshold will only be used for assessing coverage, not for
            # filtering out solutions, when using greedy search strategy
            outputs = self._greedy(
                p=p, objectives=objective, threshold_for_coverage=threshold_for_coverage
            )

            return SiteSolutionSet(
                solution_df=pd.DataFrame(outputs).sort_values(
                    [ranking, "weighted_average"]
                ),
                site_problem=self,
                objectives=objective,
                n_sites=p,
            )

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
            print(f"Loop {i}")
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

            print(f"Best combination for {i} sites: {best_indices}")

        best_solution_metrics = self.evaluate_single_solution_single_objective(
            site_indices=best_indices,
            objective=objectives,
            threshold_for_coverage=threshold_for_coverage,
        ).return_solution_metrics()

        return [best_solution_metrics]

    def evaluate_n_sites(self, min_sites, max_sites):
        pass

    # def solve_lscp(self, p, num_options=10, capacitated=False):
    #     """
    #     """
    #     pass

    # def solve_mclp(self, p, num_options=10, capacitated=False):
    #     """
    #     """
    #     pass

    # def solve_pcentre(self, p, num_options=10, capacitated=False):
    #     """
    #     """
    #     pass

    # def solve_equity_efficiency(self, p, num_options=10):
    #     """
    #     Balances average travel time (efficiency) against worst-case travel
    #     time (equity). Returns a Pareto front of trade-off solutions.
    #     """
    #     return self.solve(p=p, objectives=["p_median", "p_centre"],
    #                     num_options=num_options)

    def describe_models(self, available_only=True):
        """Prints a menu of available optimization strategies for healthcare."""
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

    def _validate_capacity_constraint(self):
        total_demand = self.demand_data[self._demand_data_demand_col].sum()
        total_capacity = self.candidate_sites[self._candidate_sites_capacity_col].sum()

        if total_demand > total_capacity:
            raise ValueError(
                f"Insufficient Capacity! Your region has {total_demand} patients "
                f"but only {total_capacity} total slots. You need to add more "
                "candidate sites or increase 'p'."
            )
