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
)

import pandas as pd
import geopandas
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
import math
from typing import Literal
import heapq

import plotly.express as px

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

    Parameters:
    -----------
    p : int
        Number of facilities to locate.
    num_options : int
        Number of diverse candidate solutions to return.
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
                plotting_df.plot(
                    column=self._demand_data_demand_col, legend=True, **kwargs
                )

        if interactive:
            m = self.region_geometry_layer.explore(tiles="CartoDB positron", **kwargs)
            return m
        else:
            self.region_geometry_layer.plot(**kwargs)

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
        self, travel_matrix_df, source_col, skip_cols=None, from_unit=None, to_unit=None
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

        if from_unit and to_unit:
            factor = conversion[(from_unit, to_unit)]
            num_cols = loaded_df.select_dtypes(include="number").columns
            loaded_df.loc[:, num_cols] *= factor

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

    def evaluate_single_solution(
        self,
        objectives: str | list[str] = "p_median",
        site_names=None,
        site_indices=None,
    ):
        """
        Evaluates a solution. User must provide either site_names OR site_indices.
        """
        # Check for valid objectives
        if isinstance(objectives, list) and len(objectives) > 1:
            warn(
                "Multi-objective optimization is coming in a future release."
                f"For now, just your first objective {objectives[0]} has been taken."
            )

        objective = objectives if isinstance(objectives, str) else objectives[0]

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

        if objective in [
            "p_median",
            "p_center",
            "simple_p_median",
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            # Assume travel to closest facility
            active_facilities["min_cost"] = active_facilities.min(axis=1)

            afi = active_facilities.index
            active_facilities = active_facilities.reset_index()

            # Re-add the demand data
            active_facilities = active_facilities.merge(
                self.demand_data,
                left_on=afi,
                right_on=self._demand_data_id_col,
                how="inner",
            )

            return EvaluatedCombination(
                objective,
                site_indices=resolved_indices,
                site_names=site_names,
                evaluated_combination_df=active_facilities,
                site_problem=self,
            )

        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Currently supported: {SUPPORTED_OBJECTIVES.join(', ')}."
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
        search_strategy: Literal["brute-force", "greedy", "local"] = "brute-force",
        ignore_brute_force_limit=False,
        show_brute_force_progress=False,
        keep_best_n=None,
        keep_worst_n=None,
        max_value_cutoff=None,  # only used for hybrid
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
        if search_strategy not in ["brute-force"]:
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
            return self._solve_pmedian_pcenter_problem(
                p,
                search_strategy=search_strategy,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                objective=objective,
            )
        elif objective in ["hybrid_p_median", "hybrid_simple_p_median"]:
            return self._solve_pmedian_pcenter_problem(
                p,
                search_strategy=search_strategy,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                objective=objective,
                max_value_cutoff=max_value_cutoff,
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
                single_solution_metrics = self.evaluate_single_solution(
                    site_indices=possible_solution, objectives=objectives
                ).return_solution_metrics()
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

    def _solve_pmedian_pcenter_problem(
        self,
        p: int,
        objective="p_median",
        search_strategy="brute-force",
        ignore_brute_force_limit=False,
        show_brute_force_progress=False,
        keep_best_n=None,
        keep_worst_n=None,
        max_value_cutoff=None,
    ):

        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError(
                "Unsupported objective passed to _solve_pmedian_pcenter_problem. Please contact a developer."
            )
        if max_value_cutoff is not None and objective not in [
            "hybrid_p_median",
            "hybrid_simple_p_median",
        ]:
            raise ValueError(
                f"A max value cutoff of {max_value_cutoff} has been provided for a model objective ({objective} that doesn't support it.)"
                "Please rerun with hybrid_p_median or hybrid_simple_p_median."
            )

        if search_strategy != "brute-force":
            raise ValueError(f"Approach {search_strategy} not yet supported.")
        if search_strategy == "brute-force":
            if objective in ["p_median", "hybrid_p_median"]:
                ranking = "weighted_average"
            elif objective in ["simple_p_median", "hybrid_simple_p_median"]:
                ranking = "unweighted_average"
            elif objective in ["p_center"]:
                ranking = "max"

            if objective in ["hybrid_p_median", "hybrid_simple_p_median"]:
                max_value_cutoff = max_value_cutoff
            else:
                max_value_cutoff = None

            outputs = self._brute_force(
                p=p,
                objectives=objective,
                ignore_brute_force_limit=ignore_brute_force_limit,
                show_brute_force_progress=show_brute_force_progress,
                keep_best_n=keep_best_n,
                keep_worst_n=keep_worst_n,
                rank_best_n_on=ranking,
                max_value_cutoff=max_value_cutoff,
            )

            return SiteSolutionSet(
                solution_df=pd.DataFrame(outputs).sort_values(
                    [ranking, "weighted_average"]
                ),
                site_problem=self,
                objectives=objective,
            )

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


class EvaluatedCombination:
    def __init__(
        self,
        solution_type,
        site_names,
        site_indices,
        evaluated_combination_df,
        site_problem,
    ):
        self.solution_type = solution_type
        self.site_names = site_names
        self.site_indices = site_indices
        self.evaluated_combination_df = evaluated_combination_df
        self.site_problem = site_problem
        self.weighted_average = np.average(
            self.evaluated_combination_df["min_cost"],
            weights=self.evaluated_combination_df[
                self.site_problem._demand_data_demand_col
            ],
        )
        self.unweighted_average = np.average(self.evaluated_combination_df["min_cost"])
        self.percentile_90th = np.percentile(
            self.evaluated_combination_df["min_cost"], q=90
        )
        self.max = np.max(self.evaluated_combination_df["min_cost"])

    def show_result_df(self):
        return self.evaluated_combination_df

    def return_solution_metrics(self):
        # Return weighted average
        return {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "weighted_average": self.weighted_average,
            "unweighted_average": self.unweighted_average,
            "90th_percentile": self.percentile_90th,
            "max": self.max,
            "problem_df": self.evaluated_combination_df,
        }


class SiteSolutionSet:
    def __init__(self, solution_df, site_problem, objectives):
        self.solution_df = solution_df.reset_index(drop=True)
        self.site_problem = site_problem
        self.objectives = objectives

    def show_solutions(self, rounding=2):
        return round(self.solution_df, rounding)

    def return_best_combination_details(self, rank_on=None, top_n=1):
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on).head(top_n).reset_index()
        else:
            return self.solution_df.head(top_n).reset_index()

    def return_best_combination_site_indices(self, rank_on=None):
        if rank_on is not None:
            return (
                self.solution_df.sort_values(rank_on)
                .head(1)["site_indices"]
                .reset_index()[0]
            )
        else:
            return self.solution_df.head(1)["site_indices"].reset_index()[0]

    def return_best_combination_site_names(self, rank_on=None):
        if rank_on is not None:
            return (
                self.solution_df.sort_values(rank_on)
                .head(1)["site_names"]
                .reset_index()[0]
            )
        else:
            return self.solution_df.head(1)["site_names"].reset_index()[0]

    def plot_travel_time_distribution(
        self,
        top_n=1,
        rank_on=None,
        secondary_ranking="max",
        title="default",
        height=None,
        height_per_plot=250,
        compare_to_best=False,
        **kwargs,
    ):
        if rank_on is not None:
            solutions_filtered = (
                self.solution_df.sort_values([rank_on, secondary_ranking])
                .reset_index(drop=True)
                .head(top_n)
            )
        else:
            solutions_filtered = self.solution_df.reset_index(drop=True).head(top_n)

        dfs = []
        if compare_to_best:
            best_df = solutions_filtered.head(1)["problem_df"][0]

        for index, row in solutions_filtered.iterrows():
            df = row["problem_df"].copy()
            df["site_indices"] = str(row["site_indices"])
            df["weighted_average"] = row["weighted_average"]
            df["unweighted_average"] = row["unweighted_average"]
            df["max"] = row["max"]
            df["90th_percentile"] = row["90th_percentile"]
            if compare_to_best:
                df["min_cost_diff"] = df["min_cost"] - best_df["min_cost"].values

            dfs.append(df)

        dfs = pd.concat(dfs)
        dfs["label"] = (
            "Sites: "
            + dfs["site_indices"].astype(str)
            + " | Weighted Average: "
            + dfs["weighted_average"].round(2).astype(str)
            + " |Unweighted Average: "
            + dfs["unweighted_average"].round(2).astype(str)
            + " | 90th percentile: "
            + dfs["90th_percentile"].round(2).astype(str)
            + " | Max: "
            + dfs["max"].round(2).astype(str)
        )
        fig = px.histogram(
            dfs,
            x="min_cost_diff" if compare_to_best else "min_cost",
            facet_row="label",
            nbins=30,
            histnorm="probability density",
            height=height,
            **kwargs,
        )

        fig.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace("label=", ""),
                textangle=0,
                x=a.x - 0.98,  # small shift left
                y=a.y + 0.092,  # move above subplot
                xanchor="left",
                yanchor="bottom",
            )
        )

        if title == "default":
            if rank_on is not None:
                fig.update_layout(
                    title=f"Distribution of Travel Times (Top {top_n} Solutions by {rank_on.replace('_', ' ').title()})"
                )
            else:
                fig.update_layout(
                    title=f"Distribution of Travel Times (Top {top_n} Solutions: {self.objectives.replace('_', ' ').title()})"
                )
        else:
            fig.update_layout(title=title)

        if compare_to_best:
            fig.add_vline(
                x=0,
                line_color="black",
                line_width=2,
                annotation_text="Best",
            )
        else:
            for i, (_, row) in enumerate(solutions_filtered.iterrows()):
                fig.add_vline(
                    x=row["weighted_average"],
                    line_dash="dash",
                    annotation_text="WA",
                    row=i + 1,
                    col=1,
                )

                fig.add_vline(
                    x=row["90th_percentile"],
                    line_dash="dot",
                    annotation_text="P90",
                    row=i + 1,
                    col=1,
                )

        if height is None:
            fig_height = max(300, height_per_plot * top_n)

            fig.update_layout(height=fig_height)

            # fig.update_layout(
            #     margin=dict(t=80, b=40, l=40, r=40),
            # )

        return fig

    def summary_table(self):
        pass

    def plot_best_combination(
        self,
        rank_on=None,
        title=None,
        show_all_locations=True,
        cmap="Blues",
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
    ):
        if self.site_problem.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        if rank_on is not None:
            solution = self.solution_df.sort_values(rank_on).head().reset_index()
        else:
            solution = self.solution_df.head().reset_index()

        nearest_site_travel_gdf = pd.merge(
            self.site_problem.region_geometry_layer,
            solution["problem_df"][0],
            left_on=self.site_problem._region_geometry_layer_common_col,
            right_on=self.site_problem._demand_data_id_col,
        )

        ax = nearest_site_travel_gdf.plot(
            "min_cost",
            legend=True,
            cmap=cmap,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            figsize=(12, 6),
        )

        selected_sites = self.site_problem.candidate_sites.iloc[
            solution.site_indices.iloc[0]
        ]

        if show_all_locations:
            all_site_points = self.site_problem.candidate_sites.plot(
                ax=ax, color=unchosen_site_colour, markersize=30, alpha=0.3
            )

        selected_site_points = selected_sites.plot(
            ax=ax, color=chosen_site_colour, markersize=60
        )

        cx.add_basemap(
            ax,
            crs=nearest_site_travel_gdf.crs.to_string(),
        )

        for x, y, label in zip(
            selected_sites.geometry.x,
            selected_sites.geometry.y,
            selected_sites[self.site_problem._candidate_sites_candidate_id_col],
        ):
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(10, 3),
                textcoords="offset points",
                bbox=dict(facecolor="white"),
            )

        ax.axis("off")

        if title is not None:
            plt.title(title)

        return ax

    def plot_n_best_combinations(
        self,
        n_best=10,
        rank_on=None,
        title=None,
        subplot_title="default",
        show_all_locations=True,
        cmap="Blues",
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
        n_cols=None,
        n_rows=None,
    ):
        if n_best > len(self.solution_df):
            n_best = len(self.solution_df)

        if n_cols is None and n_rows is None:
            max_cols = 5
            ncols = min(n_best, max_cols)
            nrows = math.ceil(n_best / ncols)

        elif n_cols is None and n_rows is not None:
            nrows = n_rows
            ncols = min(n_best, max_cols)

        else:
            ncols = n_cols
            nrows = math.ceil(n_best / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

        # flatten axs in case it's a 2D array
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()

        if self.site_problem.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        if rank_on is not None:
            sorted_df = self.solution_df.sort_values(rank_on).reset_index().head(n_best)
        else:
            sorted_df = self.solution_df.reset_index().head(n_best)

        # Calculate global color scale boundaries
        global_vmin = min(df["min_cost"].min() for df in sorted_df["problem_df"])
        global_vmax = max(df["min_cost"].max() for df in sorted_df["problem_df"])

        for i, ax in enumerate(fig.axes):
            solution = sorted_df.iloc[[i]]
            solution_df = solution["problem_df"].values[0]

            nearest_site_travel_gdf = pd.merge(
                self.site_problem.region_geometry_layer,
                solution_df,
                left_on=self.site_problem._region_geometry_layer_common_col,
                right_on=self.site_problem._demand_data_id_col,
            )

            ax = nearest_site_travel_gdf.plot(
                "min_cost",
                legend=False,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
                ax=ax,
                vmin=global_vmin,
                vmax=global_vmax,
            )

            selected_sites = self.site_problem.candidate_sites.iloc[
                solution.site_indices.iloc[0]
            ]

            if show_all_locations:
                all_site_points = self.site_problem.candidate_sites.plot(
                    ax=ax, color=unchosen_site_colour, markersize=30, alpha=0.3
                )

            selected_site_points = selected_sites.plot(
                ax=ax, color=chosen_site_colour, markersize=60
            )

            cx.add_basemap(
                ax,
                crs=nearest_site_travel_gdf.crs.to_string(),
            )

            for x, y, label in zip(
                selected_sites.geometry.x,
                selected_sites.geometry.y,
                selected_sites[self.site_problem._candidate_sites_candidate_id_col],
            ):
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(10, 3),
                    textcoords="offset points",
                    bbox=dict(facecolor="white"),
                )

            ax.axis("off")

            def safe_evaluate_title(subplot_title):
                try:
                    return eval(f"f{repr(subplot_title)}", {}, {"solution": solution})
                except Exception:
                    # fallback: treat as literal string
                    return subplot_title

            if subplot_title is not None:
                if subplot_title == "default":
                    ax.set_title(
                        f"Weighted Average: {solution['weighted_average'].values[0]:.1f} | Maximum: {solution['max'].values[0]:.1f}"
                    )
                else:
                    ax.set_title(safe_evaluate_title(subplot_title))
            if title is not None:
                ax.set_title(title)

        # Create a single colorbar based on the global scale and chosen colormap
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        )
        sm._A = []  # Empty array for the scalar mappable

        # Add the colorbar to the figure
        fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.04, label="Min Cost")

    def plot_n_best_combinations_bar(
        self,
        y_axis="weighted_average",
        n_best=10,
        interactive=True,
        rank_on=None,
        title="default",
    ):
        if rank_on is not None:
            df = self.solution_df.sort_values(rank_on)
        else:
            df = self.solution_df
        if n_best is not None:
            df = df.head(n_best)

        if interactive:
            df["site_indices"] = df["site_indices"].astype("str")
            if rank_on is not None:
                title = f"Top {n_best} Solutions by {rank_on.replace('_', ' ').title()}"
            else:
                title = f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()}"
            fig = px.bar(
                df,
                x="site_indices",
                y=y_axis,
                title=title,
            )
        else:
            fig, ax = plt.subplots()
            ax.bar(
                df["site_indices"].astype(str),
                df[y_axis],
            )
            if title == "default":
                if rank_on is not None:
                    ax.set_title(
                        f"Top {n_best} Solutions by {rank_on.replace('_', ' ').title()}"
                    )
                else:
                    fig.update_layout(
                        title=f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()})"
                    )
            elif title is None:
                pass
            else:
                ax.set_title(title)

            ax.set_xlabel("Site Indices")
            ax.set_ylabel(f"{y_axis.str.replace('_').title()}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.close(fig)

        return fig
