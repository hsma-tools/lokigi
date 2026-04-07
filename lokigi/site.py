from lokigi.utils import (
    SOLVER_DEFINITIONS,
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
    GEOPANDAS_EXTS,
    _generate_all_combinations,
    _check_crs_match_pref,
    _convert_crs,
    ALIASES,
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

# Warn if brute force will be slow
_BRUTE_FORCE_WARN_THRESHOLD = 50


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
        print(self.demand_data)

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

    def add_sites(
        self,
        candidate_site_df,
        candidate_id_col,
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

        self.candidate_sites = loaded_df
        self._candidate_sites_type = df_type
        self._candidate_sites_candidate_id_col = candidate_id_col
        self._candidate_sites_geometry_col = geometry_col
        self._candidate_sites_capacity_col = capacity_col
        self.total_n_sites = len(self.candidate_sites)

    def show_sites(self):
        print(self.candidate_sites)

    def plot_sites(self, add_basemap=True, show_labels=True):
        """
        Adds a quick plot
        """
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

        if objective in ALIASES:
            objective = ALIASES[objective]
        else:
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

        if objective == "p_median":
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
                "p-median",
                site_indices=resolved_indices,
                site_names=site_names,
                evaluated_combination_df=active_facilities,
                site_problem=self,
            )
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Currently supported: {SUPPORTED_OBJECTIVES.join(', ')}."
            )

    def solve(
        self,
        p: int,
        objectives: str | list[str] = "p_median",
        capacitated=False,
        search_strategy: Literal[
            "brute-force", "evolutionary", "genetic"
        ] = "brute-force",
        **kwargs,
    ):

        if isinstance(objectives, list) and len(objectives) > 1:
            warn(
                "Multi-objective optimization is coming in a future release."
                f"For now, just your first objective {objectives[0]} has been taken."
            )

        objective = objectives if isinstance(objectives, str) else objectives[0]

        if objective in ALIASES:
            objective = ALIASES[objective]
        else:
            raise ValueError(f"Unsupported objective ({objective}) passed.")

        if objective == "p_median":
            return self._solve_pmedian_problem(
                p,
            )
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Currently supported: 'p_median'."
            )

    def _solve_pmedian_problem(self, p: int):
        possible_combinations = _generate_all_combinations(
            n_facilities=self.total_n_sites, p=p
        )

        outputs = []

        for possible_solution in possible_combinations:
            outputs.append(
                self.evaluate_single_solution(
                    site_indices=possible_solution, objectives="p-median"
                ).generate_solution_metrics()
            )

        return SiteSolutionSet(pd.DataFrame(outputs).sort_values("weighted_average"))

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

    def generate_solution_metrics(self):
        # Return weighted average
        weighted_average = np.average(
            self.evaluated_combination_df["min_cost"],
            weights=self.evaluated_combination_df[
                self.site_problem._demand_data_demand_col
            ],
        )
        unweighted_average = np.average(self.evaluated_combination_df["min_cost"])
        percentile_90th = np.percentile(self.evaluated_combination_df["min_cost"], q=90)
        max_travel = np.max(self.evaluated_combination_df["min_cost"])

        return {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "weighted_average": weighted_average,
            "unweighted_average": unweighted_average,
            "90th_percentile": percentile_90th,
            "max": max_travel,
            "problem_df": self.evaluated_combination_df,
        }


class SiteSolutionSet:
    def __init__(self, solution_df):
        self.solution_df = solution_df.reset_index(drop=True)

    def show_solutions(self):
        return self.solution_df

    def return_best_combination_details(self, rank_on="weighted_average"):
        return self.solution_df.sort_values(rank_on).head(1)

    def return_best_combination_site_indices(self, rank_on="weighted_average"):
        return (
            self.solution_df.sort_values(rank_on)
            .head(1)["site_indices"]
            .reset_index()[0]
        )

    def return_best_combination_site_names(self, rank_on="weighted_average"):
        return (
            self.solution_df.sort_values(rank_on).head(1)["site_names"].reset_index()[0]
        )

    def plot_travel_time_distribution():
        pass

    def summary_table():
        pass

    def plot_best_combination(
        self,
        problem_class,
        rank_on="weighted_average",
        title=None,
        show_all_locations=True,
        cmap="Blues",
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
    ):

        if problem_class.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        solution = self.solution_df.sort_values(rank_on).head().reset_index()

        nearest_site_travel_gdf = pd.merge(
            problem_class.region_geometry_layer,
            solution["problem_df"][0],
            left_on=problem_class._region_geometry_layer_common_col,
            right_on=problem_class._demand_data_id_col,
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

        selected_sites = problem_class.candidate_sites.iloc[
            solution.site_indices.iloc[0]
        ]

        if show_all_locations:
            all_site_points = problem_class.candidate_sites.plot(
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
            selected_sites[problem_class._candidate_sites_candidate_id_col],
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
        problem_class,
        n_best=10,
        rank_on="weighted_average",
        title=None,
        subplot_title="default",
        show_all_locations=True,
        cmap="Blues",
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
    ):
        max_cols = 5
        ncols = min(n_best, max_cols)
        nrows = math.ceil(n_best / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

        # flatten axs in case it's a 2D array
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()

        if problem_class.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        sorted_df = self.solution_df.sort_values(rank_on).reset_index().head(n_best)

        for i, ax in enumerate(fig.axes):
            solution = sorted_df.iloc[[i]]
            solution_df = solution["problem_df"].values[0]

            nearest_site_travel_gdf = pd.merge(
                problem_class.region_geometry_layer,
                solution_df,
                left_on=problem_class._region_geometry_layer_common_col,
                right_on=problem_class._demand_data_id_col,
            )

            ax = nearest_site_travel_gdf.plot(
                "min_cost",
                legend=True,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
                ax=ax,
            )

            selected_sites = problem_class.candidate_sites.iloc[
                solution.site_indices.iloc[0]
            ]

            if show_all_locations:
                all_site_points = problem_class.candidate_sites.plot(
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
                selected_sites[problem_class._candidate_sites_candidate_id_col],
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
                    ax.set_title(f"\n{solution['weighted_average'].values[0]:.1f}")
                else:
                    ax.set_title(safe_evaluate_title(subplot_title))
            if title is not None:
                ax.set_title(title)
