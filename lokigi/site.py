from lokigi.utils import (
    SOLVER_DEFINITIONS,
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
)

import pandas as pd
import geopandas
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt

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

    def __init__(self, debug_mode=True):
        self.name = None

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

        self.travel_matrix = None  # Travel time/distance matrix
        self._travel_matrix_type = None
        self._travel_matrix_source_col = None

        self.geometry_layer = None
        self._geometry_layer_common_col = None

        self.baseline_sites = None  # Current existing clinics
        self._baseline_sites_type = None

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

    def add_demand(self, demand_df, demand_col, location_id_col):
        """
        Adds a dataframe or geodataframe containing the demand observed
        to the SiteProblem object.

        df: Pandas DataFrame or Geopandas Geodataframe
        Validates CRS and aligns patient data."""
        loaded_df, df_type = _load_spatial_or_tabular_data(demand_df)

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

    def add_sites(
        self,
        candidate_site_df,
        candidate_id_col,
        geometry_col="geometry",
        vertical_geometry_col="lat",
        horizontal_geometry_col="long",
        crs=None,
        capacity_col=None,
    ):
        """Validates CRS and identifies potential new sites."""
        loaded_df, df_type = _load_spatial_or_tabular_data(candidate_site_df)

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

            loaded_df = geopandas.GeoDataFrame(
                data=loaded_df,
                geometry=geopandas.points_from_xy(
                    loaded_df[horizontal_geometry_col], loaded_df[vertical_geometry_col]
                ),
                crs=crs,
            )

        self.candidate_sites = loaded_df
        self._candidate_sites_type = df_type
        self._candidate_sites_candidate_id_col = candidate_id_col
        self._candidate_sites_geometry_col = geometry_col
        self._candidate_sites_capacity_col = capacity_col

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

    def add_travel_matrix(self, travel_matrix_df, source_col):
        """Ensures the matrix indices match the demand/candidate IDs."""
        loaded_df, df_type = _load_spatial_or_tabular_data(travel_matrix_df)

        _validate_columns(
            df=loaded_df,
            col_names=[source_col],
            msg_template=(
                "It looks like your travel matrix data is missing these columns: {missing}. "
                "We found these instead: {available}. Please double-check the column names "
                "you are passing to the .add_travel_matrix() method."
            ),
        )

        self.travel_matrix = loaded_df
        self._travel_matrix_source_col = source_col

    def _create_joined_demand_travel_df(self):
        self.travel_and_demand_df = pd.merge()

    def solve(
        self,
        p: int,
        objectives: str | list[str] = "p_median",
        num_options=10,
        capacitated=False,
        strategy=None,
        **kwargs,
    ):
        if isinstance(objectives, list) and len(objectives) > 1:
            raise NotImplementedError(
                "Multi-objective solving is not yet implemented. "
                "Pass a single objective string, or use solve_pmedian()."
            )

        objective = objectives if isinstance(objectives, str) else objectives[0]

        if objective == "p_median":
            return self._solve_pmedian_problem(
                p,
            )
        else:
            raise ValueError(
                f"Unknown objective '{objective}'. Currently supported: 'p_median'."
            )

    def solve_pmedian(self, p, num_options=10, capacitated=False):
        """ """
        return self.solve(
            p=p, objectives="p_median", num_options=num_options, capacitated=capacitated
        )

    def _solve_pmedian_problem(self, p: int):
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


class SiteSolution:
    def __init__(self, solution_type):
        self.solution_type = solution_type

    @property
    def info(self):
        """Prints a student-friendly explanation of the model used."""
        data = SOLVER_DEFINITIONS.get(self.solution_type, {})
        print(f"--- Model Type: {data.get('name')} ---")
        print(f"Goal: {data.get('goal')}")
        print(f"Healthcare Application: {data.get('healthcare_context')}")
        print(f"Keep in mind: {data.get('trade_off')}")

    def plot_utilization(self):
        pass


class SiteSolutionSet:
    def __init__(self):
        pass
