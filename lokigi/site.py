from lokigi.utils import (
    SOLVER_DEFINITIONS,
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
    GEOPANDAS_EXTS,
    _check_crs_match_pref,
    _convert_crs,
    SUPPORTED_OBJECTIVES,
    _get_ranking_by_objective,
    _validate_capacity_constraint,
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
from .mixins.site_solvers import BruteForceMixin, GreedyMixin, GraspMixin


class SiteProblem(BruteForceMixin, GreedyMixin, GraspMixin):
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

        self.travel_matrix = None  # Travel time/distance matrix
        self._travel_matrix_type = None
        self._travel_matrix_source_col = None
        self._travel_matrix_unit = None

        self.region_geometry_layer = None
        self._region_geometry_layer_type = None
        self._region_geometry_layer_common_col = None

        self.equity_data = None
        self._equity_data_type = None
        self._equity_data_equity_col = None
        self._equity_data_common_col = None
        self._equity_data_label = None

        # self.baseline_sites = None  # Current existing clinics
        # self._baseline_sites_type = None

        self.travel_and_demand_df = None
        self._joined_demand_travel_df_key_col = None

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
        Add demand data to the site problem and validate its structure.

        This method processes an input DataFrame or GeoDataFrame (or path) containing
        observed demand. It validates the presence of required columns and
        aligns the spatial or tabular data for use within the SiteProblem
        context.

        Parameters
        ----------
        demand_df : pandas.DataFrame, geopandas.GeoDataFrame or str
            The dataset containing demand information and location identifiers, or a local or web
            path to its location.
        demand_col : str
            The name of the column in `demand_df` representing the quantity
            of demand (e.g., patient counts, request volume, or other demand weighting).
        location_id_col : str
            The name of the column in `demand_df` used as a unique identifier
            for demand locations.
        skip_cols : list of str, optional
            A list of column names to ignore during the data loading process.
            Defaults to None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the required `demand_col` or `location_id_col` are missing
            from the provided `demand_df`.

        Notes
        -----
        The method updates several internal attributes:
        - `self.demand_data`: Stores the processed DataFrame.
        - `self._demand_data_type`: Stores whether the data is spatial or tabular.
        - `self._demand_data_demand_col`: Maps the demand value column.
        - `self._demand_data_id_col`: Maps the location identifier column.

        See Also
        --------
        _load_spatial_or_tabular_data : Internal utility for data ingestion.
        _validate_columns : Internal utility for schema verification.
        """
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
        """
        Returns a loaded demand dataframe
        """
        return self.demand_data

    def add_equity_data(
        self,
        equity_data,
        equity_col,
        common_col,
        label,
        continuous_measure=False,
        n_bins=10,
        reverse=False,
    ):
        """
        Add a dataframe containing equity data into your problem.

        This method associates demand points with an equity metric (such as
        the Index of Multiple Deprivation). If a continuous measure is provided,
        it is automatically discretized into deciles (or maximum possible quantiles)
        to facilitate categorical plotting and comparative equity analysis.

        Parameters
        ----------
        equity_data : str, pandas.DataFrame, or geopandas.GeoDataFrame
            The input data containing the equity metrics. Can be a filepath
            or an already loaded dataframe object.
        equity_col : str
            The name of the column in `equity_data` containing the equity
            values or categories to be used.
        common_col : str
            The name of the ID column used to join this data to the primary
            demand/spatial data in the SiteProblem.
        label : str
            A human-readable label for the equity metric (e.g., 'IMD Decile',
            'Age Group'). This is used internally for auto-generating plot
            titles and table headers.
        continuous_measure : bool, default False
            If True, treats `equity_col` as continuous numerical data and
            uses quantile-based discretization to convert it into deciles (1-10).
            The raw continuous data is preserved in a new column named
            `{equity_col}_raw`.
        reverse : bool, default False
            Only applicable if `continuous_measure` is True. By default (False),
            lower continuous values are assigned to lower deciles (e.g., 1).
            If True, the mapping is inverted so that lower continuous values
            are assigned to the highest deciles.

        Raises
        ------
        ValueError
            If `continuous_measure` is True but the data cannot be meaningfully
            binned due to too many identical values.

        Notes
        -----
        When `continuous_measure` is True, `pandas.qcut` is used with
        `duplicates='drop'`. If the data is highly skewed with duplicate values,
        this may result in fewer than 10 bins. The method handles this dynamically
        to ensure the resulting categories always start at 1.
        """
        loaded_df, df_type = _load_spatial_or_tabular_data(equity_data)

        if continuous_measure:
            loaded_df[f"{equity_col}_raw"] = loaded_df[equity_col]

            # We use qcut to split into 10 even groups (or whatever the user passes, but we'll
            # strongly recommend deciles).
            # labels=False returns 0-9, so we add 1 to get 1-10 for 'deciles'.
            try:
                bins = pd.qcut(
                    loaded_df[f"{equity_col}_raw"],
                    n_bins,
                    labels=False,
                    duplicates="drop",
                )

                if reverse:
                    # Dynamically invert based on the actual number of bins created
                    max_bin = bins.max()
                    loaded_df[equity_col] = (max_bin - bins) + 1
                else:
                    loaded_df[equity_col] = bins + 1
            except ValueError as e:
                print(
                    f"Warning: Could not create {n_bins} distinct categories for {equity_col}. "
                    "Check if the data has too many identical values."
                )
                raise e

        cols_to_include = [common_col, equity_col]
        if continuous_measure:
            cols_to_include.append(f"{equity_col}_raw")

        self.equity_data = loaded_df[cols_to_include]
        self._equity_data_type = "pandas"  # We drop any geometry data here
        self._equity_data_equity_col = equity_col
        self._equity_data_common_col = common_col
        self._equity_data_label = label

    def show_equity_data(self):
        return self.equity_data

    def add_region_geometry_layer(self, region_geometry_df, common_col):
        """
        Add a region geodataframe to the site problem and validate its structure.

        This method processes an input GeoDataFrame (or path) containing
        geometry data for the region of interest. It validates the presence of
        required columns and aligns the data for use within the SiteProblem context.

        If a preferred CRS has been passed and this dataframe is not of the preferred CRS,
        this dataframe will be transformed on loading to the preferred CRS. If no preferred
        CRS has been specified, no transformation will take place

        Parameters
        ----------
        region_geometry_df : geopandas.GeoDataFrame or str
            The dataset containing demand information and location identifiers, or a local or web
            path to its location.
        common_col : str
            The name of the column in `region_geometry_df` that should be used when joining to
            the demand data and travel matrix.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the required `demand_col` or `location_id_col` are missing
            from the provided `demand_df`.

        TypeError if a non-geopandas dataframe is passed.

        Notes
        -----
        The method updates several internal attributes:
        - `self.demand_data`: Stores the processed DataFrame.
        - `self._demand_data_type`: Stores whether the data is spatial or tabular.
        - `self._demand_data_demand_col`: Maps the demand value column.
        - `self._demand_data_id_col`: Maps the location identifier column.

        See Also
        --------
        _load_spatial_or_tabular_data : Internal utility for data ingestion.
        _validate_columns : Internal utility for schema verification.
        """

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
        """
        Returns a loaded region geometry geodataframe
        """
        return self.region_geometry_layer

    def plot_region_geometry_layer(
        self,
        interactive=False,
        plot_demand=False,
        plot_equity=False,
        cmap="Blues",
        tiles="CartoDB positron",
        plot_region_of_interest_only=False,
        edgecolor="black",
        linewidth=0.5,
        **kwargs,
    ):
        """
        Visualize the regional geometry layer, optionally overlaid with demand data.

        This method produces either a static matplotlib plot or an interactive
        Folium map (via Geopandas' .explore()). If demand plotting is enabled,
        it performs an internal join between geometry and demand data to create
        a choropleth map.

        Parameters
        ----------
        interactive : bool, default False
            If True, returns a folium.Map object using the 'explore' backend.
            If False, returns a matplotlib.axes.Axes object.
        plot_demand : bool, default False
            If True, merges the geometry with the demand dataset and styles
            the regions based on the demand column values.
        cmap: str, default "Blues"
            Colour map to be used for plotting demand. Ignored if plot_demand=False.
        tiles: str, default "CartoDB positron"
            Tiles to be used for background in map. Ignored if interactive = False.

        **kwargs : dict
            Additional keyword arguments passed to either
            `geopandas.GeoDataFrame.plot` or `geopandas.GeoDataFrame.explore`.

        Returns
        -------
        matplotlib.axes.Axes or folium.Map
            The plotting object depending on the `interactive` parameter.

        Raises
        ------
        ValueError
            If `self.region_geometry_layer` has not been initialized.
        ValueError
            If `plot_demand` is True but `self.demand_data` is None.

        Notes
        -----
        When `plot_demand` is True, the method performs a merge using:
        - `self._region_geometry_layer_common_col` (left)
        - `self._demand_data_id_col` (right)

        Interactive maps default to the "CartoDB positron" tile set and
        the "Blues" colormap for demand visualization.
        """
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

        if plot_demand and plot_equity:
            raise ValueError(
                "Cannot plot both demand and equity. Please set one option to False."
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
                    cmap=cmap,  # use "Blues" matplotlib colormap
                    style_kwds=dict(color="black"),
                    tiles=tiles,
                    **kwargs,
                )

                return m
            else:
                fig = plotting_df.plot(
                    column=self._demand_data_demand_col,
                    legend=True,
                    cmap=cmap,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )

                return fig

        if plot_equity:
            plotting_df = pd.merge(
                self.region_geometry_layer,
                self.equity_data,
                left_on=self._region_geometry_layer_common_col,
                right_on=self._equity_data_common_col,
            )

            if plot_region_of_interest_only:
                if self.demand_data is None:
                    warn(
                        "No demand data provided so cannot restrict to region of interest."
                    )

                plotting_df = plotting_df.merge(
                    self.demand_data[[self._demand_data_id_col]],
                    left_on=self._region_geometry_layer_common_col,
                    right_on=self._demand_data_id_col,
                    how="inner",
                )

            if interactive:
                m = plotting_df.explore(
                    column=self._equity_data_equity_col,  # make choropleth based on demand col
                    tooltip=self._equity_data_equity_col,  # show demand col value in tooltip (on hover)
                    popup=True,  # show all values in popup (on click)
                    cmap=cmap,  # use "Blues" matplotlib colormap
                    style_kwds=dict(color="black"),
                    tiles=tiles,
                    **kwargs,
                )

                return m
            else:
                fig = plotting_df.plot(
                    column=self._equity_data_equity_col,
                    legend=True,
                    cmap=cmap,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )

                return fig

        if plot_region_of_interest_only:
            if self.demand_data is None:
                warn(
                    "No demand data provided so cannot restrict to region of interest."
                )

            plotting_df = plotting_df.merge(
                self.demand_data[[self._demand_data_id_col]],
                left_on=self._region_geometry_layer_common_col,
                right_on=self._demand_data_id_col,
                how="inner",
            )
        else:
            plotting_df = self.region_geometry_layer

        if interactive:
            m = self.region_geometry_layer.explore(
                tiles=tiles, edgecolor=edgecolor, linewidth=linewidth, **kwargs
            )
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
        """
        Add candidate facility sites to the problem and handle spatial alignment.

        This method ingests site data from either a standard DataFrame or a
        GeoDataFrame. If tabular data is provided, it automatically converts
        coordinates into point geometries. It also ensures the data matches the
        object's preferred CRS, attempting to guess the CRS if it's not provided.

        Parameters
        ----------
        candidate_site_df : pandas.DataFrame or geopandas.GeoDataFrame or str
            The dataset containing potential site locations, or a local or web
            path to its location.
        candidate_id_col : str
            The name of the column containing unique identifiers for each site.
        required_sites_col : str, optional
            The name of a boolean or binary column indicating if a site must be
            included in the final solution. Defaults to None.
        geometry_col : str, default "geometry"
            The name of the geometry column (used if `candidate_site_df` is
            already a GeoDataFrame or is a path to a geodataframe).
        vertical_geometry_col : str, default "lat"
            The column name for latitude/y-coordinates (used if input is tabular
            or a path to a tabular file format like .csv).
        horizontal_geometry_col : str, default "long"
            The column name for longitude/x-coordinates (used if input is tabular
            or a path to a tabular file format like .csv).
        crs : str or pyproj.CRS, optional
            The coordinate reference system of the input data. If None and the
            input is tabular, the method will attempt to guess the CRS.
        capacity_col : str, optional
            The column name representing the capacity of each site. Defaults to None.
        skip_cols : list of str, optional
            A list of column names to ignore during the data loading process.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If required columns (ID, capacity, or geometry) are missing from the
            input data.

        Notes
        -----
        The method performs the following transformations:
        1. Infers data type (spatial vs. tabular).
        2. Validates schema based on the data type.
        3. If tabular, converts to a `geopandas.GeoDataFrame` using the
           specified horizontal and vertical coordinate columns.
        4. Matches or converts the dataset to `self.preferred_crs`.

        Updates internal state including `self.candidate_sites` and
        `self.total_n_sites`.
        """
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
        """
        Returns a loaded candidate site geodataframe

        Returns
        -------
        geopandas.Geodataframe
            A geopandas geodataframe containing the candidate sites

        """
        return self.candidate_sites

    def plot_sites(self, add_basemap=True, show_labels=True, interactive=False):
        """
        Generate a visualization of the candidate facility sites.

        This method provides a quick way to inspect site locations. It supports
        both static matplotlib plots (with automatic label de-confliction)
        and interactive Folium maps.

        Parameters
        ----------
        add_basemap : bool, default True
            If True, adds a background web map using `contextily`. Only
            applicable for static plots (`interactive=False`).
        show_labels : bool, default True
            If True, adds text labels for each site using the candidate ID
            column. Labels are automatically wrapped and positioned to
            avoid overlap using `adjust_text`. Only applicable for static plots
            (`interactive=False`)..
        interactive : bool, default False
            If True, returns an interactive folium map via the `.explore()`
            method.

        Returns
        -------
        matplotlib.axes.Axes or folium.Map
            The plotting object. Returns an Axes object for static plots
            or a Map object for interactive visualizations.

        Notes
        -----
        Static plots use `adjust_text` to ensure that site labels remain
        legible even in high-density areas. Labels are title-cased and
        wrapped at a width of 15 characters.
        """
        if not interactive:
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

    def add_travel_matrix(
        self,
        travel_matrix_df,
        source_col,
        skip_cols=None,
        unit=None,
        from_unit=None,
        to_unit=None,
    ):
        """
        Add a travel cost matrix to the problem and handle unit conversions.

        This method integrates a matrix (typically time or distance) representing
        the travel cost between demand locations and candidate sites. It can
        automatically scale numeric columns if time unit conversion is required
        (e.g., converting seconds to minutes).

        Parameters
        ----------
        travel_matrix_df : pandas.DataFrame or geopandas.GeoDataFrame or str
            The dataset containing travel costs, or a local or web
            path to its location. Usually structured as an
            origin-destination matrix or a long-format table.
        source_col : str
            The column name in `travel_matrix_df` that identifies the origin
            points (should correspond to IDs in the demand or site data).
        skip_cols : list of str, optional
            A list of column names to ignore during data loading.
        unit : str, optional
            A label for the units used in the matrix (e.g., "miles", "km").
            Used if no conversion is performed.
        from_unit : {"seconds", "minutes", "hours"}, optional
            The current time unit of the numeric values in the dataframe.
        to_unit : {"seconds", "minutes", "hours"}, optional
            The target time unit for the numeric values in the dataframe.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the `source_col` is missing from the provided dataframe.
        KeyError
            If the `from_unit` to `to_unit` combination is not supported
            by the internal conversion dictionary.

        Notes
        -----
        If both `from_unit` and `to_unit` are provided, the method identifies
        all numeric columns in the dataframe and applies the appropriate
        multiplication factor. Supported conversions are limited to time-based
        units (seconds, minutes, hours).

        The resulting data is stored in `self.travel_matrix`, and the resolved
        unit label is stored in `self._travel_matrix_unit`.
        """
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
        """
        Returns a loaded travel or cost matrix dataframe

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe containing the travel times, distances or other costs for
            combinations of sources (rows) and destinations (columns)

        """
        return self.travel_matrix

    def _create_joined_demand_travel_df(self, index_col):
        """
        Merge demand data with travel costs and (if present) equity data into a single master dataframe.

        This internal method performs an inner join between the demand dataset
        and the travel matrix. It ensures that the resulting object preserves
        spatial properties if the demand data is a GeoDataFrame by managing
        the merge order.

        Parameters
        ----------
        index_col : str
            The column name to be set as the index of the resulting merged
            dataframe (typically the unique identifier for demand locations).

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the inner join results in an empty dataframe. This usually
            indicates a mismatch between the ID values in the demand data
            and the source values in the travel matrix.

        Notes
        -----
        The method logic is sensitive to the data types of the inputs:
        - If `self.demand_data` is a GeoDataFrame, it is placed on the left
          side of the join to ensure the output is also a GeoDataFrame.
        - The join is performed on `self._demand_data_id_col` and
          `self._travel_matrix_source_col`.

        The merged result is stored in the `self.travel_and_demand_df` attribute.
        """
        # If one is a geopandas dataframe, put that first in the merge call so that the
        # output object will also be a geodataframe
        if self._demand_data_type == "geopandas":
            travel_and_demand_df = pd.merge(
                self.demand_data,
                self.travel_matrix,
                left_on=self._demand_data_id_col,
                right_on=self._travel_matrix_source_col,
                how="inner",
            )

            self._joined_demand_travel_df_key_col = self._demand_data_id_col

        else:
            travel_and_demand_df = pd.merge(
                self.travel_matrix,
                self.demand_data,
                left_on=self._travel_matrix_source_col,
                right_on=self._demand_data_id_col,
                how="inner",
            )

            self._joined_demand_travel_df_key_col = self._travel_matrix_source_col

        if len(travel_and_demand_df) == 0:
            raise KeyError(
                "Warning: merging the travel matrix and demand data has failed."
                f"This may be because there are no common values found in the {self._travel_matrix_source_col}"
                f"(sample values: {self.travel_matrix.head(5)[self._travel_matrix_source_col]})"
                f"column in the travel dataframe and the {self._demand_data_id_col} column in the"
                f"demand dataframe (sample values: {self.demand_data.head(5)[self._demand_data_id_col]})"
            )

        self.travel_and_demand_df = travel_and_demand_df.set_index(index_col)

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
            site_indices=resolved_indices,
            site_names=site_names,
            evaluated_combination_df=active_facilities,
            site_problem=self,
            coverage_threshold=threshold_for_coverage,
        )

    def _setup_equal_demand_df(self):
        """
        Initialize a default demand dataset with uniform weights.

        This internal method is used when no explicit demand data has been
        provided. It creates a synthetic demand DataFrame based on the
        unique source locations found in the travel matrix, assigning a
        nominal demand value of 1 to every location.

        Returns
        -------
        None

        Notes
        -----
        This method updates the following internal attributes:
        - `self.demand_data`: A new pandas DataFrame containing location IDs
          and a demand column 'n'.
        - `self._demand_data_type`: Set to "pandas".
        - `self._demand_data_id_col`: Set to match the travel matrix source column.
        - `self._demand_data_demand_col`: Set to "n".

        This ensures that optimization objectives like p-median can still
        function by minimizing average travel time across all known
        locations equally.
        """
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
        """
        Generate a candidate sites DataFrame directly from travel matrix columns.

        This internal method is invoked when no explicit candidate site data
        has been provided. It extracts all destination column names from the
        travel matrix (excluding the source/ID column) and treats them as
        the available facility locations.

        Returns
        -------
        None

        Notes
        -----
        Because the travel matrix columns typically only contain names/IDs,
        the resulting `self.candidate_sites` will not contain spatial
        geometry (lat/long) or capacity information.

        The following internal attributes are updated:
        - `self.candidate_sites`: A DataFrame containing a 'site' column
          and an integer index.
        - `self._candidate_sites_type`: Set to "pandas".
        - `self._candidate_sites_candidate_id_col`: Set to "site".
        - `self.total_n_sites`: Set to the number of extracted columns.
        - Spatial and capacity columns are explicitly set to `None`.

        See Also
        --------
        _setup_equal_demand_df : The counterpart for generating default demand.
        """
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
                p=p,
                objectives=objective,
                show_progress=show_progress,
                threshold_for_coverage=threshold_for_coverage,
            )

            return SiteSolutionSet(
                solution_df=pd.DataFrame(outputs).sort_values(
                    [ranking, "weighted_average"]
                ),
                site_problem=self,
                objectives=objective,
                n_sites=p,
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

            return SiteSolutionSet(
                solution_df=pd.DataFrame(outputs).sort_values(
                    [ranking, "weighted_average"]
                ),
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
