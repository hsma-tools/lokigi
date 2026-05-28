import pandas as pd
from lokigi.utils import (
    _validate_columns,
    _load_spatial_or_tabular_data,
    GEOPANDAS_EXTS,
    _check_crs_match_pref,
    _convert_crs,
)
from warnings import warn


class _Problem:
    """Shared infrastructure."""

    def __init__(self, preferred_crs="EPSG:27700", debug_mode=True, **kwargs):
        self.preferred_crs = preferred_crs

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

        self.geo_lookup = None
        self._geo_lookup_data_type = None
        self._geo_lookup_common_col = None

        if debug_mode:
            self._verbose = True
        else:
            self._verbose = False

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

    ##################################
    # MARK: Equity Data
    ##################################
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

    ###############################
    # MARK: Region geometry
    ###############################
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

    def add_geo_lookup(self, lookup_df, common_col, rename=None):

        loaded_df, df_type = _load_spatial_or_tabular_data(lookup_df)

        if rename is not None:
            loaded_df = loaded_df.rename(columns=rename)

        self.geo_lookup = loaded_df
        self._geo_lookup_data_type = df_type
        self._geo_lookup_common_col = common_col

    def show_geo_lookup(self):
        return self.geo_lookup
