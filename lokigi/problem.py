import pandas as pd
from lokigi.utils import (
    _validate_columns,
    _load_spatial_or_tabular_data,
    GEOPANDAS_EXTS,
    _check_crs_match_pref,
    _convert_crs,
)
from warnings import warn
import hashlib
from typing import Literal
import numpy as np


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
        self._region_geometry_hash = None

        self.spatial_weights = None
        self.spatial_weights_method = None
        self.spatial_weights_k = None

        self.equity_data = None
        self._equity_data_type = None
        self._equity_data_equity_col = None
        self._equity_data_common_col = None
        self._equity_data_label = None
        self._equity_data_direction = None

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
        direction: Literal["higher_is_better", "higher_is_worse"] = "higher_is_worse",
        continuous_measure: bool = False,
        n_bins: int = 10,
        reverse: bool = False,
        verbose: bool = True,
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
        direction : {"higher_is_better", "higher_is_worse"}, default "higher_is_better"
            Indicates whether higher values of `equity_col` represent a more or
            less advantaged group. This is stored as metadata and applied at
            analysis time — it does not modify the stored data.

            - ``"higher_is_better"`` : higher values indicate a more favourable
            equity position (e.g. IMD decile 10 = least deprived under the
            standard DLUHC 1–10 scale).
            - ``"higher_is_worse"`` : higher values indicate greater disadvantage
            (e.g. raw IMD score, where a higher score means more deprived;
            or a custom scale where 1 = least deprived).

            .. note::
                IMD *deciles* as published by DLUHC run 1 (most deprived) to 10
                (least deprived), so for pre-binned IMD decile columns use
                ``direction="higher_is_better"``. For raw IMD *scores* (higher =
                more deprived) use ``direction="higher_is_worse"``.
        continuous_measure : bool, default False
            If True, treats `equity_col` as continuous numerical data and
            uses quantile-based discretization to convert it into deciles (1-10).
            The raw continuous data is preserved in a new column named
            `{equity_col}_raw`.
        reverse : bool, default False
            Only used when ``continuous_measure=True``. Controls the direction
            of bin labelling relative to the raw values:

            - ``False`` (default): lower raw values receive lower bin numbers.
            - ``True``: lower raw values receive higher bin numbers (i.e. the
            labelling is inverted).

            This is purely a binning convenience — for instance, to convert a
            raw IMD score (where lower = less deprived) into a decile where
            1 = least deprived. It is independent of ``direction``, which
            governs downstream analysis rather than how bins are labelled.
        verbose: bool, default True
            If True, output additional warnings and messages

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

        if verbose:
            if df_type == "geopandas":
                warn(
                    "Equity_data appears to be a GeoDataFrame; geometry will be dropped.",
                    UserWarning,
                    stacklevel=2,
                )

        if continuous_measure:
            loaded_df[f"{equity_col}_raw"] = loaded_df[equity_col]

            try:
                bins = pd.qcut(
                    loaded_df[f"{equity_col}_raw"],
                    n_bins,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError as e:
                raise ValueError(
                    f"Could not bin '{equity_col}' into any distinct quantile categories. "
                    "The column may contain too many identical values."
                ) from e

            actual_bins = int(bins.max()) + 1
            if actual_bins < n_bins:
                warn(
                    f"Requested {n_bins} bins for '{equity_col}' but only "
                    f"{actual_bins} distinct quantile bins could be formed due to "
                    "duplicate values. Consider inspecting the distribution.",
                    UserWarning,
                    stacklevel=2,
                )

            if reverse:
                loaded_df[equity_col] = (bins.max() - bins) + 1
            else:
                loaded_df[equity_col] = bins + 1

        cols_to_include = [common_col, equity_col]

        if continuous_measure:
            cols_to_include.append(f"{equity_col}_raw")

        self.equity_data = loaded_df[cols_to_include]
        self._equity_data_type = "pandas"  # We drop any geometry data here
        self._equity_data_equity_col = equity_col
        self._equity_data_common_col = common_col
        self._equity_data_label = label
        self._equity_data_direction = direction

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
        self._region_geometry_hash = self._get_geometry_hash(self.region_geometry_layer)

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

    def _get_geometry_hash(self, df) -> str:
        # 1. Convert the entire geometry series to WKB
        wkb_series = df.geometry.to_wkb()

        # 2. Join all the byte strings in the series together
        combined_bytes = b"".join(wkb_series)

        # 3. Hash the resulting single byte string
        return hashlib.sha256(combined_bytes).hexdigest()
