from lokigi.utils import (
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
    _check_crs_match_pref,
    _convert_crs,
)

# Data manipulation imports
import pandas as pd
import geopandas

# Plotting imports
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt


class SiteAttributeMixin:
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

    ########################
    # MARK: Demand
    ########################
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

    ###############################
    # MARK: Sites
    ###############################
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

        loaded_df = loaded_df.reset_index(drop=False, names="canonical_site_index")

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

        sites_df_temp = sites_df_temp.reset_index(
            drop=False, names="canonical_site_index"
        )

        self.candidate_sites = sites_df_temp
        self._candidate_sites_type = "pandas"
        self._candidate_sites_candidate_id_col = "site"
        self._candidate_sites_vertical_col = None
        self._candidate_sites_horizontal_col = None
        self._candidate_sites_capacity_col = None
        self.total_n_sites = len(self.candidate_sites)
