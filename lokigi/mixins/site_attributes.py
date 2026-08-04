from lokigi.plot_utils import _add_plot_caption, _attach_deferred_fit_bounds
from lokigi.utils import (
    _validate_columns,
    _load_spatial_or_tabular_data,
    _guess_crs,
    _check_crs_match_pref,
    _convert_crs,
    _min_max_normalize,
)

# Data manipulation imports
import pandas as pd
import geopandas

# Plotting imports
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
from requests.exceptions import RequestException


_TIME_UNIT_CONVERSION = {
    ("seconds", "minutes"): 1 / 60,
    ("seconds", "hours"): 1 / 3600,
    ("minutes", "seconds"): 60,
    ("minutes", "hours"): 1 / 60,
    ("hours", "seconds"): 3600,
    ("hours", "minutes"): 60,
}


def _apply_unit_conversion(loaded_df, unit, from_unit, to_unit):
    """Resolve a travel matrix's unit label and (if requested) convert its
    numeric columns between time units. Mutates and returns `loaded_df`,
    plus the resolved unit label, matching add_travel_matrix()'s existing
    (in-place) behaviour."""
    if from_unit is not None and to_unit is not None:
        factor = _TIME_UNIT_CONVERSION[(from_unit, to_unit)]
        num_cols = loaded_df.select_dtypes(include="number").columns
        loaded_df.loc[:, num_cols] *= factor
        resolved_unit = to_unit
    elif from_unit is None and to_unit is not None:
        resolved_unit = to_unit
    elif from_unit is not None and to_unit is None:
        resolved_unit = from_unit
    else:
        resolved_unit = unit

    return loaded_df, resolved_unit


def _apply_treat_as_missing(loaded_df, source_col, treat_as_missing):
    """Convert an existing missing-data sentinel (e.g. 9999) in a travel
    matrix's cost columns to a proper NaN, before any missing-value check or
    unit conversion runs -- so `treat_as_missing` is always matched against
    the value as the caller supplied it, not a unit-converted one.
    `treat_as_missing` may be a scalar (matched by equality) or a callable
    taking a cell value and returning bool. Mutates and returns `loaded_df`.
    """
    if treat_as_missing is None:
        return loaded_df

    cost_cols = loaded_df.columns.drop(source_col)
    if callable(treat_as_missing):
        sentinel_mask = loaded_df[cost_cols].map(treat_as_missing)
    else:
        sentinel_mask = loaded_df[cost_cols] == treat_as_missing
    loaded_df[cost_cols] = loaded_df[cost_cols].where(~sentinel_mask)
    return loaded_df


def _reject_missing_travel_values(loaded_df, cost_cols, allow_missing, method_name):
    """Raise unless `allow_missing` opts in, if any of `cost_cols` in
    `loaded_df` hold a missing (NaN) travel cost.

    A missing travel cost means "no feasible journey" for that
    origin-destination pair (e.g. no public transport route within a
    permissive search radius) -- lokigi treats that as a real, first-class
    outcome rather than something to silently paper over with a sentinel
    like 9999, but it still has to be an explicit opt-in: an unnoticed NaN
    (an ID mismatch, a botched generation run) should keep failing loudly.
    """
    if allow_missing:
        return

    nan_mask = loaded_df[cost_cols].isna()
    if not nan_mask.any().any():
        return

    n_missing = int(nan_mask.sum().sum())
    example_rows = loaded_df.index[nan_mask.any(axis=1)][:5].tolist()
    raise ValueError(
        f"{method_name} found {n_missing} missing (NaN) travel cost "
        f"value(s) (e.g. row(s) {example_rows}). A missing value means "
        "\"no feasible journey\" for that origin-destination pair, not a "
        "data error to be assumed away -- pass allow_missing=True to "
        "register the matrix as-is (unreachable pairs are then treated as "
        "genuinely unreachable throughout evaluation and plotting, rather "
        "than needing a large sentinel travel time), or "
        "treat_as_missing=<value or callable> to convert an existing "
        "sentinel (e.g. 9999) to a proper missing value first."
    )


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
    def add_demand(
        self,
        demand_df,
        demand_col,
        location_id_col,
        drop_extra_cols=True,
        skip_cols=None,
    ):
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
        drop_extra_cols: bool, optional
            Whether to drop (remove) additional columns present in the dataset other than the demand_col
            and location_id_col.
            Defaults to True.
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

        if drop_extra_cols:
            self.demand_data = loaded_df[[location_id_col, demand_col]]
        else:
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
        - `self._demand_data_demand_col`: Set to "demand".

        This ensures that optimization objectives like p-median can still
        function by minimizing average travel time across all known
        locations equally.
        """
        demand_data_temp = pd.DataFrame(
            self.travel_matrix[self._travel_matrix_source_col],
            columns=[self._travel_matrix_source_col],
        )
        demand_data_temp["demand"] = 1

        self.demand_data = demand_data_temp
        self._demand_data_type = "pandas"
        self._demand_data_id_col = self._travel_matrix_source_col
        self._demand_data_demand_col = "demand"

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
        cost_col=None,
        allow_missing_cost=False,
        current_load_col=None,
        utilisation_col=None,
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
        cost_col : str, optional
            The column name representing the fixed cost (e.g. build or
            operating cost) of each site. Must contain numeric values.
            Defaults to None. The total cost of the sites selected in a
            solution is reported in `solutions_df`, but cost only
            influences which solution is chosen if it is explicitly passed
            as a weight via `solve(weights={"cost": ...})`. By default, any
            site missing a value in this column raises a `ValueError` --
            see `allow_missing_cost`.
        allow_missing_cost : bool, default False
            Whether to allow rows with a missing (NaN) `cost_col` value.
            By default, missing costs raise a `ValueError` naming the
            affected sites, since a missing cost would otherwise silently
            behave like a $0 cost. Pass `True` to allow it: sites with a
            missing cost then propagate as `NaN` in any solution's
            `total_cost` (rather than being treated as free), for cases
            where cost data is genuinely incomplete. Ignored if `cost_col`
            is not provided.
        current_load_col : str, optional
            The column name representing each site's current real-world
            activity/caseload -- today's baseline, not anything derived
            from `solve()`. Must be paired with `capacity_col` (needed to
            turn a raw count into a utilisation ratio); pass a
            precomputed ratio via `utilisation_col` instead if you don't
            have raw counts. Used by `site_utilisation_summary()` and
            `plot_site_utilisation()`. Defaults to None.
        utilisation_col : str, optional
            The column name representing each site's current utilisation
            as a precomputed ratio or percentage, for analysts who don't
            have raw current-load/capacity counts to hand. Mutually
            exclusive with `current_load_col`. May optionally be combined
            with `capacity_col` to also derive `headroom` in
            `site_utilisation_summary()`. Defaults to None.
        skip_cols : list of str, optional
            A list of column names to ignore during the data loading process.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If required columns (ID, capacity, or geometry) are missing from the
            input data, if `cost_col` has a missing value for one or more
            sites and `allow_missing_cost=False` (the default), if both
            `current_load_col` and `utilisation_col` are given, if
            `current_load_col` is given without `capacity_col`, or if
            `current_load_col`/`utilisation_col` contains a negative value.

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

        Unlike `cost_col`, `current_load_col`/`utilisation_col` have no
        `allow_missing_*` flag and a missing (NaN) value is never rejected
        here. `candidate_sites` typically mixes already-operating sites
        (which have a real current load) with not-yet-built proposals (which
        have no current load simply because they don't exist yet, not
        because of a data-entry gap). A missing cost has a dangerous silent
        default (looks free); a missing utilisation figure doesn't -- it
        means "not applicable", not "0% full" -- so `site_utilisation_summary()`
        reports it as an explicit `NaN` rather than coercing it to 0.
        """
        loaded_df, df_type = _load_spatial_or_tabular_data(
            candidate_site_df, skip_cols=skip_cols
        )

        if current_load_col is not None and utilisation_col is not None:
            raise ValueError(
                "Please provide at most one of 'current_load_col' or "
                "'utilisation_col', not both -- lokigi wouldn't know which "
                "to trust."
            )

        if current_load_col is not None and capacity_col is None:
            raise ValueError(
                "current_load_col requires capacity_col to also be provided, "
                "so a utilisation ratio can be derived from the two raw "
                "counts. If you don't have raw counts, pass a precomputed "
                "ratio via utilisation_col instead."
            )

        col_list = [candidate_id_col]
        if capacity_col is not None:
            col_list.extend([capacity_col])
        if cost_col is not None:
            col_list.extend([cost_col])
        if current_load_col is not None:
            col_list.extend([current_load_col])
        if utilisation_col is not None:
            col_list.extend([utilisation_col])

        numeric_col_names = [
            c
            for c in (capacity_col, current_load_col, utilisation_col, cost_col)
            if c is not None
        ] or None

        if df_type == "geopandas":
            col_list.extend([geometry_col])
            _validate_columns(
                df=loaded_df,
                col_names=col_list,
                numeric_col_names=numeric_col_names,
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
                numeric_col_names=numeric_col_names,
                msg_template=(
                    "It looks like your candidate site data is missing these columns: {missing}. "
                    "We found these instead: {available}. Please double-check the column names you are "
                    "passing in to the .add_candidates() method and try running this method again."
                ),
            )

        if cost_col is not None and not allow_missing_cost:
            missing_cost_mask = loaded_df[cost_col].isna()
            if missing_cost_mask.any():
                missing_site_names = loaded_df.loc[
                    missing_cost_mask, candidate_id_col
                ].tolist()
                raise ValueError(
                    f"The following sites are missing a value in cost_col='{cost_col}': "
                    f"{missing_site_names}. A missing cost would otherwise be silently "
                    "treated as $0. Either fill in the missing cost values, or pass "
                    "add_sites(..., allow_missing_cost=True) to proceed anyway -- "
                    "solutions containing these sites will then report total_cost as NaN "
                    "rather than a possibly-misleading number."
                )

        for negative_col in (current_load_col, utilisation_col):
            if negative_col is not None:
                negative_mask = loaded_df[negative_col] < 0
                if negative_mask.any():
                    negative_site_names = loaded_df.loc[
                        negative_mask, candidate_id_col
                    ].tolist()
                    raise ValueError(
                        f"The following sites have a negative value in "
                        f"'{negative_col}', which is not a valid current "
                        f"load/utilisation value: {negative_site_names}."
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
        self._candidate_sites_cost_col = cost_col
        self._candidate_sites_current_load_col = current_load_col
        self._candidate_sites_utilisation_col = utilisation_col
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
                try:
                    cx.add_basemap(
                        ax, crs=self.candidate_sites.crs.to_string(), timeout=30
                    )
                except RequestException as e:
                    warnings.warn(
                        f"Unable to download background map tiles ({type(e).__name__}). "
                        "Continuing without a basemap.",
                        stacklevel=2,
                    )
        else:
            m = self.candidate_sites.explore(marker_kwds=dict(radius=8))
            return _attach_deferred_fit_bounds(m)

    ###############################
    # MARK: Site Utilisation
    ###############################
    def site_utilisation_summary(self, site_names=None, site_indices=None):
        """
        Summarise each candidate site's real-world current utilisation.

        This is a baseline diagnostic, not a `solve()` output: it reports
        today's actual load against capacity, exactly as registered via
        `add_sites(current_load_col=..., capacity_col=...)` or
        `add_sites(utilisation_col=...)`. It works with or without ever
        calling `solve()`, and has no `SiteSolutionSet` counterpart --
        there is nothing "solved" to select from.

        Parameters
        ----------
        site_names : list of str, optional
            Restrict the summary to these site names (as they appear in
            `candidate_id_col`). Mutually exclusive with `site_indices`. If
            neither is given, every registered candidate site is included.
        site_indices : list of int, optional
            Restrict the summary to these `canonical_site_index` values.
            Mutually exclusive with `site_names`.

        Returns
        -------
        pandas.DataFrame
            Indexed by site name (`index.name == "site"`), in
            `canonical_site_index` order (or `site_names`' given order, if
            supplied). Columns are included conditionally, depending on
            which of `capacity_col`/`current_load_col`/`utilisation_col`
            were registered via `add_sites()`:

            - ``capacity`` -- present iff `capacity_col` was registered.
            - ``current_load`` -- present iff `current_load_col` was
              registered (the raw-counts input path only).
            - ``utilisation_ratio`` -- always present. Either
              `current_load / capacity` (raw-counts path), or
              `utilisation_col`'s values as-is (precomputed-ratio path).
            - ``headroom`` -- present iff derivable: `capacity -
              current_load` (raw-counts path), or `capacity * (1 -
              utilisation_ratio)` (precomputed-ratio path, only when
              `capacity_col` was also registered). Omitted otherwise.

        Raises
        ------
        ValueError
            If neither `current_load_col`+`capacity_col` nor
            `utilisation_col` was registered via `add_sites()`, or if both
            `site_names` and `site_indices` are given.
        IndexError
            If `site_indices` contains a value not in `candidate_sites`.
        KeyError
            If `site_names` contains a name not in `candidate_sites`.

        Notes
        -----
        Every requested site appears as an explicit row, but a site with no
        registered baseline data (typically: not yet built) gets `NaN` in
        `utilisation_ratio`/`headroom`, not `0.0` -- an important
        distinction to preserve, since `0.0` would instead mean "measured,
        and currently idle", an alarming finding for an *operating* site.
        Values above 1.0 (genuinely over capacity) are left as-is, not
        clipped -- a negative `headroom` is itself the finding. This method
        does not sort its result; use e.g.
        `.sort_values("utilisation_ratio", ascending=False)` to rank sites
        by fullness.
        """
        if site_names is not None and site_indices is not None:
            raise ValueError(
                "Please provide at most one of 'site_names' or "
                "'site_indices', not both."
            )

        has_raw_counts = (
            self._candidate_sites_current_load_col is not None
            and self._candidate_sites_capacity_col is not None
        )
        has_precomputed_ratio = self._candidate_sites_utilisation_col is not None

        if not has_raw_counts and not has_precomputed_ratio:
            raise ValueError(
                "site_utilisation_summary() requires baseline utilisation "
                "data. Call add_sites(..., current_load_col=..., "
                "capacity_col=...) to supply raw current-load/capacity "
                "counts, or add_sites(..., utilisation_col=...) to supply a "
                "precomputed utilisation ratio."
            )

        if site_indices is not None:
            if len(site_indices) != len(set(site_indices)):
                raise ValueError(
                    f"site_indices contains duplicate entries: {site_indices}."
                )
            valid_indices = set(self.candidate_sites["canonical_site_index"])
            invalid_indices = sorted(set(site_indices) - valid_indices)
            if invalid_indices:
                raise IndexError(
                    f"Site indices {invalid_indices} not found in candidate "
                    f"sites (valid range: 0 to {self.total_n_sites - 1})."
                )
            site_names_resolved = (
                self.candidate_sites[
                    self.candidate_sites["canonical_site_index"].isin(site_indices)
                ]
                .sort_values("canonical_site_index")[
                    self._candidate_sites_candidate_id_col
                ]
                .tolist()
            )
        elif site_names is not None:
            if len(site_names) != len(set(site_names)):
                raise ValueError(
                    f"site_names contains duplicate entries: {site_names}."
                )
            valid_names = set(
                self.candidate_sites[self._candidate_sites_candidate_id_col]
            )
            missing = [s for s in site_names if s not in valid_names]
            if missing:
                raise KeyError(f"Sites not found in candidate_sites: {missing}.")
            site_names_resolved = list(site_names)
        else:
            site_names_resolved = (
                self.candidate_sites.sort_values("canonical_site_index")[
                    self._candidate_sites_candidate_id_col
                ]
                .tolist()
            )

        selected = self.candidate_sites.set_index(
            self._candidate_sites_candidate_id_col
        ).loc[site_names_resolved]

        result = pd.DataFrame(index=pd.Index(site_names_resolved, name="site"))

        if self._candidate_sites_capacity_col is not None:
            result["capacity"] = selected[self._candidate_sites_capacity_col].to_numpy()

        if has_raw_counts:
            result["current_load"] = selected[
                self._candidate_sites_current_load_col
            ].to_numpy()
            result["utilisation_ratio"] = result["current_load"] / result["capacity"]
            result["headroom"] = result["capacity"] - result["current_load"]
        else:
            result["utilisation_ratio"] = selected[
                self._candidate_sites_utilisation_col
            ].to_numpy()
            if self._candidate_sites_capacity_col is not None:
                result["headroom"] = result["capacity"] * (
                    1 - result["utilisation_ratio"]
                )

        return result

    def plot_site_utilisation(
        self,
        utilisation_df=None,
        site_names=None,
        site_indices=None,
        interactive=False,
        cmap="RdYlGn_r",
        missing_site_colour="lightgrey",
        marker_size_range=(40, 220),
        show_labels=False,
        add_basemap=True,
        tiles="CartoDB positron",
        title=None,
        caption=None,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Map each candidate site's current utilisation ratio.

        Site markers are coloured and sized by `utilisation_ratio` from
        `site_utilisation_summary()`, so an analyst can see at a glance
        which sites are already near or over capacity *today* -- entirely
        independent of `solve()` or any catchment/demand modelling.

        Parameters
        ----------
        utilisation_df : pandas.DataFrame, optional
            A precomputed result from `site_utilisation_summary()`. If
            None (the default), it is computed automatically using
            `site_names`/`site_indices`.
        site_names, site_indices
            Forwarded to `site_utilisation_summary()` when `utilisation_df`
            is not supplied; see that method's docstring (mutually
            exclusive).
        interactive : bool, default False
            If True, returns an interactive Folium map via `.explore()`.
            Otherwise returns a static matplotlib Axes.
        cmap : str, default "RdYlGn_r"
            Colormap for site markers (`utilisation_ratio`). Deliberately
            the *reversed* variant of `plot_accessibility`'s `site_cmap`
            default ("RdYlGn"): there, a high ratio is good (uncontested
            supply); here, a high ratio is bad (near/at/over capacity), so
            red must map to the high end and green to the low end.
        missing_site_colour : str, default "lightgrey"
            Colour (and static marker size, at the smallest of
            `marker_size_range`) for a site with no baseline utilisation
            data -- typically a not-yet-built proposal.
        marker_size_range : tuple of (float, float), default (40, 220)
            Smallest and largest static marker size, linearly scaled by
            `utilisation_ratio`. Note this is the *opposite* sizing
            convention to `plot_accessibility`: there, a low (bad) ratio is
            drawn small; here, a high (bad) ratio is drawn large, so a
            hotspot is easier to spot on the map. Ignored on interactive
            maps, where Folium markers are a fixed size.
        show_labels : bool, default False
            If True, adds text labels for each site (see `plot_sites`).
            Default False since colour/size already carry the signal.
        add_basemap : bool, default True
            If True, adds a background web map. Set False to skip the tile
            download entirely.
        title : str, optional
            Axes title. Ignored on interactive maps.
        caption : str or None, default None
            `None` prints a short "how to read this" explanation of the
            marker colour/size below the chart; pass `""` to suppress it
            or a custom string to replace it, matching the existing
            `caption` convention on `plot_accessibility()` /
            `plot_pareto_summary()` / `plot_site_reallocation_matrix()` /
            `plot_population_impact_histogram()`. Static branch only.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto. Ignored if `interactive=True`.
        figsize : tuple, optional
            Passed to `plt.subplots()` if `ax` is not supplied. Ignored if
            `interactive=True`.
        **kwargs : dict
            Additional keyword arguments passed to the site plotting call
            (`GeoDataFrame.plot`/`.explore`).

        Returns
        -------
        matplotlib.axes.Axes or folium.Map

        Raises
        ------
        ValueError
            If `candidate_sites` has no real geometry (i.e. `add_sites()`
            was never given a GeoDataFrame or lat/long columns), or if
            `utilisation_df` is not supplied and neither
            `current_load_col`+`capacity_col` nor `utilisation_col` was
            registered via `add_sites()` (see `site_utilisation_summary()`).
        """
        if not isinstance(self.candidate_sites, geopandas.GeoDataFrame):
            raise ValueError(
                "plot_site_utilisation() requires real site geometry -- "
                "add_sites() must have been given a GeoDataFrame or "
                "lat/long columns, not a bare site list inferred from the "
                "travel matrix's column names."
            )

        if utilisation_df is None:
            utilisation_df = self.site_utilisation_summary(
                site_names=site_names, site_indices=site_indices
            )

        site_gdf = self.candidate_sites.merge(
            utilisation_df.reset_index(),
            left_on=self._candidate_sites_candidate_id_col,
            right_on="site",
            suffixes=("", "_y"),
        )
        site_gdf = site_gdf.drop(site_gdf.filter(regex="_y$").columns, axis=1)

        min_size, max_size = marker_size_range
        site_gdf["_marker_size"] = (
            min_size
            + _min_max_normalize(site_gdf["utilisation_ratio"], constant_fill=1.0)
            * (max_size - min_size)
        ).fillna(min_size)

        tooltip_cols = [
            c
            for c in (
                self._candidate_sites_candidate_id_col,
                "capacity",
                "current_load",
                "utilisation_ratio",
                "headroom",
            )
            if c in site_gdf.columns
        ]

        if interactive:
            m = site_gdf.explore(
                column="utilisation_ratio",
                cmap=cmap,
                tooltip=tooltip_cols,
                popup=True,
                tiles=tiles if add_basemap else None,
                marker_kwds=dict(radius=8),
                missing_kwds=dict(color=missing_site_colour),
                **kwargs,
            )
            return _attach_deferred_fit_bounds(m)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        site_gdf.plot(
            column="utilisation_ratio",
            cmap=cmap,
            markersize=site_gdf["_marker_size"],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            legend=True,
            legend_kwds={"label": "Utilisation ratio", "shrink": 0.6},
            missing_kwds=dict(color=missing_site_colour, label="No baseline data"),
            **kwargs,
        )

        if show_labels:
            texts = []
            for x, y, label in zip(
                site_gdf.geometry.x,
                site_gdf.geometry.y,
                site_gdf[self._candidate_sites_candidate_id_col],
            ):
                wrapped_label = textwrap.fill(label, 15).title()
                texts.append(ax.text(x, y, wrapped_label))
            adjust_text(texts, force_explode=(0.05, 0.05), ax=ax)

        # geopandas' missing_kwds `label` only surfaces in a discrete
        # `scheme=` legend, not the continuous colorbar used here, so the
        # grey "no baseline data" markers would otherwise have no legend
        # entry explaining them.
        if site_gdf["utilisation_ratio"].isna().any():
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=missing_site_colour,
                        markeredgecolor="black",
                        markersize=8,
                        label="No baseline data",
                    )
                ],
                loc="lower left",
            )

        if title is not None:
            ax.set_title(title)

        if add_basemap:
            try:
                cx.add_basemap(ax, crs=site_gdf.crs.to_string(), timeout=30)
            except RequestException as e:
                warnings.warn(
                    f"Unable to download background map tiles ({type(e).__name__}). "
                    "Continuing without a basemap.",
                    stacklevel=2,
                )

        default_caption = (
            "Markers are coloured (and sized) by each site's current "
            "real-world utilisation ratio -- today's baseline, independent "
            "of any solve() or catchment/demand modelling."
        )
        _add_plot_caption(fig, caption, default_caption)

        return ax

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
        self._candidate_sites_cost_col = None
        self._candidate_sites_current_load_col = None
        self._candidate_sites_utilisation_col = None
        self.total_n_sites = len(self.candidate_sites)

    #############################
    # MARK: Travel Matrix
    #############################
    def add_travel_matrix(
        self,
        travel_matrix_df,
        source_col,
        skip_cols=None,
        unit=None,
        from_unit=None,
        to_unit=None,
        allow_missing=False,
        treat_as_missing=None,
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
        allow_missing : bool, default False
            Whether to allow missing (NaN) travel costs -- a demand
            location with no feasible journey to a given site (e.g. no
            public transport route within a permissive search radius). By
            default a missing value raises `ValueError`, since an unnoticed
            NaN usually means an ID mismatch or a botched generation run.
            Once `allow_missing=True`, `min_cost`/`weighted_average`/etc.
            skip unreachable rows rather than propagating NaN into every
            metric, and separate `regions_unreachable`/`demand_unreachable`
            metrics report how many/how much was excluded. `solve()` does
            not yet support optimising over a matrix with missing values --
            see its own error for the current workaround.
        treat_as_missing : scalar or callable, optional
            An existing missing-data sentinel already baked into
            `travel_matrix_df` (e.g. `9999`), converted to a proper missing
            value before anything else runs. A scalar is matched by
            equality; a callable is applied to every cost cell and should
            return `True` for values that mean "missing" (e.g. `lambda v: v
            >= 9000`). Implies data containing missing values, so also pass
            `allow_missing=True` unless the conversion is expected to find
            nothing.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the `source_col` is missing from the provided dataframe, or
            if the matrix contains missing (NaN) travel costs and
            `allow_missing` was not set.
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

        loaded_df = _apply_treat_as_missing(loaded_df, source_col, treat_as_missing)
        _reject_missing_travel_values(
            loaded_df,
            loaded_df.columns.drop(source_col),
            allow_missing,
            "add_travel_matrix()",
        )

        loaded_df, self._travel_matrix_unit = _apply_unit_conversion(
            loaded_df, unit, from_unit, to_unit
        )

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

    #############################
    # MARK: Secondary Travel Matrices
    #############################
    def add_secondary_travel_matrix(
        self,
        travel_matrix_df,
        source_col,
        label,
        skip_cols=None,
        unit=None,
        from_unit=None,
        to_unit=None,
        threshold_for_coverage=None,
        allow_missing=False,
        treat_as_missing=None,
    ):
        """
        Register an additional travel/cost matrix for a different mode or
        scenario (e.g. public transport alongside a primary car matrix).

        Unlike `add_travel_matrix()`, a secondary matrix is never used as the
        optimisation cost matrix -- the primary matrix set via
        `add_travel_matrix()` always drives site selection and search/pruning.
        Instead, each registered secondary matrix contributes its own set of
        per-solution metric columns (suffixed `__<label>`, e.g.
        `min_cost__public_transport`, `weighted_average__public_transport`)
        into every solution `solve()` produces, so it can be used directly in
        plots, `Metric(column=...)`, and post-hoc ranking
        (`sort_by="max__public_transport"`) -- without needing to `.copy()`
        the problem and solve it twice.

        Post-hoc ranking reorders whatever `solve()` already returned, so it
        only reorders candidates that survived the *primary* matrix's search
        and pruning. If `brute_force_keep_best_n` (or `_worst_n`) is set,
        candidates were discarded on primary-matrix performance before
        secondary metrics were ever considered -- so a secondary ranking over
        what survives is not the true best solution for that mode. To search
        on a secondary matrix rather than merely re-sort by it, pass the
        column to `solve(rank_on=...)`, which makes the pruning itself use
        that metric. Alternatively, retain every combination (no
        `brute_force_keep_best_n`/`_worst_n`) and use `compute_pareto_front()`
        to explore the trade-off across both matrices at once.

        You may call this method multiple times with different `label`s to
        register as many secondary matrices as needed. Each one adds
        approximately the per-candidate cost of evaluating the primary
        matrix, so `solve()` time scales close to linearly with the number
        of secondary matrices registered. Under `n_jobs > 1`, every
        registered matrix's aligned frame is pickled to each worker process
        alongside the problem, so peak memory also scales with matrix count.

        Parameters
        ----------
        travel_matrix_df : pandas.DataFrame or geopandas.GeoDataFrame or str
            The dataset containing travel costs for this matrix, or a local
            or web path to its location.
        source_col : str
            The column name in `travel_matrix_df` that identifies the origin
            points (should correspond to IDs in the demand data).
        label : str
            A unique label identifying this matrix, used to suffix every
            metric column it produces (e.g. `label="public_transport"` ->
            `min_cost__public_transport`). Must be non-empty, must not
            contain `"__"`, and must not already be registered.
        skip_cols : list of str, optional
            A list of column names to ignore during data loading.
        unit : str, optional
            A label for the units used in the matrix. Used if no conversion
            is performed.
        from_unit : {"seconds", "minutes", "hours"}, optional
            The current time unit of the numeric values in the dataframe.
        to_unit : {"seconds", "minutes", "hours"}, optional
            The target time unit for the numeric values in the dataframe.
        threshold_for_coverage : float, optional
            The coverage threshold to apply to this matrix specifically
            (e.g. 60 minutes for public transport vs 20 for car). If not
            provided, falls back to the `threshold_for_coverage` passed to
            `solve()` / `evaluate_single_solution_single_objective()`.
        allow_missing : bool, default False
            Whether to allow missing (NaN) travel costs -- a demand
            location with no feasible journey to a given site. By default a
            missing value raises `KeyError` once `solve()` builds this
            matrix's aligned frame (see Notes below); `allow_missing=True`
            treats it as genuinely unreachable instead, the same as
            `add_travel_matrix(allow_missing=True)`.
        treat_as_missing : scalar or callable, optional
            An existing missing-data sentinel already baked into
            `travel_matrix_df` (e.g. `9999`), converted to a proper missing
            value before anything else runs -- see
            `add_travel_matrix(treat_as_missing=...)`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `label` is empty, contains `"__"`, or has already been
            registered.
        KeyError
            If `source_col` is missing from the provided dataframe.

        Notes
        -----
        Secondary matrices must be complete: every demand location and every
        candidate site must have a row/column, and (unless `allow_missing=
        True`) a non-missing value. This is validated when `solve()` builds
        the aligned per-solution frames (not here, since demand/sites may
        not yet be registered) -- see `solve()` for the specific errors
        raised for missing rows or missing columns.
        """
        if not label or not isinstance(label, str):
            raise ValueError(
                "label must be a non-empty string identifying this secondary "
                "travel matrix (e.g. 'public_transport')."
            )
        if "__" in label:
            raise ValueError(
                f"label '{label}' must not contain '__' -- this sequence is "
                "reserved to separate a metric name from its matrix label "
                "(e.g. 'min_cost__public_transport')."
            )
        if label in self.secondary_travel_matrices:
            raise ValueError(
                f"A secondary travel matrix labelled '{label}' has already "
                "been registered. Registered labels: "
                f"{sorted(self.secondary_travel_matrices)}. Choose a "
                "different label."
            )
        if label in self.secondary_demand_matrices:
            raise ValueError(
                f"label '{label}' is already registered as a secondary "
                "demand scenario. Secondary travel matrices and secondary "
                "demand scenarios share the same suffix namespace, so "
                "labels must be unique across both. Choose a different "
                "label."
            )

        loaded_df, df_type = _load_spatial_or_tabular_data(
            travel_matrix_df, skip_cols=skip_cols
        )

        _validate_columns(
            df=loaded_df,
            col_names=[source_col],
            msg_template=(
                "It looks like your secondary travel matrix data is missing these "
                "columns: {missing}. We found these instead: {available}. Please "
                "double-check the column names you are passing to the "
                ".add_secondary_travel_matrix() method."
            ),
        )

        loaded_df = _apply_treat_as_missing(loaded_df, source_col, treat_as_missing)

        loaded_df, resolved_unit = _apply_unit_conversion(
            loaded_df, unit, from_unit, to_unit
        )

        self.secondary_travel_matrices[label] = {
            "data": loaded_df,
            "source_col": source_col,
            "unit": resolved_unit,
            "threshold_for_coverage": threshold_for_coverage,
            "allow_missing": allow_missing,
        }

    def show_secondary_travel_matrix(self, label):
        """
        Returns a registered secondary travel/cost matrix dataframe.

        Parameters
        ----------
        label : str
            The label the matrix was registered under via
            `add_secondary_travel_matrix()`.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        KeyError
            If no secondary matrix has been registered under `label`.
        """
        if label not in self.secondary_travel_matrices:
            raise KeyError(
                f"No secondary travel matrix registered under label '{label}'. "
                f"Registered labels: {sorted(self.secondary_travel_matrices)}."
            )
        return self.secondary_travel_matrices[label]["data"]

    def _build_secondary_travel_frames(self):
        """
        Build, for each registered secondary matrix, a DataFrame aligned to
        `self.travel_and_demand_df.index` with one column per candidate site
        (by name). Stored in `self._secondary_travel_frames[label]`.

        Reindexing (rather than merging) guarantees row alignment to the
        primary demand/travel index and cannot duplicate rows. Every
        secondary matrix must have every demand location present as a row
        and every candidate site present as a column -- a partially
        populated matrix would silently compute its averages over a
        different denominator than the primary matrix, making the two
        non-comparable. Missing (NaN) *values* within those cells are
        rejected too, unless the matrix was registered with
        `add_secondary_travel_matrix(allow_missing=True)`, in which case
        they are kept as genuinely-unreachable travel costs (see
        `EvaluatedCombination._compute_travel_metrics`).
        """
        self._secondary_travel_frames = {}

        if not self.secondary_travel_matrices:
            return

        demand_index = self.travel_and_demand_df.index
        site_names = self.candidate_sites[
            self._candidate_sites_candidate_id_col
        ].tolist()

        for label, meta in self.secondary_travel_matrices.items():
            matrix = meta["data"]
            source_col = meta["source_col"]

            missing_demand_ids = demand_index.difference(matrix[source_col])
            if len(missing_demand_ids) > 0:
                raise KeyError(
                    f"Secondary travel matrix '{label}' is missing "
                    f"{len(missing_demand_ids)} of {len(demand_index)} demand "
                    f"location(s) present in the primary travel matrix (e.g. "
                    f"{list(missing_demand_ids[:5])}). Every demand location "
                    "must have a row in every secondary matrix, or metrics "
                    "computed from it would silently cover a different set "
                    "of demand locations than the primary matrix."
                )

            missing_sites = [s for s in site_names if s not in matrix.columns]
            if missing_sites:
                raise KeyError(
                    f"Secondary travel matrix '{label}' is missing these "
                    f"candidate sites as columns: {missing_sites}. Every "
                    "candidate site must have a column in every secondary "
                    "matrix."
                )

            frame = matrix.set_index(source_col).reindex(demand_index)[site_names]

            if not meta.get("allow_missing", False):
                nan_mask = frame.isna().any(axis=1)
                if nan_mask.any():
                    nan_ids = frame.index[nan_mask].tolist()
                    raise KeyError(
                        f"Secondary travel matrix '{label}' has {len(nan_ids)} "
                        f"demand location(s) with a missing value for at least "
                        f"one candidate site (e.g. {nan_ids[:5]}). This usually "
                        "means a route is missing for that origin-destination "
                        "pair (e.g. no public transport route). Register this "
                        "matrix with add_secondary_travel_matrix(allow_missing="
                        "True) to treat these as genuinely unreachable, or "
                        "restrict the demand set to locations this matrix "
                        "covers."
                    )

            self._secondary_travel_frames[label] = frame

    #############################
    # MARK: Secondary Demand Matrices
    #############################
    def add_secondary_demand(
        self,
        demand_df,
        demand_col,
        location_id_col,
        label,
        skip_cols=None,
        also_weight_matrices=None,
    ):
        """
        Register an additional demand scenario (e.g. projected future
        demand alongside a primary current-demand dataset).

        Unlike `add_demand()`, a secondary demand scenario never drives the
        optimisation objective -- the primary demand set via `add_demand()`
        (or the equal-demand fallback) always drives site selection and
        search/pruning. Instead, each registered secondary demand scenario
        contributes its own `weighted_average__<label>` and
        `proportion_within_coverage_threshold__<label>` metric columns --
        the two metrics that actually change with demand -- into every
        solution `solve()` produces, so it can be used directly in plots,
        `Metric(column=...)`, ranking (`sort_by=
        "weighted_average__future_demand"` post-hoc, or
        `solve(rank_on="weighted_average__future_demand")` to search on it
        directly), or blended into the optimisation objective via
        `weights={"future_demand": ...}`.

        By default a secondary demand scenario only re-weights the primary
        travel matrix. Pass `also_weight_matrices` to additionally compute
        `weighted_average__<travel_label>__<label>` /
        `proportion_within_coverage_threshold__<travel_label>__<label>`
        against one or more registered secondary travel matrices -- this is
        opt-in because the combination of secondary travel and secondary
        demand matrices grows the output multiplicatively.

        You may call this method multiple times with different `label`s to
        register as many demand scenarios as needed. Each one adds
        approximately the cost of one extra weighted-average/coverage pass
        over the primary matrix (plus one per named `also_weight_matrices`
        entry), so `solve()` time scales close to linearly with the number
        of secondary demand scenarios registered.

        Parameters
        ----------
        demand_df : pandas.DataFrame or geopandas.GeoDataFrame or str
            The dataset containing demand for this scenario, or a local or
            web path to its location.
        demand_col : str
            The name of the column in `demand_df` representing the quantity
            of demand for this scenario.
        location_id_col : str
            The name of the column in `demand_df` used as a unique
            identifier for demand locations (should correspond to IDs in
            the primary demand data).
        label : str
            A unique label identifying this demand scenario, used to
            suffix every metric column it produces (e.g.
            `label="future_demand"` -> `weighted_average__future_demand`).
            Must be non-empty, must not contain `"__"`, and must not
            already be registered as either a secondary demand scenario or
            a secondary travel matrix (suffix labels are shared between
            the two).
        skip_cols : list of str, optional
            A list of column names to ignore during data loading.
        also_weight_matrices : list of str, optional
            Labels of registered secondary travel matrices (via
            `add_secondary_travel_matrix()`) to also weight by this demand
            scenario, producing `<metric>__<travel_label>__<label>`
            columns. Order of registration does not matter -- membership
            is checked when `solve()` builds the aligned frames.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `label` is empty, contains `"__"`, or has already been
            registered (as a secondary demand scenario or secondary travel
            matrix).
        KeyError
            If `demand_col` or `location_id_col` is missing from the
            provided dataframe.

        Notes
        -----
        Secondary demand scenarios must be complete: every demand location
        present in the primary demand/travel data must have a
        non-missing value. This is validated when `solve()` builds the
        aligned per-solution frames (not here, since the primary
        demand/travel matrix may not yet be registered).
        """
        if not label or not isinstance(label, str):
            raise ValueError(
                "label must be a non-empty string identifying this "
                "secondary demand scenario (e.g. 'future_demand')."
            )
        if "__" in label:
            raise ValueError(
                f"label '{label}' must not contain '__' -- this sequence "
                "is reserved to separate a metric name from its "
                "scenario/matrix label (e.g. "
                "'weighted_average__future_demand')."
            )
        if label in self.secondary_demand_matrices:
            raise ValueError(
                f"A secondary demand scenario labelled '{label}' has "
                "already been registered. Registered labels: "
                f"{sorted(self.secondary_demand_matrices)}. Choose a "
                "different label."
            )
        if label in self.secondary_travel_matrices:
            raise ValueError(
                f"label '{label}' is already registered as a secondary "
                "travel matrix. Secondary demand scenarios and secondary "
                "travel matrices share the same suffix namespace, so "
                "labels must be unique across both. Choose a different "
                "label."
            )

        loaded_df, df_type = _load_spatial_or_tabular_data(
            demand_df, skip_cols=skip_cols
        )

        _validate_columns(
            df=loaded_df,
            col_names=[demand_col, location_id_col],
            msg_template=(
                "It looks like your secondary demand data is missing "
                "these columns: {missing}. We found these instead: "
                "{available}. Please double-check the column names you "
                "are passing to the .add_secondary_demand() method."
            ),
        )

        self.secondary_demand_matrices[label] = {
            "data": loaded_df[[location_id_col, demand_col]],
            "demand_col": demand_col,
            "id_col": location_id_col,
            "also_weight_matrices": (
                list(also_weight_matrices) if also_weight_matrices else []
            ),
        }

    def show_secondary_demand(self, label):
        """
        Returns a registered secondary demand scenario dataframe.

        Parameters
        ----------
        label : str
            The label the scenario was registered under via
            `add_secondary_demand()`.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        KeyError
            If no secondary demand scenario has been registered under
            `label`.
        """
        if label not in self.secondary_demand_matrices:
            raise KeyError(
                f"No secondary demand scenario registered under label "
                f"'{label}'. Registered labels: "
                f"{sorted(self.secondary_demand_matrices)}."
            )
        return self.secondary_demand_matrices[label]["data"]

    def _build_secondary_demand_frames(self):
        """
        Build, for each registered secondary demand scenario, a Series of
        demand values aligned to `self.travel_and_demand_df.index`. Stored
        in `self._secondary_demand_frames[label]`.

        Also validates every label in that scenario's
        `also_weight_matrices` against `self.secondary_travel_matrices`,
        so an unregistered or mistyped travel-matrix label is caught here
        rather than surfacing as a missing-column error deep inside metric
        computation.
        """
        self._secondary_demand_frames = {}

        if not self.secondary_demand_matrices:
            return

        demand_index = self.travel_and_demand_df.index

        for label, meta in self.secondary_demand_matrices.items():
            for travel_label in meta["also_weight_matrices"]:
                if travel_label not in self.secondary_travel_matrices:
                    raise KeyError(
                        f"Secondary demand scenario '{label}' names "
                        f"'{travel_label}' in also_weight_matrices, but no "
                        "secondary travel matrix is registered under that "
                        "label. Registered secondary travel matrices: "
                        f"{sorted(self.secondary_travel_matrices)}."
                    )

            series = meta["data"].set_index(meta["id_col"])[meta["demand_col"]]

            missing_demand_ids = demand_index.difference(series.index)
            if len(missing_demand_ids) > 0:
                raise KeyError(
                    f"Secondary demand scenario '{label}' is missing "
                    f"{len(missing_demand_ids)} of {len(demand_index)} "
                    "demand location(s) present in the primary demand "
                    f"data (e.g. {list(missing_demand_ids[:5])}). Every "
                    "demand location must have a row in every secondary "
                    "demand scenario, or metrics computed from it would "
                    "silently cover a different set of demand locations "
                    "than the primary scenario."
                )

            series = series.reindex(demand_index).astype(float)

            if series.isna().any():
                nan_ids = series.index[series.isna()].tolist()
                raise KeyError(
                    f"Secondary demand scenario '{label}' has "
                    f"{len(nan_ids)} demand location(s) with a missing "
                    f"value (e.g. {nan_ids[:5]}). Fill or remove these "
                    "rows before registering the scenario."
                )

            self._secondary_demand_frames[label] = series

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
        if self.travel_matrix is None:
            raise ValueError(
                "No travel matrix or other cost matrix has been provided, "
                "so demand locations cannot be linked to sites. Please add "
                "one using the .add_travel_matrix() method and try again."
            )

        if self.demand_data is None:
            raise ValueError(
                "No demand data has been provided, so demand locations "
                "cannot be linked to sites. Please add it using the "
                ".add_demand() method and try again."
            )

        # If one is a geopandas dataframe, put that first in the merge call so that the
        # output object will also be a geodataframe
        if self._demand_data_type == "geopandas":
            travel_and_demand_df = pd.merge(
                self.demand_data,
                self.travel_matrix,
                left_on=self._demand_data_id_col,
                right_on=self._travel_matrix_source_col,
                how="inner",
                suffixes=("", "_y"),
            )

            travel_and_demand_df = travel_and_demand_df.drop(
                travel_and_demand_df.filter(regex="_y$").columns, axis=1
            )

            self._joined_demand_travel_df_key_col = self._demand_data_id_col

        else:
            travel_and_demand_df = pd.merge(
                self.travel_matrix,
                self.demand_data,
                left_on=self._travel_matrix_source_col,
                right_on=self._demand_data_id_col,
                how="inner",
                suffixes=("", "_y"),
            )

            travel_and_demand_df = travel_and_demand_df.drop(
                travel_and_demand_df.filter(regex="_y$").columns, axis=1
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
