import pandas as pd
from lokigi.plot_utils import _attach_deferred_fit_bounds
from lokigi.utils import (
    _load_spatial_or_tabular_data,
    GEOPANDAS_EXTS,
    _check_crs_match_pref,
    _convert_crs,
    _reject_removed_basemap_alias,
)
from warnings import warn
import hashlib
from typing import Literal
import contextily as cx
import warnings
from requests.exceptions import RequestException


class _Problem:
    """Shared infrastructure."""

    def __init__(self, preferred_crs="EPSG:27700", debug_mode=True, **kwargs):
        self.preferred_crs = preferred_crs

        self.travel_matrix = None  # Travel time/distance matrix
        self._travel_matrix_type = None
        self._travel_matrix_source_col = None
        self._travel_matrix_unit = None

        # label -> {"data", "source_col", "unit", "threshold_for_coverage"}
        self.secondary_travel_matrices = {}
        # label -> aligned DataFrame (built at solve()/evaluate time)
        self._secondary_travel_frames = {}

        # label -> {"data", "demand_col", "id_col", "also_weight_matrices"}
        self.secondary_demand_matrices = {}
        # label -> aligned Series (built at solve()/evaluate time)
        self._secondary_demand_frames = {}

        self.region_geometry_layer = None
        self._region_geometry_layer_type = None
        self._region_geometry_layer_common_col = None
        self._region_geometry_hash = None

        self.spatial_weights = None
        self.spatial_weights_method = None
        self.spatial_weights_k = None

        # label -> {"data", "equity_col", "common_col", "label",
        #           "disadvantaged_end", "data_type"}. Insertion-ordered, so
        # `_primary_equity_label` is the only thing deciding which dimension
        # drives solve()'s metrics -- never registration order.
        self.equity_datasets = {}
        self._primary_equity_label = None

        self.geo_lookup = None
        self._geo_lookup_data_type = None
        self._geo_lookup_common_col = None

        self.additional_data = None
        self._additional_data_labels = None

        if debug_mode:
            self._verbose = True
        else:
            self._verbose = False

    # Flat equity attributes as written by versions before the equity
    # registry (<= v0.9.1), in the order add_equity_data() used to set them.
    _LEGACY_EQUITY_KEYS = (
        "equity_data",
        "_equity_data_type",
        "_equity_data_equity_col",
        "_equity_data_common_col",
        "_equity_data_label",
        "_equity_data_disadvantaged_end",
    )

    def __setstate__(self, state):
        """
        Restore a pickled problem, migrating pre-registry pickles.

        Solved problems and solution sets get pickled and reloaded in
        practice -- an app that solves offline and ships the result is the
        normal pattern -- so a pickle written by <= v0.9.1 carries the flat
        `_equity_data_*` attributes and no `equity_datasets`. Those
        attributes are now read-only properties over the registry, and a
        property is a data descriptor: it wins over whatever the unpickled
        `__dict__` holds. So without this, restoring an old pickle appears
        to succeed and then raises a bare `AttributeError` naming
        `_primary_equity_label` -- an internal attribute the user never
        set -- the first time anything touches its equity data.

        Rebuilding the single registered dimension from those attributes
        keeps the equity analysis working across the upgrade, rather than
        merely failing more politely.
        """
        if "equity_datasets" not in state:
            legacy = {key: state.pop(key, None) for key in self._LEGACY_EQUITY_KEYS}

            state["equity_datasets"] = {}
            state["_primary_equity_label"] = None

            if legacy["equity_data"] is not None and legacy["_equity_data_equity_col"]:
                # `label` was optional on the old call, so fall back to the
                # column name -- `_resolve_equity()` keys off the label and
                # cannot register a dimension under `None`.
                label = (
                    legacy["_equity_data_label"] or legacy["_equity_data_equity_col"]
                )
                state["equity_datasets"][label] = {
                    "data": legacy["equity_data"],
                    "data_type": legacy["_equity_data_type"] or "pandas",
                    "equity_col": legacy["_equity_data_equity_col"],
                    "common_col": legacy["_equity_data_common_col"],
                    "label": label,
                    "disadvantaged_end": legacy["_equity_data_disadvantaged_end"]
                    or "low",
                }
                state["_primary_equity_label"] = label
        else:
            # A current pickle should not also carry the flat attributes,
            # but drop them if it does rather than leaving keys in
            # `__dict__` that are permanently shadowed by the properties
            # and would mislead anyone inspecting the instance.
            for key in self._LEGACY_EQUITY_KEYS:
                state.pop(key, None)

        self.__dict__.update(state)

    ###############################
    # MARK: Equity registry
    ###############################

    @property
    def primary_equity(self):
        """
        The registered equity dimension that drives `solve()`'s metrics, or
        `None` if none is registered.

        Every equity dimension registered via `add_equity_data()` is merged
        into each solution's `problem_df` and can be selected for analysis
        by label, but exactly one is the *primary* -- the one whose bands
        the `solution_df` equity columns (`inter_tertile_ratio`,
        `weighted_by_equity_group`, ...) are computed against, and the one
        `weights={"equity": ...}` weights by. Set at registration via
        `add_equity_data(primary=...)`.
        """
        if self._primary_equity_label is None:
            return None
        return self.equity_datasets[self._primary_equity_label]

    # The five attributes below were plain instance attributes before the
    # equity registry existed, and are read in ~50 places across the
    # solvers, metrics and plots. They are kept as read-only views onto the
    # primary dimension so those call sites keep meaning "the dimension
    # this problem is optimised and reported against" without each having
    # to resolve the registry itself. Read-only deliberately: writing one
    # of them used to half-configure a problem (e.g. a new equity_col with
    # a stale common_col), which the registry makes impossible.

    @property
    def equity_data(self):
        primary = self.primary_equity
        return None if primary is None else primary["data"]

    @property
    def _equity_data_equity_col(self):
        primary = self.primary_equity
        return None if primary is None else primary["equity_col"]

    @property
    def _equity_data_common_col(self):
        primary = self.primary_equity
        return None if primary is None else primary["common_col"]

    @property
    def _equity_data_label(self):
        primary = self.primary_equity
        return None if primary is None else primary["label"]

    @property
    def _equity_data_disadvantaged_end(self):
        primary = self.primary_equity
        return None if primary is None else primary["disadvantaged_end"]

    @property
    def _equity_data_type(self):
        primary = self.primary_equity
        return None if primary is None else primary["data_type"]

    def _resolve_primary_flag(self, label, primary, replace):
        """
        Decide whether an `add_equity_data()` call registers the primary
        dimension, and reject the ambiguous cases.

        The awkward case is a second call on a problem that already has a
        dimension registered. Before the registry existed, that overwrote
        the first -- so code written against the old behaviour expects the
        *second* call's dimension to be the one every metric is computed
        against. Silently registering it alongside and leaving the first as
        primary would keep that code running while quietly changing which
        bands its numbers describe, so `primary` is required to be explicit
        there. First registrations, and re-registrations of a label that
        already exists, are unambiguous and need no flag.
        """
        if replace and label not in self.equity_datasets:
            raise ValueError(
                f"replace=True was given but no equity dimension '{label}' is "
                "registered. "
                + (
                    f"Registered: {sorted(self.equity_datasets)}."
                    if self.equity_datasets
                    else "No equity data has been registered yet -- drop "
                    "replace=True to register this as the first dimension."
                )
            )

        if not self.equity_datasets:
            if primary is False:
                raise ValueError(
                    f"'{label}' is the only equity dimension registered, so it "
                    "cannot be registered with primary=False -- something has "
                    "to drive the equity metrics. Drop the argument, or pass "
                    "primary=True."
                )
            return True

        if label in self.equity_datasets:
            if not replace:
                raise ValueError(
                    f"An equity dimension labelled '{label}' is already "
                    "registered. Pass replace=True to overwrite it, or use a "
                    "different label to register an additional dimension "
                    "alongside it."
                )
            # Re-registering an existing label keeps its current primary
            # status unless the caller says otherwise -- replacing the data
            # behind a label shouldn't silently move the primary elsewhere.
            if primary is None:
                return self._primary_equity_label == label
            return primary

        if primary is None:
            raise ValueError(
                f"'{self._primary_equity_label}' is already registered as the "
                f"primary equity dimension, so adding '{label}' needs "
                "primary= stated explicitly: primary=True to make "
                f"'{label}' the dimension that solve()'s equity metrics and "
                "weights={'equity': ...} are computed against, or "
                "primary=False to register it as an additional dimension for "
                "analysis only. (If you meant to replace "
                f"'{self._primary_equity_label}' rather than add to it, "
                "re-register it under the same label with replace=True.)"
            )
        return primary

    def _resolve_equity(self, equity=None):
        """
        Return one registered equity dimension's metadata dict.

        `equity=None` gives the primary dimension -- the right default for
        every caller that doesn't offer a choice. A label selects that
        dimension instead.

        Returns `None` when nothing is registered at all (so callers can
        degrade gracefully, as several plots do), but raises when a label
        is given that isn't registered: asking for a specific dimension and
        silently getting the primary one back would misreport which bands a
        chart or metric describes.
        """
        if equity is None:
            return self.primary_equity

        if equity not in self.equity_datasets:
            registered = sorted(self.equity_datasets)
            raise ValueError(
                f"Unknown equity dimension '{equity}'. "
                + (
                    f"Registered: {registered}."
                    if registered
                    else "No equity data has been registered -- call "
                    "add_equity_data() first."
                )
            )
        return self.equity_datasets[equity]

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
        disadvantaged_end: Literal["low", "high"] | None = None,
        continuous_measure: bool = False,
        n_bins: int = 10,
        reverse: bool = False,
        verbose: bool = True,
        direction: Literal["higher_is_better", "higher_is_worse"] | None = None,
        primary: bool | None = None,
        replace: bool = False,
    ):
        """
        Add a dataframe containing equity data into your problem.

        This method associates demand points with an equity metric (such as
        the Index of Multiple Deprivation). If a continuous measure is provided,
        it is automatically discretized into deciles (or maximum possible quantiles)
        to facilitate categorical plotting and comparative equity analysis.

        Call it more than once to register several equity dimensions --
        e.g. deprivation, rurality and ethnicity -- each under its own
        `label`. Every registered dimension is merged into each solution's
        `problem_df`, so any of them can be selected for analysis by label.
        Exactly one is the **primary** dimension: the one `solve()` computes
        its `solution_df` equity columns (`inter_tertile_ratio`,
        `weighted_by_equity_group`, ...) against, and the one
        `weights={"equity": ...}` weights by. See `primary` below, and
        `show_equity_dimensions()` for what is currently registered.

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
        disadvantaged_end : {"low", "high"}, default "low"
            Which end of the `equity_col` scale represents the most
            disadvantaged (e.g. most deprived) regions. This is stored as
            metadata and applied at analysis time — it does not modify the
            stored data. Equity-aware weighting and analysis prioritise
            whichever end you name here.

            - ``"low"`` : the lowest values are the most disadvantaged
            (e.g. IMD decile 1 = most deprived under the standard DLUHC
            1–10 scale).
            - ``"high"`` : the highest values are the most disadvantaged
            (e.g. raw IMD score, where a higher score means more deprived;
            or a custom scale where 1 = least deprived).

            .. note::
                IMD *deciles* as published by DLUHC run 1 (most deprived) to 10
                (least deprived), so for pre-binned IMD decile columns use
                ``disadvantaged_end="low"``. For raw IMD *scores* (higher =
                more deprived) use ``disadvantaged_end="high"``.
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
        primary : bool, optional
            Whether this dimension becomes the primary one (see above).

            Required -- and rejected as ambiguous if omitted -- only when
            registering a *new* label on a problem that already has an
            equity dimension. That case used to overwrite the existing
            dimension, so silently registering alongside it and leaving the
            first as primary would keep old code running while quietly
            changing which bands its numbers describe.

            Defaults are unambiguous everywhere else: the first dimension
            registered is always primary (passing `primary=False` for it
            raises), and re-registering an existing label with
            `replace=True` keeps that label's current primary status unless
            you say otherwise.
        replace : bool, default False
            Overwrite the dimension already registered under `label`.
            Without it, re-using a label raises rather than silently
            replacing the data behind it. Passing it for a label that isn't
            registered also raises, rather than quietly registering a new
            dimension.
        direction : {"higher_is_better", "higher_is_worse"}, optional
            .. deprecated::
                Use ``disadvantaged_end`` instead. ``"higher_is_better"``
                (higher values = more advantaged, e.g. DLUHC IMD deciles)
                maps to ``disadvantaged_end="low"``; ``"higher_is_worse"``
                (higher values = more disadvantaged, e.g. raw IMD scores)
                maps to ``disadvantaged_end="high"``. Passing both arguments
                raises a ``ValueError``.

        Raises
        ------
        ValueError
            If `continuous_measure` is True but the data cannot be meaningfully
            binned due to too many identical values; if `label` is already
            registered and `replace` is not True; if `replace=True` names a
            label that is not registered; if a new label is registered
            alongside an existing dimension without an explicit `primary`;
            if `primary=False` is given for the only registered dimension;
            or if `equity_col` is already used by another registered
            dimension (they share one per-region frame, so one would
            overwrite the other).

        Notes
        -----
        When `continuous_measure` is True, `pandas.qcut` is used with
        `duplicates='drop'`. If the data is highly skewed with duplicate values,
        this may result in fewer than 10 bins. The method handles this dynamically
        to ensure the resulting categories always start at 1.
        """
        # Resolved before any data is parsed so an ambiguous registration
        # fails immediately, rather than after binning a large frame.
        is_primary = self._resolve_primary_flag(label, primary, replace)

        if direction is not None:
            if disadvantaged_end is not None:
                raise ValueError(
                    "Please provide only 'disadvantaged_end' -- 'direction' is "
                    "its deprecated alias, and passing both is ambiguous."
                )
            if direction not in ("higher_is_better", "higher_is_worse"):
                raise ValueError(
                    "direction must be one of 'higher_is_better' or "
                    f"'higher_is_worse'. You provided '{direction}'."
                )
            warn(
                "The 'direction' argument to add_equity_data() is deprecated. "
                "Use disadvantaged_end='low' instead of "
                "direction='higher_is_better' (lowest values = most deprived, "
                "e.g. DLUHC IMD deciles), and disadvantaged_end='high' instead "
                "of direction='higher_is_worse' (highest values = most "
                "deprived, e.g. raw IMD scores).",
                FutureWarning,
                stacklevel=2,
            )
            disadvantaged_end = "low" if direction == "higher_is_better" else "high"
        elif disadvantaged_end is None:
            # DLUHC IMD deciles (1 = most deprived) are the most common input.
            disadvantaged_end = "low"

        if disadvantaged_end not in ("low", "high"):
            raise ValueError(
                "disadvantaged_end must be one of 'low' or 'high'. "
                f"You provided '{disadvantaged_end}'."
            )

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

        # Two dimensions sharing an equity_col would collide when both are
        # merged into the same problem_df, and the second would silently
        # overwrite the first's bands there. Checked against the resolved
        # column name (post-binning), and only against the datasets this
        # call is not itself replacing.
        for other_label, other in self.equity_datasets.items():
            if other_label != label and other["equity_col"] == equity_col:
                raise ValueError(
                    f"Equity dimension '{other_label}' already uses the column "
                    f"'{equity_col}'. Two dimensions cannot share a column "
                    "name -- they are merged into the same per-region frame, "
                    "where one would overwrite the other. Rename the column in "
                    "one of the two source frames before registering."
                )

        self.equity_datasets[label] = {
            "data": loaded_df[cols_to_include],
            "data_type": "pandas",  # We drop any geometry data here
            "equity_col": equity_col,
            "common_col": common_col,
            "label": label,
            "disadvantaged_end": disadvantaged_end,
        }
        if is_primary:
            self._primary_equity_label = label
        elif self._primary_equity_label == label:
            # Re-registering the current primary as primary=False would
            # otherwise leave _primary_equity_label pointing at a dimension
            # that no longer claims to be primary.
            raise ValueError(
                f"'{label}' is currently the primary equity dimension, so it "
                "cannot be re-registered with primary=False. Promote another "
                "dimension with primary=True first, or re-register this one "
                "with primary=True."
            )

    def show_equity_data(self, label=None):
        """
        The registered equity data for one dimension, or the primary
        dimension's when `label` is omitted. Returns `None` if no equity
        data is registered; raises if `label` names an unregistered
        dimension.
        """
        resolved = self._resolve_equity(label)
        return None if resolved is None else resolved["data"]

    def show_equity_dimensions(self):
        """
        One row per registered equity dimension: its label, the column its
        bands live in, the join column, which end of the scale is the
        disadvantaged one, and whether it is the primary dimension.

        Returns an empty DataFrame with those columns when nothing is
        registered, rather than `None`, so callers can display it either
        way.
        """
        return pd.DataFrame(
            [
                {
                    "label": meta["label"],
                    "equity_col": meta["equity_col"],
                    "common_col": meta["common_col"],
                    "disadvantaged_end": meta["disadvantaged_end"],
                    "primary": name == self._primary_equity_label,
                }
                for name, meta in self.equity_datasets.items()
            ],
            columns=[
                "label",
                "equity_col",
                "common_col",
                "disadvantaged_end",
                "primary",
            ],
        )

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
        demand=None,
        cmap="Blues",
        plot_region_of_interest_only=False,
        edgecolor="black",
        linewidth=0.5,
        add_basemap: bool = True,
        tiles="CartoDB positron",
        show_axis: bool = False,
        title=None,
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
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. When given (with `plot_demand=True`),
            plots that scenario's demand instead of the primary demand
            data. Ignored (and rejected) when `plot_demand=False`.
        cmap: str, default "Blues"
            Colour map to be used for plotting demand. Ignored if plot_demand=False.
        tiles: str, default "CartoDB positron"
            Tiles to be used for background in map. Ignored if interactive = False.
        add_basemap : bool, default True
            If True, adds a background web map (via `contextily` for static
            plots, or the `tiles` basemap for interactive ones). Set False to
            skip the tile download entirely.

            .. versionchanged:: 0.7.0
                Renamed from ``show_basemap``, which has been removed.

        title : str, optional
            Title for a static plot (`ax.set_title()`). Ignored if
            `interactive=True` -- a Folium map has no equivalent built-in
            title support. `None` (the default) leaves the plot untitled,
            matching prior behaviour.

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
            If `plot_demand` is True but `self.demand_data` is None, if
            `demand` is given but `plot_demand` is False, or if `demand`
            does not name a registered secondary demand scenario.
        TypeError
            If the removed `show_basemap` argument is supplied.

        Notes
        -----
        When `plot_demand` is True, the method performs a merge using:
        - `self._region_geometry_layer_common_col` (left)
        - `self._demand_data_id_col` (right)

        Interactive maps default to the "CartoDB positron" tile set and
        the "Blues" colormap for demand visualization.

        The static plot's colour bar is labelled "Demand" (`plot_demand=
        True`) or with the human-readable `add_equity_data(label=...)`,
        falling back to the raw equity column name if no label was
        registered (`plot_equity=True`) -- previously unlabelled either
        way.
        """
        _reject_removed_basemap_alias(kwargs, "plot_region_geometry_layer")

        if self.region_geometry_layer is None:
            raise ValueError(
                "No region geometry layer has been initialised."
                "Please run `.add_region_geometry_layer()` first."
            )
        if demand is not None and not plot_demand:
            raise ValueError(
                "demand=<label> selects which demand dataset to plot -- "
                "set plot_demand=True to enable demand plotting."
            )

        if plot_demand and self.demand_data is None:
            raise ValueError(
                "Cannot plot demand when no demand data is present."
                "Please run `.add_demand()` first or change the `plot_demand` parameter to False."
            )

        if demand is not None and demand not in self.secondary_demand_matrices:
            raise ValueError(
                f"Unknown secondary demand scenario '{demand}'. Registered "
                f"labels: {sorted(self.secondary_demand_matrices)}."
            )

        if plot_demand and plot_equity:
            raise ValueError(
                "Cannot plot both demand and equity. Please set one option to False."
            )

        if plot_demand:
            if demand is None:
                demand_source = self.demand_data
                demand_id_col = self._demand_data_id_col
                demand_value_col = self._demand_data_demand_col
            else:
                demand_meta = self.secondary_demand_matrices[demand]
                demand_source = demand_meta["data"]
                demand_id_col = demand_meta["id_col"]
                demand_value_col = demand_meta["demand_col"]

            plotting_df = self.region_geometry_layer.merge(
                demand_source,
                left_on=self._region_geometry_layer_common_col,
                right_on=demand_id_col,
            )

            if interactive:
                m = plotting_df.explore(
                    column=demand_value_col,  # make choropleth based on demand col
                    tooltip=demand_value_col,  # show demand col value in tooltip (on hover)
                    popup=True,  # show all values in popup (on click)
                    cmap=cmap,  # use "Blues" matplotlib colormap
                    style_kwds=dict(color="black"),
                    tiles=tiles if add_basemap else None,
                    **kwargs,
                )

                return _attach_deferred_fit_bounds(m)
            else:
                ax = plotting_df.plot(
                    column=demand_value_col,
                    legend=True,
                    legend_kwds={"label": "Demand"},
                    cmap=cmap,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )

                if title:
                    ax.set_title(title)

                if add_basemap:
                    try:
                        cx.add_basemap(ax, crs=plotting_df.crs.to_string(), timeout=30)
                    except RequestException as e:
                        warnings.warn(
                            f"Unable to download background map tiles ({type(e).__name__}). "
                            "Continuing without a basemap.",
                            stacklevel=2,
                        )

                if not show_axis:
                    ax.axis("off")

                return ax

        if plot_equity:
            plotting_df = pd.merge(
                self.region_geometry_layer,
                self.equity_data,
                left_on=self._region_geometry_layer_common_col,
                right_on=self._equity_data_common_col,
            )

            if plot_region_of_interest_only:
                if self.demand_data is None:
                    warn("No demand data provided; estimating region of interest.")

                    self._setup_equal_demand_df()

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

                return _attach_deferred_fit_bounds(m)
            else:
                equity_axis_label = self._equity_data_label or self._equity_data_equity_col
                ax = plotting_df.plot(
                    column=self._equity_data_equity_col,
                    legend=True,
                    legend_kwds={"label": equity_axis_label},
                    cmap=cmap,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    **kwargs,
                )

                if title:
                    ax.set_title(title)

                if add_basemap:
                    try:
                        cx.add_basemap(ax, crs=plotting_df.crs.to_string(), timeout=30)
                    except RequestException as e:
                        warnings.warn(
                            f"Unable to download background map tiles ({type(e).__name__}). "
                            "Continuing without a basemap.",
                            stacklevel=2,
                        )

                if not show_axis:
                    ax.axis("off")

                return ax

        if plot_region_of_interest_only:
            if self.demand_data is None:
                warn("No demand data provided; estimating region of interest.")

                self._setup_equal_demand_df()

            plotting_df = self.region_geometry_layer.merge(
                self.demand_data[[self._demand_data_id_col]],
                left_on=self._region_geometry_layer_common_col,
                right_on=self._demand_data_id_col,
                how="inner",
            )
        else:
            plotting_df = self.region_geometry_layer

        if interactive:
            m = plotting_df.explore(
                tiles=tiles, edgecolor=edgecolor, linewidth=linewidth, **kwargs
            )
            return _attach_deferred_fit_bounds(m)
        else:
            ax = plotting_df.plot(**kwargs)

            if title:
                ax.set_title(title)

            if add_basemap:
                try:
                    cx.add_basemap(ax, crs=plotting_df.crs.to_string(), timeout=30)
                except RequestException as e:
                    warnings.warn(
                        f"Unable to download background map tiles ({type(e).__name__}). "
                        "Continuing without a basemap.",
                        stacklevel=2,
                    )

            if not show_axis:
                ax.axis("off")

            return ax

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

    ###############################
    # MARK: Additional Data
    ###############################
    def add_additional_data(
        self,
        df,
        column_of_interest: str,
        common_col: str,
        label: str,
        direction: Literal["higher_better", "lower_better"],
    ):
        """
        Append a user-defined dataset to be tracked during optimization.

        Allows analysts to register secondary spatial or tabular datasets (e.g.,
        additional travel matrices, mileage, CO2, additional demand)
        to evaluate trade-offs alongside the primary optimization metrics.

        Parameters
        ----------
        df : pandas.DataFrame or geopandas.GeoDataFrame
            The input dataset containing the secondary metric attributes.
            This dataframe may contain multiple columns, but only a single column can be
            added at a time, which is specified as column_of_interest.
        column_of_interest: str
            The name of the column containing the required metric.
            This is the only column that will be included.
        common_col : str
            The name of the column in the dataset used to join or map the data
            back to the demand regions or candidate facility sites.
        label : str
            A unique human-readable identifier for the dataset (e.g., 'distance_matrix'),
            which will serve as the column header in the final wide results matrix.
        direction : {'higher_better', 'lower_better'}
            The optimization objective direction. Determines whether the library
            should maximize ('higher_better') or minimize ('lower_better') this
            metric during downstream trade-off calculations or Pareto sorting.

        Returns
        -------
        None

        """

        if direction not in ["higher_better", "lower_better"]:
            raise ValueError(
                f"direction must be one of 'higher_better' or 'lower_better'. You provided '{direction}'."
            )

        # if common_col not in [
        #     self._region_geometry_layer_common_col,
        #     self._candidate_sites_candidate_id_col,
        # ]:
        #     warn(
        #         "Provided common col not found in the region geometry or site dataset."
        #         "You can ignore this warning if you haven't provided those datasets yet."
        #     )

        loaded_df, df_type = _load_spatial_or_tabular_data(df)

        loaded_df = loaded_df.rename(columns={column_of_interest: label})

        # Filter to only the joining col and a single column of interest
        loaded_df = loaded_df[[common_col, label]]

        # If this is the first bit of additional data being added, convert from None to a list.
        # Easier for general checking elsewhere if additional_data attribute is initialised to
        # None as more in keeping with defaults for other attributes so less chance of accidentally
        # checking for Noneness and getting False when it's actually an empty list
        if self.additional_data is None:
            self.additional_data = []
        if self._additional_data_labels is None:
            self._additional_data_labels = []

        self.additional_data.append(
            {
                "label": label,
                "data": loaded_df,
                "join_col": common_col,
                "direction": direction,
                "data_type": df_type,
            }
        )
        self._additional_data_labels.append(label)

    def plot_additional_data_map(label):
        pass
        # TODO - general method that will join to region geometry and plot a map
        # Support one at a time rather than trying to plot all

    def plot_additional_data_bar(label):
        pass
        # TODO - general method to show spread of additional data
        # Support one at a time rather than trying to plot all
