import esda
from pysal.lib import weights
import warnings
from typing import Literal
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import contextily as cx
import folium
import matplotlib.pyplot as plt
import geopandas

NeighbourhoodMethod = Literal[
    "rook",
    "queen",
    "k-nearest",
]

ClusterDisplay = Literal[
    "all",
    "hotspots",
    "coldspots",
    "outliers",
]

SupportedProblemInputs = Literal["demand", "equity", "demand_equity_combined"]
SupportedSolutionInputs = Literal["demand", "equity"]

SupportedInputs = SupportedProblemInputs | SupportedSolutionInputs

SupportedCombinationMethods = Literal["multiply", "sum", "rank"]


class SiteProblemEDAMixin:
    @property
    def _prob_ctx(self):
        """
        Dynamically routes requests to the object holding the problem data.
        Returns 'self.site_problem' if attached to a SiteSolutionSet,
        otherwise returns 'self' if attached directly to a Problem.
        """
        return getattr(self, "site_problem", self)

    def _get_weights(
        self,
        df,
        neighbourhood_method: NeighbourhoodMethod = "rook",
        k: int | None = None,
        verbose: bool = True,
        return_weights: bool = True,
        force_recalculation: bool = True,
    ):
        """
        Create and cache a spatial weights matrix.

        Generates a PySAL spatial weights object based on the region
        geometry layer attached to the problem. The resulting weights
        matrix is row-standardised and cached for reuse by exploratory
        spatial data analysis methods.

        Parameters
        ----------
        neighbourhood_method : {"rook", "queen", "k-nearest"}, default="rook"
            Method used to define neighbouring regions.

            * ``"rook"``: Regions sharing an edge are considered neighbours.
            * ``"queen"``: Regions sharing an edge or vertex are considered
            neighbours.
            * ``"k-nearest"``: Each region is connected to its ``k`` nearest
            neighbours.

        k : int, optional
            Number of neighbours to use when ``neighbourhood_method="k-nearest"``.
            Ignored for rook and queen contiguity.

        Raises
        ------
        ValueError
            If ``neighbourhood_method="k-nearest"`` and ``k`` is not provided.

        Notes
        -----
        The resulting weights matrix is row-standardised using
        ``w.transform = "R"``.

        The weights matrix and metadata describing how it was generated
        are stored on the instance as:

        * ``self.spatial_weights``
        * ``self.spatial_weights_method``
        * ``self.spatial_weights_k``
        """
        valid_neighbourhood_method = ["rook", "queen", "k-nearest"]

        if neighbourhood_method not in valid_neighbourhood_method:
            raise ValueError(
                f"neighbourhood_method must be one of {valid_neighbourhood_method}, got {neighbourhood_method!r}"
            )
        # Get the unified context object, meaning we handle this appropriately
        # regardless of whether this is a Problem or SiteSolutionSet
        ctx = self._prob_ctx

        # We look at the hash for the entire region geometry as we're just trying to capture
        # if the region geometry has changed, not whether it's the same region requested as
        # before
        current_hash = ctx._get_geometry_hash(ctx.region_geometry_layer)

        if (
            ctx.spatial_weights is None
            or neighbourhood_method != ctx.spatial_weights_method
            or k != ctx.spatial_weights_k
            or current_hash != ctx._region_geometry_hash
            or force_recalculation
        ):
            if neighbourhood_method == "queen":
                w = weights.Queen.from_dataframe(df)
            elif neighbourhood_method == "rook":
                w = weights.Rook.from_dataframe(df)
            elif neighbourhood_method == "k-nearest":
                if k is None:
                    raise ValueError(
                        "Please specify a value of k in get_hotspots if you want to use k-nearest neighbours"
                    )
                w = weights.KNN.from_dataframe(df, k=k)

            w.transform = "R"

            ctx.spatial_weights = w
            ctx.spatial_weights_method = neighbourhood_method
            ctx.spatial_weights_k = k
            ctx._region_geometry_hash = current_hash
        else:
            if verbose:
                warnings.warn(
                    "Using cached spatial weights.",
                    stacklevel=2,
                )

        if return_weights:
            return ctx.spatial_weights


class SiteProblemHotspotCalculationMixin:
    @property
    def _prob_ctx(self):
        return getattr(self, "site_problem", self)

    def _prepare_analysis_dataframe(
        self, what: str, combination_method: str
    ) -> tuple[pd.DataFrame, str, str]:
        """
        Base data-prep method. Handles standard Problem-level fields.
        Returns: (working_dataframe, column_to_analyze, id_merge_column)
        """
        ctx = self._prob_ctx

        if what == "demand":
            df = ctx.demand_data.copy()
            return df, ctx._demand_data_demand_col, ctx._demand_data_id_col

        elif what == "equity":
            df = ctx.equity_data.copy()
            df_col = ctx._equity_data_equity_col
            df_merge_col = ctx._equity_data_common_col

            # Filter to active demand IDs
            df = df[
                df[df_merge_col].isin(ctx.demand_data[ctx._demand_data_id_col].unique())
            ]

            if ctx._equity_data_direction == "higher_is_worse":
                analysis_col = f"_{df_col}_analysis"
                df[analysis_col] = -df[df_col]
                df_col = analysis_col
            return df, df_col, df_merge_col

        elif what == "demand_equity_combined":
            if ctx.equity_data is None:
                raise ValueError("No equity data loaded. Call add_equity_data() first.")

            # --- Align & Merge (using ctx attributes) ---
            demand_df = ctx.demand_data.copy()
            equity_df = ctx.equity_data.copy()
            shared_ids = demand_df[ctx._demand_data_id_col].unique()
            equity_df = equity_df[
                equity_df[ctx._equity_data_common_col].isin(shared_ids)
            ]

            equity_col = ctx._equity_data_equity_col
            if ctx._equity_data_direction == "higher_is_worse":
                equity_df["_equity_directed"] = -equity_df[equity_col]
            else:
                equity_df["_equity_directed"] = equity_df[equity_col]

            df = demand_df.merge(
                equity_df[
                    [ctx._equity_data_common_col, "_equity_directed", equity_col]
                ],
                left_on=ctx._demand_data_id_col,
                right_on=ctx._equity_data_common_col,
                how="inner",
            )

            # --- MinMax Normalization ---
            def _minmax(series):
                mn, mx = series.min(), series.max()
                return (
                    pd.Series(0.5, index=series.index)
                    if mx == mn
                    else (series - mn) / (mx - mn)
                )

            df["_demand_norm"] = _minmax(df[ctx._demand_data_demand_col])
            df["_equity_norm"] = _minmax(df["_equity_directed"])

            # --- Combined Scoring ---
            if combination_method == "multiply":
                df["combined_score"] = df["_demand_norm"] * df["_equity_norm"]
            elif combination_method == "sum":
                df["combined_score"] = (df["_demand_norm"] + df["_equity_norm"]) / 2
            elif combination_method == "rank":
                df["combined_score"] = (
                    df["_demand_norm"].rank() + df["_equity_norm"].rank()
                ) / (2 * len(df))
            else:
                raise ValueError(f"Invalid combination_method: {combination_method}")

            # --- Typology Calculation ---
            demand_median = df["_demand_norm"].median()
            equity_median = df["_equity_norm"].median()
            high_demand = df["_demand_norm"] >= demand_median
            high_deprivation = df["_equity_norm"] >= equity_median

            df["attribute_typology"] = "Low Demand / Low Deprivation"
            df.loc[high_demand & high_deprivation, "attribute_typology"] = (
                "High Demand / High Deprivation"
            )
            df.loc[high_demand & ~high_deprivation, "attribute_typology"] = (
                "High Demand / Low Deprivation"
            )
            df.loc[~high_demand & high_deprivation, "attribute_typology"] = (
                "Low Demand / High Deprivation"
            )

            return df, "combined_score", ctx._demand_data_id_col

        raise ValueError(f"Unsupported analysis input type: '{what}'")

    def get_hotspots(
        self,
        what: str = "demand",
        neighbourhood_method: str = "rook",
        combination_method: str = "multiply",
        k: int | None = None,
        verbose: bool = True,
        significance_threshold: float = 0.05,
        force_weight_recalculation: bool = False,
    ):
        """
        Identify statistically significant hotspots and coldspots.

        Returns hotspot, coldspot, outlier and non-significant classifications
        derived from Local Moran's I.

        Performs local spatial autocorrelation analysis on a selected
        variable using the specified spatial weights matrix. Areas with
        significantly high values surrounded by high values are classified
        as hotspots, while areas with significantly low values surrounded
        by low values are classified as coldspots.

        Parameters
        ----------
        what : str, default="demand"
            Variable to analyse. This may be the name of a stored dataset
            or metric associated with the problem, such as demand,
            accessibility, or equity measures.

        neighbourhood_method : {"rook", "queen", "k-nearest"}, default="rook"
            Method used to define neighbouring regions.

        k : int, optional
            Number of neighbours to use when ``neighbourhood_method="k-nearest"``.
            Required if ``neighbourhood_method="k-nearest"``.

        Returns
        -------
        geopandas.GeoDataFrame
            Region geometry layer with additional columns describing
            hotspot/coldspot classification and associated statistics.

        Notes
        -----
        Spatial weights are generated when required and cached for
        reuse. Cached weights are automatically invalidated when the
        region geometry or weighting parameters change.
        """
        # Build the weights if they've not been generated before or if anything about
        # the weight parameters or region geometry layer have changed

        # Demand is assumed to only be passed for the region of interest.
        # In effect, it's expected to define the region of interest!
        # And this holds if you are generating a problem simply from a travel matrix without
        # passing in demand (although then you're not going to be looking for demand hotspots,
        # though you could be interested in equity hotspots)
        # However, there's a good chance you'll have passed an unfiltered whole-country
        # equity dataset, so this needs to filter down too.

        ctx = self._prob_ctx

        # 1. Delegate data prep out to the extensible method
        df, df_col, df_merge_col = self._prepare_analysis_dataframe(
            what, combination_method
        )
        result = df.copy()

        # 2. Geometry Filtering
        filtered_region_geometry = ctx.region_geometry_layer.copy()
        filtered_region_geometry = filtered_region_geometry[
            filtered_region_geometry[ctx._region_geometry_layer_common_col].isin(
                ctx.demand_data[ctx._demand_data_id_col].unique()
            )
        ]

        # 3. Weights Resolution
        w = self._get_weights(
            df=filtered_region_geometry,
            neighbourhood_method=neighbourhood_method,
            k=k,
            verbose=verbose,
            return_weights=True,
            force_recalculation=force_weight_recalculation,
        )

        # 4. Local Moran's I Core Math Engine
        lisa = esda.moran.Moran_Local(result[df_col], w)

        result["local_moran_i"] = lisa.Is
        result["p_value"] = lisa.p_sim
        result["quadrant"] = lisa.q
        result["cluster_type"] = "Not Significant"

        significant = lisa.p_sim < significance_threshold
        result.loc[significant & (lisa.q == 1), "cluster_type"] = "Hotspot"
        result.loc[significant & (lisa.q == 3), "cluster_type"] = "Coldspot"
        result.loc[significant & (lisa.q == 2), "cluster_type"] = "Low-High Outlier"
        result.loc[significant & (lisa.q == 4), "cluster_type"] = "High-Low Outlier"

        # Cleanup working columns
        result = result.drop(
            columns=["_demand_norm", "_equity_norm", "_equity_directed"],
            errors="ignore",
        )

        return ctx.region_geometry_layer.merge(
            result,
            left_on=ctx._region_geometry_layer_common_col,
            right_on=df_merge_col,
        )


class SiteSolutionHotspotCalculationMixin(SiteProblemHotspotCalculationMixin):
    """Overrides data preparation to inject solution-specific travel metrics."""

    def _prepare_analysis_dataframe(
        self, what: str, combination_method: str
    ) -> tuple[pd.DataFrame, str, str]:
        # intercept solution-specific metric calculations
        if what == "travel_time":
            # 1. Grab base context data
            ctx = self._prob_ctx
            df = ctx.demand_data.copy()

            # 2. Compute your travel multiplier/metrics unique to this Solution Set
            # (e.g., pulling allocation maps, routing arrays, etc. from 'self')
            df["travel_metric"] = (
                self.calculate_travel_times() * df[ctx._demand_data_demand_col]
            )

            # 3. Return the modified frame ready for the spatial math pipeline
            return df, "travel_metric", ctx._demand_data_id_col

        # Fallback to default problem parameters ("demand", "equity", "demand_equity_combined")
        return super()._prepare_analysis_dataframe(what, combination_method)


class HotspotPlotMixin:
    def plot_hotspots(
        self,
        hotspots_df: pd.DataFrame | None = None,
        ax: plt.Axes | None = None,
        interactive: bool = False,
        show_hotspots: bool = True,
        show_coldspots: bool = True,
        show_low_high_outliers: bool = True,
        show_high_low_outliers: bool = True,
        show_non_significance: bool = True,
        hotspot_colour: str = "#d7191c",  # red
        coldspot_colour: str = "#2c7bb6",  # dark blue
        low_high_outlier_colour: str = "#abd9e9",  # light blue
        high_low_outlier_colour: str = "#fee08b",  # yellow
        not_significant_colour: str = "#bdbdbd",  # grey
        tiles: str = "CartoDB positron",
        edgecolor: str = "black",
        linewidth: float = 0.5,
        show_basemap: bool = True,
        show_axis: bool = False,
        opacity: float = 0.7,
        what: SupportedInputs = "demand",
        combination_method: SupportedCombinationMethods = "multiply",
        neighbourhood_method: NeighbourhoodMethod = "rook",
        k: int | None = None,
        verbose: bool = True,
        significance_threshold: float = 0.05,
        force_weight_recalculation: bool = False,
        **kwargs,
    ):
        """
        Plot statistically significant hotspots and coldspots.

        Creates either a static GeoPandas choropleth or an interactive
        Folium map showing the results of Local Moran's I analysis.
        Areas are coloured according to hotspot classification.

        Parameters
        ----------
        hotspots_df : geopandas.GeoDataFrame, optional
            Hotspot analysis results returned by :meth:`get_hotspots`.
            If ``None``, hotspot analysis is performed automatically
            using the supplied parameters.

        interactive : bool, default=False
            If ``True``, return an interactive Folium map. Otherwise,
            return a static GeoPandas plot.

        hotspot_color : str, default="#d7191c"
            Colour used for statistically significant hotspot
            (high-high) regions.

        coldspot_color : str, default="#2c7bb6"
            Colour used for statistically significant coldspot
            (low-low) regions.

        low_high_outlier_color : str, default="#abd9e9"
            Colour used for low-high spatial outliers.

        high_low_outlier_color : str, default="#fee08b"
            Colour used for high-low spatial outliers.

        not_significant_color : str, default="#bdbdbd"
            Colour used for regions that are not statistically
            significant.

        tiles : str, default="CartoDB positron"
            Tile provider used for interactive maps.

        edgecolor : str, default="black"
            Boundary colour used when plotting polygons.

        linewidth : float, default=0.5
            Width of polygon boundaries.

        what : {"demand", "equity"}, default="demand"
            Variable used for hotspot analysis when
            ``hotspots_df`` is not provided.

        neighbourhood_method : {"rook", "queen", "k-nearest"}, default="rook"
            Method used to define neighbouring regions.

        k : int, optional
            Number of neighbours when
            ``neighbourhood_method="k-nearest"``.

        verbose : bool, default=True
            Whether to print progress information.

        significance_threshold : float, default=0.05
            Significance threshold used to classify hotspot and
            coldspot regions.

        force_weight_recalculation : bool, default=False
            If ``True``, spatial weights are recalculated even if a
            cached version is available.

        **kwargs
            Additional keyword arguments passed to
            ``GeoDataFrame.plot`` or ``GeoDataFrame.explore``.

        Returns
        -------
        matplotlib.axes.Axes or folium.Map
            Static or interactive hotspot map depending on the value
            of ``interactive``.

        Notes
        -----
        Hotspot classifications are derived from Local Moran's I and
        consist of:

        * Hotspot (high-high)
        * Coldspot (low-low)
        * Low-high spatial outlier
        * High-low spatial outlier
        * Not significant

        The default colour scheme mirrors the conventions commonly
        used in GeoDa and PySAL visualisations.
        """

        if hotspots_df is None:
            hotspots_df = self.get_hotspots(
                what=what,
                neighbourhood_method=neighbourhood_method,
                combination_method=combination_method,
                k=k,
                verbose=verbose,
                significance_threshold=significance_threshold,
                force_weight_recalculation=force_weight_recalculation,
            )

        is_combined = "attribute_typology" in hotspots_df.columns

        # Determine which tooltip fields are available — attribute_typology
        # is only present for combined runs
        _base_tooltip_fields = ["cluster_type", "local_moran_i", "p_value", "quadrant"]
        _base_tooltip_aliases = ["Cluster", "Local Moran I", "p-value", "Quadrant"]

        if "attribute_typology" in (
            hotspots_df.columns if hotspots_df is not None else []
        ):
            _tooltip_fields = ["attribute_typology"] + _base_tooltip_fields
            _tooltip_aliases = ["Attribute Typology"] + _base_tooltip_aliases
        else:
            _tooltip_fields = _base_tooltip_fields
            _tooltip_aliases = _base_tooltip_aliases

        def add_cluster_layer(
            df, cluster_name, group_name, colour, visible, opacity, edgecolor
        ):
            subset = df[df["cluster_type"] == cluster_name]

            if subset.empty:
                return

            fg = folium.FeatureGroup(name=group_name, show=visible)

            folium.GeoJson(
                subset,
                style_function=lambda feature, c=colour: {
                    "fillColor": c,
                    "color": edgecolor,
                    "weight": linewidth,
                    "fillOpacity": opacity,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=_tooltip_fields,
                    aliases=_tooltip_aliases,
                ),
            ).add_to(fg)

            fg.add_to(m)

        cluster_groups = {
            "Hotspot": "Hotspots",
            "Coldspot": "Coldspots",
            "Low-High Outlier": "Low-High Outliers",
            "High-Low Outlier": "High-Low Outliers",
            "Not Significant": "Not Significant",
        }

        visible = {
            "Hotspot": show_hotspots,
            "Coldspot": show_coldspots,
            "Low-High Outlier": show_low_high_outliers,
            "High-Low Outlier": show_high_low_outliers,
            "Not Significant": show_non_significance,
        }

        colours = {
            "Hotspot": hotspot_colour,
            "Coldspot": coldspot_colour,
            "Low-High Outlier": low_high_outlier_colour,
            "High-Low Outlier": high_low_outlier_colour,
            "Not Significant": not_significant_colour,
        }

        hotspots_df["_alpha"] = hotspots_df["cluster_type"].map(
            lambda c: opacity if visible.get(c, False) else 0.0
        )

        if interactive:
            hotspots_df["_plot_colour"] = hotspots_df.apply(
                lambda r: (
                    colours[r["cluster_type"]]
                    if visible.get(r["cluster_type"], False)
                    else "#e6e6e6"  # light grey for hidden categories
                ),
                axis=1,
            )

            # Get bounds from the GeoDataFrame to centre the map correctly
            bounds = hotspots_df.to_crs(
                epsg=4326
            ).total_bounds  # [minx, miny, maxx, maxy]
            centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

            m = folium.Map(
                location=centre,
                tiles=tiles if show_basemap else None,
            )

            for cluster_name, group_name in cluster_groups.items():
                add_cluster_layer(
                    hotspots_df,
                    cluster_name,
                    group_name,
                    colour=colours[cluster_name],
                    visible=visible[cluster_name],
                    opacity=opacity,
                    edgecolor=edgecolor,
                )

            folium.LayerControl(collapsed=False).add_to(m)

            # Fit the map to the data extent
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            return m
        else:
            hotspots_df["_plot_colour"] = hotspots_df["cluster_type"].map(colours)

            hotspots_df["_plot_rgba"] = hotspots_df.apply(
                lambda r: mcolors.to_rgba(
                    r["_plot_colour"],
                    r["_alpha"],
                ),
                axis=1,
            )

            if ax is None:
                _, ax = plt.subplots()

            ax = hotspots_df.plot(
                ax=ax,
                legend=True,
                color=hotspots_df["_plot_rgba"],
                edgecolor=edgecolor,
                linewidth=linewidth,
                **kwargs,
            )

            handles = [
                Patch(facecolor=colour, label=label)
                for label, colour in colours.items()
            ]

            ax.legend(
                handles=handles,
                title="Cluster Type",
                loc="best",
            )

            if is_combined:
                ax.set_title(
                    f"Combined Demand / Deprivation Hotspots ({combination_method})",
                    fontsize=10,
                )

            if show_basemap:
                cx.add_basemap(
                    ax,
                    crs=hotspots_df.crs.to_string(),
                )

            if not show_axis:
                ax.axis("off")

            return ax

    def plot_demand_deprivation_quadrants(
        self,
        hotspots_df: geopandas.GeoDataFrame | None = None,
        ax: plt.Axes | None = None,
        interactive: bool = False,
        combination_method: Literal["multiply", "sum", "rank"] = "multiply",
        neighbourhood_method: NeighbourhoodMethod = "rook",
        k: int | None = None,
        verbose: bool = True,
        significance_threshold: float = 0.05,
        force_weight_recalculation: bool = False,
        # Typology colours — 2x2 quadrant palette
        high_high_colour: str = "#d7191c",  # red   — high demand, high deprivation
        high_low_colour: str = "#fee08b",  # amber — high demand, low deprivation
        low_high_colour: str = "#abd9e9",  # light blue — low demand, high deprivation
        low_low_colour: str = "#bdbdbd",  # grey  — low demand, low deprivation
        # Significance overlay
        significant_edgecolor: str = "#1a1a1a",
        significant_linewidth: float = 1.8,
        non_significant_edgecolor: str = "#bbbbbb",
        non_significant_linewidth: float = 0.4,
        tiles: str = "CartoDB positron",
        show_basemap: bool = True,
        show_axis: bool = False,
        opacity: float = 0.7,
        **kwargs,
    ):
        """
        Plot the 2×2 attribute typology for a combined demand/deprivation analysis.

        Each area is coloured according to whether it is high or low on demand
        and deprivation independently (the ``attribute_typology`` column produced
        by :meth:`get_hotspots` with ``what="demand_equity_combined"``). Areas that are also
        statistically significant spatial clusters (Hotspot or Coldspot) are
        additionally highlighted with a bolder border.

        This plot complements :meth:`plot_hotspots`: where that method shows
        *spatial clustering*, this one shows the underlying *attribute structure*
        — i.e. why an area is (or is not) a hotspot.

        Parameters
        ----------
        hotspots_df : geopandas.GeoDataFrame, optional
            Pre-computed results from :meth:`get_hotspots` with
            ``what="demand_equity_combined"``. If ``None``, the analysis is run automatically
            using the supplied parameters.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. A new figure and axes are created if not provided.
            Ignored when ``interactive=True``.
        interactive : bool, default False
            If ``True``, returns an interactive Folium map. Otherwise returns
            a static matplotlib axes.
        combination_method : {"multiply", "sum", "rank"}, default "multiply"
            Score combination method passed to :meth:`get_hotspots` when
            ``hotspots_df`` is ``None``.
        neighbourhood_method : {"rook", "queen", "k-nearest"}, default "rook"
            Spatial weights method passed to :meth:`get_hotspots` when
            ``hotspots_df`` is ``None``.
        k : int, optional
            Number of neighbours for ``neighbourhood_method="k-nearest"``.
        verbose : bool, default True
            Whether to print progress when recomputing hotspots.
        significance_threshold : float, default 0.05
            P-value threshold used to determine which areas receive a bold
            significance border.
        force_weight_recalculation : bool, default False
            If ``True``, spatial weights are recalculated even if cached.
        high_high_colour : str, default "#d7191c"
            Colour for High Demand / High Deprivation areas.
        high_low_colour : str, default "#fee08b"
            Colour for High Demand / Low Deprivation areas.
        low_high_colour : str, default "#abd9e9"
            Colour for Low Demand / High Deprivation areas.
        low_low_colour : str, default "#bdbdbd"
            Colour for Low Demand / Low Deprivation areas.
        significant_edgecolor : str, default "#1a1a1a"
            Border colour for statistically significant spatial clusters.
        significant_linewidth : float, default 1.8
            Border width for statistically significant spatial clusters.
        non_significant_edgecolor : str, default "#bbbbbb"
            Border colour for non-significant areas.
        non_significant_linewidth : float, default 0.4
            Border width for non-significant areas.
        tiles : str, default "CartoDB positron"
            Tile provider for interactive maps.
        show_basemap : bool, default True
            Whether to add a basemap (static) or tile layer (interactive).
        show_axis : bool, default False
            Whether to show axis ticks and labels on static plots.
        opacity : float, default 0.7
            Fill opacity for all areas.
        **kwargs
            Additional keyword arguments passed to ``GeoDataFrame.plot``
            or ``GeoDataFrame.explore``.

        Returns
        -------
        matplotlib.axes.Axes or folium.Map

        Raises
        ------
        ValueError
            If ``hotspots_df`` is provided but does not contain an
            ``attribute_typology`` column, suggesting it was not produced
            by a combined analysis.
        """
        if hotspots_df is None:
            hotspots_df = self.get_hotspots(
                what="demand_equity_combined",
                combination_method=combination_method,
                neighbourhood_method=neighbourhood_method,
                k=k,
                verbose=verbose,
                significance_threshold=significance_threshold,
                force_weight_recalculation=force_weight_recalculation,
            )

        if "attribute_typology" not in hotspots_df.columns:
            raise ValueError(
                "hotspots_df does not contain an 'attribute_typology' column. "
                "Ensure it was produced by get_hotspots(what='demand_equity_combined')."
            )

        def get_typology_colour(label: str) -> str:
            left, right = label.split(" / ")

            left_high = left.startswith("High")
            right_high = right.startswith("High")

            if left_high and right_high:
                return high_high_colour
            elif left_high and not right_high:
                return high_low_colour
            elif not left_high and right_high:
                return low_high_colour
            else:
                return low_low_colour

        hotspots_df["_plot_colour"] = hotspots_df["attribute_typology"].map(
            get_typology_colour
        )

        typology_colours = {
            label: get_typology_colour(label)
            for label in hotspots_df["attribute_typology"].unique()
        }

        # Significance overlay: bold border for spatial clusters, quiet for the rest
        hotspots_df["_is_significant"] = hotspots_df["cluster_type"].isin(
            ["Hotspot", "Coldspot"]
        )
        hotspots_df["_edge_colour"] = hotspots_df["_is_significant"].map(
            {True: significant_edgecolor, False: non_significant_edgecolor}
        )
        hotspots_df["_edge_width"] = hotspots_df["_is_significant"].map(
            {True: significant_linewidth, False: non_significant_linewidth}
        )
        hotspots_df["_fill_colour"] = hotspots_df["attribute_typology"].map(
            typology_colours
        )
        hotspots_df["_fill_rgba"] = hotspots_df["_fill_colour"].apply(
            lambda c: mcolors.to_rgba(c, opacity)
        )

        if interactive:
            bounds = hotspots_df.to_crs(epsg=4326).total_bounds
            centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

            m = folium.Map(
                location=centre,
                tiles=tiles if show_basemap else None,
            )

            for typology_label, colour in typology_colours.items():
                subset = hotspots_df[
                    hotspots_df["attribute_typology"] == typology_label
                ]
                if subset.empty:
                    continue

                fg = folium.FeatureGroup(name=typology_label, show=True)

                for _, row in subset.iterrows():
                    edge_w = (
                        significant_linewidth
                        if row["_is_significant"]
                        else non_significant_linewidth
                    )
                    edge_c = (
                        significant_edgecolor
                        if row["_is_significant"]
                        else non_significant_edgecolor
                    )
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda feature, c=colour, ew=edge_w, ec=edge_c: {
                            "fillColor": c,
                            "color": ec,
                            "weight": ew,
                            "fillOpacity": opacity,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=[
                                "attribute_typology",
                                "cluster_type",
                                "combined_score",
                                "p_value",
                            ],
                            aliases=[
                                "Attribute Typology",
                                "Spatial Cluster",
                                "Combined Score",
                                "p-value",
                            ],
                        ),
                    ).add_to(fg)

                fg.add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            return m

        else:
            if ax is None:
                _, ax = plt.subplots()

            # Draw non-significant areas first, significant on top so bold
            # borders aren't occluded by neighbouring polygons
            for is_significant in [False, True]:
                subset = hotspots_df[hotspots_df["_is_significant"] == is_significant]
                if subset.empty:
                    continue

                subset.plot(
                    ax=ax,
                    color=subset["_fill_rgba"].tolist(),
                    edgecolor=subset["_edge_colour"].tolist(),
                    linewidth=(
                        significant_linewidth
                        if is_significant
                        else non_significant_linewidth
                    ),
                    **kwargs,
                )

            # Typology legend
            typology_handles = [
                Patch(facecolor=colour, edgecolor="#555555", linewidth=0.5, label=label)
                for label, colour in typology_colours.items()
            ]
            # Significance legend
            significance_handles = [
                Patch(
                    facecolor="white",
                    edgecolor=significant_edgecolor,
                    linewidth=significant_linewidth,
                    label="Significant spatial cluster",
                ),
                Patch(
                    facecolor="white",
                    edgecolor=non_significant_edgecolor,
                    linewidth=non_significant_linewidth,
                    label="Not significant",
                ),
            ]

            typology_legend = ax.legend(
                handles=typology_handles,
                title="Attribute Typology",
                loc="lower left",
                framealpha=0.9,
            )
            ax.add_artist(typology_legend)  # preserve first legend when adding second
            ax.legend(
                handles=significance_handles,
                title="Spatial Significance",
                loc="lower right",
                framealpha=0.9,
            )

            ax.set_title(
                f"Demand / Deprivation Attribute Typology\n"
                f"(bold border = significant spatial cluster, "
                f"method: {combination_method})",
                fontsize=10,
            )

            if show_basemap:
                cx.add_basemap(ax, crs=hotspots_df.crs.to_string())

            if not show_axis:
                ax.axis("off")

            return ax
