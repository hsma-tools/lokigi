import warnings

import contextily as cx
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from requests.exceptions import RequestException

from lokigi.utils import _min_max_normalize


class SFCAMixin:
    """
    Two-step floating catchment area (2SFCA) accessibility.

    Mixed into `SiteProblem`, giving it `two_step_floating_catchment()`.
    `SiteSolutionSet.two_step_floating_catchment()` resolves a solution's
    site list and delegates back to the `SiteProblem` method rather than
    duplicating this engine, since it has no travel frame of its own beyond
    what the problem already builds.
    """

    @staticmethod
    def _resolve_supply(candidate_sites, candidate_id_col, supply_col, site_names):
        if supply_col not in candidate_sites.columns:
            raise ValueError(
                f"supply_col={supply_col!r} not found in candidate_sites. "
                f"Available columns: {list(candidate_sites.columns)}."
            )

        indexed = candidate_sites.set_index(candidate_id_col)[supply_col]

        missing = [s for s in site_names if s not in indexed.index]
        if missing:
            raise KeyError(f"Sites not found in candidate_sites: {missing}.")

        supply = indexed.loc[site_names]

        if not pd.api.types.is_numeric_dtype(supply):
            raise TypeError(
                f"supply_col={supply_col!r} must contain numeric values. "
                "Hint: try pd.to_numeric() on this column before calling "
                "add_sites()."
            )

        null_sites = supply[supply.isna()].index.tolist()
        if null_sites:
            raise ValueError(
                f"The following sites are missing a value in "
                f"supply_col={supply_col!r}: {null_sites}."
            )

        negative_sites = supply[supply < 0].index.tolist()
        if negative_sites:
            raise ValueError(
                f"The following sites have a negative value in "
                f"supply_col={supply_col!r}, which is not a valid supply "
                f"quantity: {negative_sites}."
            )

        return supply

    @staticmethod
    def _two_step_floating_catchment(
        cost_frame, demand, supply, catchment_size, per_capita, return_site_ratios
    ):
        """
        Parameters
        ----------
        cost_frame : pandas.DataFrame
            Wide travel-cost frame: index = demand location IDs, columns =
            site names, matching `supply.index`.
        demand : pandas.Series
            Demand weight per demand location, aligned to `cost_frame.index`.
        supply : pandas.Series
            Supply quantity per site, aligned to `cost_frame.columns`.
        catchment_size : float
            Hard catchment threshold d0. Membership is inclusive (`<=`),
            unlike the library's coverage metrics, which use strict `<` --
            deliberate, matching the inclusive convention in the 2SFCA
            literature.
        per_capita : float
            Multiplier applied to `accessibility`, e.g. 1_000 to express
            supply per 1,000 head instead of raw supply units per head.
        return_site_ratios : bool
            If True, also return the step-1 per-site table.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, pandas.DataFrame)
        """
        in_catchment = cost_frame <= catchment_size

        # Step 1: supply-to-demand ratio per site.
        catchment_demand = in_catchment.mul(demand, axis=0).sum(axis=0)
        n_regions_in_catchment = in_catchment.sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = supply / catchment_demand
        # A site with nobody in its catchment divides by zero either way
        # (0/0 or S/0): both mean "this site's ratio is undefined", so both
        # collapse to NaN rather than one becoming +/-inf.
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        empty_sites = catchment_demand[catchment_demand == 0].index.tolist()
        if empty_sites:
            warnings.warn(
                "two_step_floating_catchment: the following sites have no "
                f"demand within catchment_size={catchment_size}, so their "
                "supply-to-demand ratio is undefined (NaN) and they are "
                f"excluded from every region's accessibility score: "
                f"{empty_sites}.",
                stacklevel=3,
            )

        site_frame = pd.DataFrame(
            {
                "supply": supply,
                "catchment_demand": catchment_demand,
                "n_regions_in_catchment": n_regions_in_catchment,
                "ratio": ratio,
            }
        )
        # catchment_demand/n_regions_in_catchment/ratio all derive their
        # index from cost_frame.columns, which carries no name -- without
        # this, pd.DataFrame(dict-of-Series) drops supply.index.name too,
        # silently turning site_frame.reset_index()'s site-name column
        # into a generic "index" rather than the candidate ID column name.
        site_frame.index.name = supply.index.name

        # Step 2: sum reachable sites' ratios per demand region. A NaN
        # ratio (empty-catchment site) must drop out of this sum rather
        # than propagate -- an unreachable site's undefined ratio
        # shouldn't poison every other region that happens to be near it
        # too.
        usable_ratio = ratio.fillna(0.0)
        accessibility = in_catchment.mul(usable_ratio, axis=1).sum(axis=1) * per_capita
        n_sites_in_catchment = in_catchment.sum(axis=1)

        region_frame = pd.DataFrame(
            {
                "accessibility": accessibility,
                "n_sites_in_catchment": n_sites_in_catchment,
                "demand": demand,
            }
        )

        if return_site_ratios:
            return region_frame, site_frame
        return region_frame

    def two_step_floating_catchment(
        self,
        supply_col,
        catchment_size,
        site_names=None,
        site_indices=None,
        matrix=None,
        per_capita=1,
        return_site_ratios=False,
    ):
        """
        Two-step floating catchment area (2SFCA) accessibility score.

        Unlike binary threshold coverage, 2SFCA accounts for competition:
        a demand region near a site that is also the closest option for
        many other regions gets less credit for that site than a region
        near an uncontested one. Scores an arbitrary set of sites directly
        -- no `solve()` is required, so it can describe any subset of
        `candidate_sites` as a baseline, e.g. only the sites that are
        already open, via `site_names`/`site_indices`. With neither
        argument, every registered candidate site is scored -- the right
        default when the candidate pool *is* the current network, but not
        when it also includes not-yet-built proposals, in which case pass
        the currently-open subset explicitly.

        Parameters
        ----------
        supply_col : str
            Column in `candidate_sites` holding each site's supply
            quantity (e.g. number of GPs, beds, weekly appointment slots).
            Named at call time rather than registered via `add_sites()`,
            so the same problem can be scored under different supply
            definitions without re-adding sites.
        catchment_size : float
            Hard catchment threshold d0, in the travel matrix's registered
            units. A site is "in catchment" for a demand region if the
            travel cost between them is `<= catchment_size`.
        site_names : list of str, optional
        site_indices : list of int, optional
            The site set to score. At most one may be given. If neither is
            given, every candidate site is scored.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Scores accessibility under
            that matrix's travel costs instead of the primary matrix's.
        per_capita : float, default 1
            Multiplier applied to the `accessibility` column, e.g. 1_000
            to express supply per 1,000 head instead of raw supply units
            per head. Does not affect `site_frame`.
        return_site_ratios : bool, default False
            If True, also return the step-1 per-site table -- useful for
            finding which site is driving an implausible regional score.

        Returns
        -------
        pandas.DataFrame
            Per demand region, indexed by demand location ID: `accessibility`
            (supply units per head, x `per_capita`), `n_sites_in_catchment`,
            `demand`.
        pandas.DataFrame
            Only if `return_site_ratios=True`. Per site, indexed by site
            name: `supply`, `catchment_demand`, `n_regions_in_catchment`,
            `ratio`.

        Raises
        ------
        ValueError
            If both `site_names` and `site_indices` are given, if no demand
            data is registered, if `supply_col` is missing or contains a
            null/negative value for a scored site, or if `matrix` is not a
            registered secondary travel matrix label.
        TypeError
            If `supply_col` is not numeric.
        IndexError
            If `site_indices` are out of range.
        KeyError
            If `site_names` are not found in the travel matrix or
            `candidate_sites`.

        Notes
        -----
        A site with no demand within `catchment_size` has an undefined
        (`NaN`) supply-to-demand ratio and is excluded from every region's
        accessibility score, with a warning naming it. A demand region with
        no site in `catchment_size` correctly scores `accessibility == 0`
        (a real "no supply available" result, not a missing value) --
        these two zero-like cases are deliberately kept distinguishable.

        `accessibility` obeys `sum(demand * accessibility) == sum(supply)`
        (before `per_capita` scaling) whenever every scored site has at
        least one region in its catchment, since the sum of demand-weighted
        ratios is exactly the supply that produced them.
        """
        if site_names is not None and site_indices is not None:
            raise ValueError(
                "Please provide at most one of 'site_names' or "
                "'site_indices', not both."
            )

        if self._demand_data_demand_col is None:
            raise ValueError(
                "two_step_floating_catchment requires demand data. Call "
                "add_demand() before using this method."
            )

        if self.travel_and_demand_df is None:
            self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
            self._build_secondary_travel_frames()

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
                .sort_values("canonical_site_index")[self._candidate_sites_candidate_id_col]
                .tolist()
            )
        elif site_names is not None:
            site_names_resolved = list(site_names)
        else:
            site_names_resolved = (
                self.candidate_sites.sort_values("canonical_site_index")[
                    self._candidate_sites_candidate_id_col
                ]
                .tolist()
            )

        if matrix is None:
            missing = [
                s for s in site_names_resolved if s not in self.travel_and_demand_df.columns
            ]
            if missing:
                raise KeyError(f"Sites not found in travel matrix: {missing}")
            cost_frame = self.travel_and_demand_df[site_names_resolved]
        else:
            if matrix not in self.secondary_travel_matrices:
                raise ValueError(
                    f"Unknown secondary travel matrix '{matrix}'. Registered "
                    f"labels: {sorted(self.secondary_travel_matrices)}."
                )
            frame = self._secondary_travel_frames[matrix]
            missing = [s for s in site_names_resolved if s not in frame.columns]
            if missing:
                raise KeyError(
                    f"Sites not found in secondary travel matrix '{matrix}': {missing}"
                )
            cost_frame = frame[site_names_resolved]

        demand = self.travel_and_demand_df[self._demand_data_demand_col].reindex(
            cost_frame.index
        )

        supply = self._resolve_supply(
            self.candidate_sites,
            self._candidate_sites_candidate_id_col,
            supply_col,
            site_names_resolved,
        )

        return self._two_step_floating_catchment(
            cost_frame=cost_frame,
            demand=demand,
            supply=supply,
            catchment_size=catchment_size,
            per_capita=per_capita,
            return_site_ratios=return_site_ratios,
        )


class AccessibilityPlotMixin:
    """
    Shared 2SFCA accessibility map for `SiteProblem` and `SiteSolutionSet`.

    Mirrors the `get_hotspots()`/`plot_hotspots()` split in `site_eda.py`:
    the plotting method optionally takes a precomputed result, computing it
    automatically via `two_step_floating_catchment()` if not supplied.
    """

    @property
    def _prob_ctx(self):
        """
        Dynamically routes requests to the object holding the problem data.
        Returns `self.site_problem` if attached to a `SiteSolutionSet`,
        otherwise `self` (already a `SiteProblem`).
        """
        return getattr(self, "site_problem", self)

    def plot_accessibility(
        self,
        region_frame=None,
        site_frame=None,
        supply_col=None,
        catchment_size=None,
        site_names=None,
        site_indices=None,
        matrix=None,
        per_capita=1,
        interactive=False,
        cmap="Blues",
        site_cmap="RdYlGn",
        missing_site_colour="lightgrey",
        marker_size_range=(40, 220),
        edgecolor="black",
        linewidth=0.5,
        tiles="CartoDB positron",
        add_basemap=True,
        show_axis=False,
        title=None,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Map 2SFCA accessibility: a region choropleth of `accessibility`,
        overlaid with site markers coloured and sized by their step-1
        supply-to-demand `ratio` -- so a map reader can see both where
        access is poor and which site is driving it (an overloaded,
        low-ratio site drawn small and red) in one view.

        Parameters
        ----------
        region_frame, site_frame : pandas.DataFrame, optional
            The two tables returned by
            `two_step_floating_catchment(return_site_ratios=True)`. If
            either is None (the default), both are computed automatically
            from `supply_col`/`catchment_size` and the selection arguments
            below.
        supply_col, catchment_size, site_names, site_indices, matrix,
        per_capita
            Forwarded to `two_step_floating_catchment()` when
            `region_frame`/`site_frame` are not supplied; see that
            method's docstring. On a `SiteSolutionSet`, this always scores
            `solution_rank=1` -- to plot a specific `rank_on`/
            `solution_rank` solution, call
            `two_step_floating_catchment(..., return_site_ratios=True)`
            yourself and pass the results in as `region_frame`/`site_frame`.
        interactive : bool, default False
            If True, returns an interactive Folium map via `.explore()`.
            Otherwise returns a static matplotlib Axes.
        cmap : str, default "Blues"
            Colormap for the region choropleth (`accessibility`).
        site_cmap : str, default "RdYlGn"
            Colormap for site markers (`ratio`) -- red for an overloaded,
            low-ratio site, green for a relatively uncontested one.
        missing_site_colour : str, default "lightgrey"
            Colour (and static marker size, at the smallest of
            `marker_size_range`) for a site with an undefined `ratio` --
            no demand fell within its catchment, so there is nothing to
            colour or size it by.
        marker_size_range : tuple of (float, float), default (40, 220)
            Smallest and largest static marker size, linearly scaled by
            `ratio`. Ignored on interactive maps, where Folium markers are
            a fixed size.
        add_basemap : bool, default True
            If True, adds a background web map. Set False to skip the tile
            download entirely.
        title : str, optional
            Axes title. Ignored on interactive maps.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto. Ignored if `interactive=True`.
        figsize : tuple, optional
            Passed to `plt.subplots()` if `ax` is not supplied. Ignored if
            `interactive=True`.
        **kwargs : dict
            Additional keyword arguments passed to the region choropleth's
            plotting call (`GeoDataFrame.plot`/`.explore`).

        Returns
        -------
        matplotlib.axes.Axes or folium.Map

        Raises
        ------
        ValueError
            If neither `region_frame`/`site_frame` nor both `supply_col`
            and `catchment_size` are supplied, or if no region geometry
            layer has been registered via `add_region_geometry_layer()`.

        Notes
        -----
        Site markers are only drawn if `candidate_sites` was registered
        with real geometry (i.e. `add_sites()` was given a GeoDataFrame or
        lat/long columns, not a bare site list derived from the travel
        matrix's column names) -- otherwise only the region choropleth is
        shown.
        """
        ctx = self._prob_ctx

        if ctx.region_geometry_layer is None:
            raise ValueError(
                "No region geometry layer has been initialised. Please run "
                "`.add_region_geometry_layer()` first."
            )

        if region_frame is None or site_frame is None:
            if supply_col is None or catchment_size is None:
                raise ValueError(
                    "Either pass precomputed `region_frame`/`site_frame` "
                    "(from `two_step_floating_catchment(return_site_ratios="
                    "True)`), or both `supply_col` and `catchment_size` so "
                    "they can be computed automatically."
                )
            region_frame, site_frame = self.two_step_floating_catchment(
                supply_col=supply_col,
                catchment_size=catchment_size,
                site_names=site_names,
                site_indices=site_indices,
                matrix=matrix,
                per_capita=per_capita,
                return_site_ratios=True,
            )

        region_gdf = ctx.region_geometry_layer.merge(
            region_frame.reset_index(),
            left_on=ctx._region_geometry_layer_common_col,
            right_on=ctx._demand_data_id_col,
            suffixes=("", "_y"),
        )
        region_gdf = region_gdf.drop(region_gdf.filter(regex="_y$").columns, axis=1)

        # NOTE: deliberately not `ctx._candidate_sites_type == "geopandas"`.
        # That attribute records add_sites()'s *input* format, not whether
        # self.candidate_sites ended up with real geometry -- tabular
        # lat/long input is converted to a GeoDataFrame internally but
        # still leaves _candidate_sites_type == "pandas". Checking the
        # actual type of candidate_sites is correct for both input paths;
        # it is only a plain DataFrame with no geometry when sites were
        # never registered via add_sites() at all (site names inferred
        # from the travel matrix's columns instead).
        has_site_geometry = isinstance(ctx.candidate_sites, geopandas.GeoDataFrame)
        site_gdf = None
        if has_site_geometry:
            site_gdf = ctx.candidate_sites.merge(
                site_frame.reset_index(),
                left_on=ctx._candidate_sites_candidate_id_col,
                right_on=ctx._candidate_sites_candidate_id_col,
                suffixes=("", "_y"),
            )
            site_gdf = site_gdf.drop(site_gdf.filter(regex="_y$").columns, axis=1)

            min_size, max_size = marker_size_range
            site_gdf["_marker_size"] = (
                min_size
                + _min_max_normalize(site_gdf["ratio"], constant_fill=1.0)
                * (max_size - min_size)
            ).fillna(min_size)

        if interactive:
            m = region_gdf.explore(
                column="accessibility",
                tooltip=[
                    ctx._region_geometry_layer_common_col,
                    "accessibility",
                    "n_sites_in_catchment",
                ],
                popup=True,
                cmap=cmap,
                style_kwds=dict(color="black"),
                tiles=tiles if add_basemap else None,
                **kwargs,
            )

            if has_site_geometry:
                site_gdf.explore(
                    m=m,
                    column="ratio",
                    cmap=site_cmap,
                    tooltip=[
                        ctx._candidate_sites_candidate_id_col,
                        "supply",
                        "catchment_demand",
                        "ratio",
                    ],
                    popup=True,
                    marker_kwds=dict(radius=8),
                    missing_kwds=dict(color=missing_site_colour),
                )

            return m

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        region_gdf.plot(
            column="accessibility",
            legend=True,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            legend_kwds={"label": "Accessibility"},
            **kwargs,
        )

        if has_site_geometry:
            site_gdf.plot(
                column="ratio",
                cmap=site_cmap,
                markersize=site_gdf["_marker_size"],
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
                legend=True,
                legend_kwds={"label": "Site supply:demand ratio", "shrink": 0.6},
                missing_kwds=dict(color=missing_site_colour, label="No catchment demand"),
            )

            # geopandas' missing_kwds `label` only surfaces in a discrete
            # `scheme=` legend, not the continuous colorbar used here, so
            # the grey "no catchment demand" markers would otherwise have
            # no legend entry explaining them.
            if site_gdf["ratio"].isna().any():
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
                            label="No catchment demand",
                        )
                    ],
                    loc="lower left",
                )

        if add_basemap:
            try:
                cx.add_basemap(ax, crs=region_gdf.crs.to_string(), timeout=30)
            except RequestException as e:
                warnings.warn(
                    f"Unable to download background map tiles ({type(e).__name__}). "
                    "Continuing without a basemap.",
                    stacklevel=2,
                )

        if title:
            ax.set_title(title)

        if not show_axis:
            ax.axis("off")

        return ax
