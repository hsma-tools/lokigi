import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from lokigi.plot_utils import _add_plot_caption, plot_solution_sets_comparison
from lokigi.utils import _select_solution, _split_bins_into_tertiles


class SolutionComparatorPlotsMixin:
    def plot_comparison(
        self,
        config_1=None,
        config_2=None,
        figsize=(16, 8),
        title=None,
        title_fontsize=14,
        **shared_plot_kwargs,
    ):
        """
        Plot solutions from both solution sets side-by-side.

        Convenience wrapper around plot_solution_sets_comparison() for comparing
        the two solution sets managed by this comparator.

        Parameters
        ----------
        config_1 : dict, optional
            Configuration for plotting from solution_set_1. If None, plots the
            best solution (solution_rank=1).
        config_2 : dict, optional
            Configuration for plotting from solution_set_2. If None, plots the
            best solution (solution_rank=1).
        figsize : tuple, default=(16, 8)
            Figure size as (width, height) in inches.
        title : str, optional
            Overall figure title. If None, generates a default title.
        title_fontsize : int, default=14
            Font size for the overall figure title.
        **shared_plot_kwargs
            Keyword arguments applied to both subplots.

        Returns
        -------
        tuple
            (fig, axes) where fig is the Figure and axes is an array of Axes objects.

        Examples
        --------
        # Compare best solutions from both sets
        fig, axes = comparator.plot_comparison()

        # Compare specific site configurations
        balanced_car, balanced_pt = comparator.get_balanced_solution()
        fig, axes = comparator.plot_comparison(
            config_1={'site_indices': balanced_car},
            config_2={'site_indices': balanced_pt},
            title='Balanced Solutions: Car vs Public Transport'
        )

        # Compare ranked solutions
        fig, axes = comparator.plot_comparison(
            config_1={'solution_rank': 2, 'sort_by': 'weighted_average'},
            config_2={'solution_rank': 2, 'sort_by': 'weighted_average'},
            title='2nd Best Solutions Comparison'
        )
        """
        # Set defaults
        if config_1 is None:
            config_1 = {"solution_rank": 1}
        if config_2 is None:
            config_2 = {"solution_rank": 1}

        # Add default titles if not provided
        if "title" not in config_1:
            config_1["title"] = "Solution Set 1"
        if "title" not in config_2:
            config_2["title"] = "Solution Set 2"

        # Generate default overall title if needed
        if title is None:
            title = "Solution Set Comparison"

        # Call standalone function
        return plot_solution_sets_comparison(
            solution_sets=[self.set_a, self.set_b],
            solutions_config=[config_1, config_2],
            figsize=figsize,
            title=title,
            title_fontsize=title_fontsize,
            **shared_plot_kwargs,
        )

    def plot_population_impact_summary(
        self,
        by="demand",
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
        colors=("#9fb8ad", "#1b7a5e"),
        figsize=(9, 4),
        title="default",
        ax=None,
    ):
        """
        Two-panel bar chart pairing `population_impact_summary()`'s
        region-wide and per-region views: the `weighted_average` shift
        between `set_a` (baseline) and `set_b` (candidate) on the left,
        against how many people (or regions) actually experienced a
        change on the right. Makes visible the "dilution" a region-wide
        average alone can hide -- a large, genuinely local effect can
        move `weighted_average` by very little once averaged across
        everyone else who is unaffected by it.

        Parameters
        ----------
        by : {"demand", "regions"}, default "demand"
            Whether the right-hand panel counts people
            (`demand_improved`/`demand_unchanged`, demand-weighted) or
            regions (`regions_improved`/`regions_unchanged`). "demand"
            raises `ValueError` if no demand data is registered on the
            problem, matching `site_allocation_summary()`'s `by="demand"`
            behaviour; "regions" always works.
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_summary()`.
        colors : (str, str), default ("#9fb8ad", "#1b7a5e")
            `(unchanged_color, changed_color)`. The left panel always
            colours `set_a` with `unchanged_color` and `set_b` with
            `changed_color`; the right panel colours the "improved" bar
            with `changed_color` and "unchanged" with `unchanged_color`.
        figsize : tuple, default (9, 4)
            Figure size, ignored if `ax` is given.
        title : str, default "default"
            Overall figure title. If "default", an automatic title is
            generated.
        ax : (matplotlib.axes.Axes, matplotlib.axes.Axes), optional
            A pair of existing Axes to draw the two panels into instead
            of creating a new figure -- e.g. to embed this as two panels
            of a larger layout.

        Returns
        -------
        (matplotlib.figure.Figure, (matplotlib.axes.Axes, matplotlib.axes.Axes))
        """
        if by not in ("demand", "regions"):
            raise ValueError(f"by must be 'demand' or 'regions', got {by!r}.")

        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        solution_a = _select_solution(self.set_a.solution_df, **config_a)
        solution_b = _select_solution(self.set_b.solution_df, **config_b)

        # Resolved via set_b, matching population_impact_summary()'s own
        # convention -- the suffix is empty for the primary matrix, so this
        # is a no-op unless matrix= actually names a secondary one.
        _, _, _, _, suffix = self.set_b._resolve_travel_columns(matrix)
        weighted_average_a = solution_a.iloc[0][f"weighted_average{suffix}"]
        weighted_average_b = solution_b.iloc[0][f"weighted_average{suffix}"]

        impact = self.population_impact_summary(
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
            as_dict=True,
        )

        if by == "demand" and pd.isna(impact["total_demand"]):
            raise ValueError(
                "by='demand' requires demand data to be registered on the "
                "problem -- call add_demand(), or pass by='regions' to "
                "count regions instead."
            )

        changed_key = "demand_improved" if by == "demand" else "regions_improved"
        unchanged_key = "demand_unchanged" if by == "demand" else "regions_unchanged"
        y_label = "People" if by == "demand" else "Regions"

        unchanged_color, changed_color = colors

        if ax is None:
            fig, axes = plt.subplots(ncols=2, figsize=figsize, constrained_layout=True)
        else:
            axes = ax
            fig = axes[0].get_figure()

        axes[0].bar(
            [self.labels[0], self.labels[1]],
            [weighted_average_a, weighted_average_b],
            color=[unchanged_color, changed_color],
        )
        axes[0].set_ylabel("Weighted average travel cost")
        axes[0].set_title(f"Region-wide shift: {weighted_average_b - weighted_average_a:+.2f}")

        axes[1].bar(
            ["Improved", "Unchanged"],
            [impact[changed_key], impact[unchanged_key]],
            color=[changed_color, unchanged_color],
        )
        axes[1].set_ylabel(y_label)
        axes[1].set_title(f"{impact[changed_key]:,.0f} {y_label.lower()} improved")

        if title == "default":
            title = "A region-wide average can hide a concentrated local effect"
        if title:
            fig.suptitle(title)

        return fig, axes

    def plot_site_reallocation_matrix(
        self,
        matrix=None,
        demand=None,
        config_a=None,
        config_b=None,
        by="demand",
        changed_only=False,
        cmap="Blues",
        annot=True,
        fmt=",.0f",
        figsize=(8, 6),
        title="default",
        caption=None,
        ax=None,
    ):
        """
        Heatmap of `site_reallocation_matrix()`: rows = `set_a`'s selected
        sites, columns = `set_b`'s, cell colour/annotation = the demand
        (or region count) whose closest site moved from that row's site
        to that column's site. A closed site's row and a newly opened
        site's column are the reallocation the table exists to show; the
        diagonal is the demand that stayed with the same site in both
        solutions. Both axes are ordered persisting sites first, then
        that axis's own changed sites last (see `site_reallocation_matrix`),
        so the actual reallocation is grouped at the bottom/right rather
        than scattered wherever an unrelated unchanged site's candidate
        index happens to fall.

        Parameters
        ----------
        matrix, demand, config_a, config_b, by, changed_only
            Passed straight through to `site_reallocation_matrix()`.
            `changed_only=True` drops every row/column that saw no
            reallocation at all, useful once the full matrix is too big
            (many unaffected sites) to scan comfortably as a heatmap.
        cmap : str, default "Blues"
            Colormap for the heatmap. Sequential rather than diverging --
            unlike `plot_population_impact_map()`'s travel-cost change,
            reallocated demand has no natural "centre" and is never
            negative.
        annot : bool, default True
            Whether to print each cell's value on the heatmap
            (`seaborn.heatmap`'s own `annot`).
        fmt : str, default ",.0f"
            Format string for the cell annotations.
        figsize : tuple, default (8, 6)
            Figure size, ignored if `ax` is given.
        title : str, default "default"
            Title. If "default", an automatic title naming `set_a`/`set_b`
            by their comparator labels is generated.
        caption : str, optional
            Explanatory text shown below the chart, wrapped to fit. If
            `None` (the default), a sensible stakeholder-facing caption
            explaining how to read the heatmap is generated; pass `""` to
            suppress it, or a custom string to replace it.
        ax : matplotlib.axes.Axes, optional
            Existing Axes to draw into instead of creating a new figure.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        reallocation = self.site_reallocation_matrix(
            matrix=matrix,
            demand=demand,
            config_a=config_a,
            config_b=config_b,
            by=by,
            changed_only=changed_only,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        cbar_label = "People" if by == "demand" else "Regions"
        sns.heatmap(
            reallocation,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            ax=ax,
            cbar_kws={"label": cbar_label},
        )
        ax.set_xlabel(self.labels[1])
        ax.set_ylabel(self.labels[0])

        if title == "default":
            title = f"Site reallocation: {self.labels[0]} to {self.labels[1]}"
        if title:
            ax.set_title(title)

        default_caption = (
            f"How to read this: read a row across to see where that site's "
            f"demand ({self.labels[0]}) ended up in {self.labels[1]}; read a "
            "column down to see which sites its demand came from. The "
            "diagonal is demand that stayed with the same site. Sites "
            "unique to one side -- closed (bottom rows) or newly opened "
            "(rightmost columns) -- are grouped at the end of their axis "
            "rather than sorted by candidate index."
        )
        _add_plot_caption(fig, caption, default_caption)

        return fig, ax

    def plot_population_impact_histogram(
        self,
        matrix=None,
        demand=None,
        config_a=None,
        config_b=None,
        kind="kde",
        bins=30,
        bw_adjust=1,
        colors=("#9fb8ad", "#1b7a5e"),
        alpha=0.55,
        figsize=(9, 5),
        title="default",
        caption=None,
        ax=None,
    ):
        """
        Overlaid before/after distribution of per-region travel cost --
        `set_a` (baseline) against `set_b` (candidate) -- with reference
        lines and a legend annotating each side's weighted mean and
        maximum travel cost.

        The distributional counterpart to `population_impact_summary()`'s
        aggregate numbers and `plot_population_impact_summary()`'s bar
        charts: those answer "how many, how much" and "how did the
        average move", this shows the whole shape of the shift -- e.g. a
        long unaffected tail, or a bimodal split between a helped
        subgroup and everyone else, that a handful of summary numbers can
        collapse.

        Weighted by demand where available, so the plotted mass
        represents people rather than regions -- consistent with
        `population_impact_summary()`'s own demand-weighted framing.
        Falls back to unweighted (one region = one count) when no demand
        data is registered.

        Parameters
        ----------
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Plots that matrix's own
            travel costs instead of the primary matrix's.
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Weights the distribution by that
            scenario's demand instead of the primary demand data.
        config_a, config_b : dict, optional
            Passed straight through to `population_impact_summary()`.
        kind : {"kde", "hist"}, default "kde"
            "kde" draws a smoothed kernel density estimate for each side
            (via `seaborn.kdeplot`) -- usually the easier way to compare
            the *shape* of two overlaid distributions, since it isn't
            broken up into discrete bars that can visually clash between
            the two sides. "hist" draws a traditional binned histogram
            instead, which is more literal (bar height is an actual
            people/region count rather than a smoothed density) at the
            cost of being noisier and more prone to visual clutter where
            the two distributions overlap.
        bins : int, default 30
            Number of histogram bins. Only used when `kind="hist"`; the
            same bin edges (spanning both distributions) are used for
            both sides, so bar heights are directly comparable.
        bw_adjust : float, default 1
            Kernel bandwidth multiplier, forwarded to `seaborn.kdeplot`.
            Only used when `kind="kde"` -- above 1 smooths the curve
            further, below 1 follows the data more closely (and more
            noisily). See `seaborn.kdeplot`'s own `bw_adjust` for details.
        colors : (str, str), default ("#9fb8ad", "#1b7a5e")
            `(set_a colour, set_b colour)`, used for both the
            distributions and their matching mean/max reference lines.
        alpha : float, default 0.55
            Opacity of the filled distributions, so the overlap between
            the two sides stays visible.
        figsize : tuple, default (9, 5)
            Figure size, ignored if `ax` is given.
        title : str, default "default"
            Plot title. If "default", an automatic title is generated.
        caption : str, optional
            Explanatory text shown below the chart, wrapped to fit. If
            `None` (the default) and `kind="kde"`, a caption explaining
            that curve height is a relative share rather than a literal
            headcount is generated -- a plain histogram's bar heights need
            no such explanation, so `kind="hist"` adds none by default.
            Pass `""` to suppress it, or a custom string to replace it.
        ax : matplotlib.axes.Axes, optional
            An existing Axes to draw into instead of creating a new
            figure -- e.g. to embed this in a larger layout.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)

        Raises
        ------
        ValueError
            If `kind` is not "kde" or "hist"; if `demand` names an
            unregistered secondary demand scenario; or if `set_a` and
            `set_b`'s selected solutions were evaluated against different
            demand locations (see `population_impact_summary()`).
        """
        if kind not in ("kde", "hist"):
            raise ValueError(f"kind must be 'kde' or 'hist', got {kind!r}.")

        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        solution_a = _select_solution(self.set_a.solution_df, **config_a)
        solution_b = _select_solution(self.set_b.solution_df, **config_b)

        cost_col, _, _, unit, suffix = self.set_b._resolve_travel_columns(matrix)
        weighted_average_a = solution_a.iloc[0][f"weighted_average{suffix}"]
        weighted_average_b = solution_b.iloc[0][f"weighted_average{suffix}"]
        max_a = solution_a.iloc[0][f"max{suffix}"]
        max_b = solution_b.iloc[0][f"max{suffix}"]

        _, per_region = self.population_impact_summary(
            matrix=matrix,
            demand=demand,
            config_a=config_a,
            config_b=config_b,
            return_per_region=True,
            as_dict=True,
        )

        has_demand = "demand" in per_region.columns
        weights = per_region["demand"].to_numpy() if has_demand else None

        color_a, color_b = colors

        if ax is None:
            fig, axis = plt.subplots(figsize=figsize)
        else:
            axis = ax
            fig = axis.get_figure()

        if kind == "hist":
            bin_edges = np.histogram_bin_edges(
                np.concatenate(
                    [
                        per_region["baseline_cost"].to_numpy(),
                        per_region["current_cost"].to_numpy(),
                    ]
                ),
                bins=bins,
            )
            axis.hist(
                per_region["baseline_cost"], bins=bin_edges, weights=weights, color=color_a,
                alpha=alpha,
            )
            axis.hist(
                per_region["current_cost"], bins=bin_edges, weights=weights, color=color_b,
                alpha=alpha,
            )
            y_label = f"{'People' if has_demand else 'Regions'}"
        else:
            # clip=(0, None): travel cost can't be negative, so don't let
            # the kernel's tails spill into a nonsensical negative range
            # near zero.
            for values, color in (
                (per_region["baseline_cost"], color_a),
                (per_region["current_cost"], color_b),
            ):
                sns.kdeplot(
                    x=values,
                    weights=weights,
                    color=color,
                    fill=True,
                    alpha=alpha,
                    bw_adjust=bw_adjust,
                    clip=(0, None),
                    ax=axis,
                )
            y_label = f"Relative share of {'people' if has_demand else 'regions'}"

        axis.axvline(weighted_average_a, color=color_a, linestyle="--", linewidth=2)
        axis.axvline(weighted_average_b, color=color_b, linestyle="--", linewidth=2)
        axis.axvline(max_a, color=color_a, linestyle=":", linewidth=2)
        axis.axvline(max_b, color=color_b, linestyle=":", linewidth=2)

        unit_suffix = f" {unit}" if unit else ""
        unit_parenthetical = f" ({unit})" if unit else ""
        legend_handles = [
            Line2D([0], [0], color=color_a, alpha=alpha, linewidth=8, label=self.labels[0]),
            Line2D([0], [0], color=color_b, alpha=alpha, linewidth=8, label=self.labels[1]),
            Line2D(
                [0],
                [0],
                color="grey",
                linestyle="--",
                linewidth=2,
                label=f"Mean -- {self.labels[0]}: {weighted_average_a:.1f}{unit_suffix}, "
                f"{self.labels[1]}: {weighted_average_b:.1f}{unit_suffix}",
            ),
            Line2D(
                [0],
                [0],
                color="grey",
                linestyle=":",
                linewidth=2,
                label=f"Max -- {self.labels[0]}: {max_a:.1f}{unit_suffix}, "
                f"{self.labels[1]}: {max_b:.1f}{unit_suffix}",
            ),
        ]
        axis.legend(handles=legend_handles, loc="upper right", fontsize=9)

        axis.set_xlabel(f"Travel cost{unit_parenthetical}")
        axis.set_ylabel(y_label)

        if title == "default":
            title = "Travel cost distribution: before vs after"
        if title:
            axis.set_title(title)

        default_caption = None
        if kind == "kde":
            cohort = "people" if has_demand else "regions"
            default_caption = (
                f"How to read this: curve height is the relative share of "
                f"{cohort} at each travel time, not a literal headcount -- a "
                "taller peak means more concentrated here, not more affected "
                "overall. Pass kind=\"hist\" for literal counts instead."
            )
        _add_plot_caption(fig, caption, default_caption)

        return fig, axis

    def plot_population_impact_map(
        self,
        direction="all",
        n=None,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
        cmap=None,
        background_color="#dddddd",
        edgecolor="black",
        linewidth=0.3,
        show_sites=None,
        site_color="black",
        site_markersize=40,
        closed_site_color="black",
        closed_site_marker="X",
        closed_site_markersize=120,
        figsize=(9, 7),
        title="default",
        legend=True,
        ax=None,
    ):
        """
        Map version of `population_impact_worst_affected()` --
        colours each demand location's region by how its journey changed
        between `set_a` (baseline) and `set_b` (candidate), so a reader
        can see *where* an aggregate figure like "9,669 people worsened"
        actually falls, not just how many.

        Requires a `region_geometry_layer` registered on `set_b`'s
        `SiteProblem` via `add_region_geometry_layer()`.

        Parameters
        ----------
        direction : {"all", "worsened", "improved"}, default "all"
            "all" colours every region by its signed change (`delta`,
            `current_cost - baseline_cost`) on a diverging scale centred
            on 0 -- red for a longer journey, blue for a shorter one,
            white/pale for no meaningful change. This is the only mode
            where the scale is genuinely diverging, since it is the only
            one showing both directions at once.
            "worsened"/"improved" colour only that bucket (per
            `population_impact_summary()`'s `meaningful_change_threshold`
            -gated classification) on a plain sequential scale from 0 to
            the largest change in that direction; every other region is
            drawn in `background_color` for geographic context.
        n : int, optional
            Restrict the coloured regions to the `n` most affected,
            exactly matching `population_impact_worst_affected(n=n,
            direction=direction, ...)` -- the two are meant to be shown
            together. Only valid when `direction` is "worsened" or
            "improved"; raises `ValueError` alongside `direction="all"`,
            since "all" already means "every region" and silently
            ignoring `n` there would be confusing.
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_summary()`.
        cmap : str, optional
            Matplotlib colormap name. Defaults to `"RdBu_r"` for
            `direction="all"` (diverging: red=longer, blue=shorter),
            `"Reds"` for `"worsened"`, `"Blues"` for `"improved"`.
        background_color : str, default "#dddddd"
            Fill colour for regions outside the coloured bucket. Ignored
            when `direction="all"`, since every region is coloured there.
        edgecolor, linewidth
            Region outline styling, applied to every region.
        show_sites : {None, "all", "closed"}, optional
            Overlay candidate sites as points, using the same point
            geometry `plot_best_combination()` draws from (`add_sites()`
            with lat/long columns, or a geopandas input). `None`
            (default) draws no site markers.
            "all" plots every candidate site on `set_b`'s `SiteProblem`:
            sites closed relative to the baseline (in `set_a`'s selected
            solution but not `set_b`'s) are drawn as a distinct marker
            (`closed_site_marker`/`closed_site_color`/
            `closed_site_markersize`); every other site (open in `set_b`,
            or never open in either solution) is drawn as a plain point
            (`site_color`/`site_markersize`).
            "closed" draws only the closed sites -- the closure itself,
            without the clutter of every other site.
        site_color, site_markersize : default "black", 40
            Styling for open/context site markers (`show_sites="all"`
            only).
        closed_site_color, closed_site_marker, closed_site_markersize :
        default "black", "X", 120
            Styling for the closed-site markers. `closed_site_marker`'s
            shape ("X" vs `site_color`'s plain circle) is what
            distinguishes the two by default -- deliberately not colour,
            since a red marker would clash with the "worsened"/"all"
            sequential and diverging red scales this plot itself uses.
            Pass a different `closed_site_color` (e.g. a bright colour
            not otherwise used by `cmap`) if the two need to stand apart
            more strongly, e.g. against a busy basemap.
        figsize : tuple, default (9, 7)
            Figure size, ignored if `ax` is given.
        title : str, default "default"
            Plot title. If "default", an automatic title is generated.
        legend : bool, default True
            Whether to draw a colour bar, labelled with the travel-time
            unit if one was set via `add_travel_matrix(unit=...)`.
        ax : matplotlib.axes.Axes, optional
            An existing Axes to draw into instead of creating a new
            figure -- e.g. to embed this in a larger layout.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)

        Raises
        ------
        ValueError
            If `direction` isn't "all"/"worsened"/"improved"; if `n` is
            given alongside `direction="all"`; if `show_sites` isn't
            `None`/"all"/"closed"; if `show_sites` is given but `set_b`'s
            `SiteProblem` has no point geometry on its candidate sites;
            if `set_b`'s `SiteProblem` has no `region_geometry_layer`
            registered; or any of the errors `population_impact_summary()`
            raises.
        """
        if direction not in ("all", "worsened", "improved"):
            raise ValueError(
                f'direction must be "all", "worsened", or "improved", got '
                f"{direction!r}."
            )
        if direction == "all" and n is not None:
            raise ValueError(
                'n is only valid when direction is "worsened" or "improved" '
                '-- "all" already plots every region.'
            )
        if show_sites not in (None, "all", "closed"):
            raise ValueError(
                f'show_sites must be None, "all", or "closed", got '
                f"{show_sites!r}."
            )

        site_problem = self.set_b.site_problem
        if site_problem.region_geometry_layer is None:
            raise ValueError(
                "plot_population_impact_map() requires a "
                "region_geometry_layer -- call add_region_geometry_layer() "
                "on set_b's SiteProblem first."
            )
        if show_sites is not None and not isinstance(
            site_problem.candidate_sites, geopandas.GeoDataFrame
        ):
            raise ValueError(
                "show_sites requires point geometry on set_b's SiteProblem's "
                "candidate sites -- register add_sites() with lat/long "
                "columns, or a geopandas GeoDataFrame."
            )

        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        _, per_region = self.population_impact_summary(
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
            return_per_region=True,
        )

        id_col = per_region.index.name
        common_col = site_problem._region_geometry_layer_common_col
        gdf = pd.merge(
            site_problem.region_geometry_layer,
            per_region.reset_index(),
            left_on=common_col,
            right_on=id_col,
            suffixes=("", "_y"),
        )
        gdf = gdf.drop(gdf.filter(regex="_y$").columns, axis=1)

        _, _, _, unit, _ = self.set_b._resolve_travel_columns(matrix)
        unit_parenthetical = f" ({unit})" if unit else ""

        if ax is None:
            fig, axis = plt.subplots(figsize=figsize)
        else:
            axis = ax
            fig = axis.get_figure()

        if direction == "all":
            resolved_cmap = cmap or "RdBu_r"
            # min/max nudged away from exactly 0 -- TwoSlopeNorm requires
            # vmin < vcenter < vmax strictly, which a candidate that only
            # ever improves (or only ever worsens) things would otherwise
            # violate.
            delta_min = min(gdf["delta"].min(), -1e-9)
            delta_max = max(gdf["delta"].max(), 1e-9)
            norm = TwoSlopeNorm(vmin=delta_min, vcenter=0, vmax=delta_max)

            plot_kwargs = {}
            if legend:
                plot_kwargs["legend"] = True
                plot_kwargs["legend_kwds"] = {
                    "label": f"Change in travel time{unit_parenthetical}"
                }
            gdf.plot(
                column="delta",
                cmap=resolved_cmap,
                norm=norm,
                edgecolor=edgecolor,
                linewidth=linewidth,
                ax=axis,
                **plot_kwargs,
            )
            default_title = "Change in travel time by region"
        else:
            resolved_cmap = cmap or ("Reds" if direction == "worsened" else "Blues")

            if n is not None:
                worst_affected = self.population_impact_worst_affected(
                    n=n,
                    direction=direction,
                    matrix=matrix,
                    demand=demand,
                    meaningful_change_threshold=meaningful_change_threshold,
                    config_a=config_a,
                    config_b=config_b,
                )
                subset = gdf[gdf[id_col].isin(worst_affected.index)]
            else:
                subset = gdf[gdf["bucket"] == direction]

            gdf.plot(color=background_color, edgecolor=edgecolor, linewidth=linewidth, ax=axis)

            if len(subset) > 0:
                magnitude = subset["delta"].abs()
                change_word = "Increase" if direction == "worsened" else "Reduction"
                plot_kwargs = {}
                if legend:
                    plot_kwargs["legend"] = True
                    plot_kwargs["legend_kwds"] = {
                        "label": f"{change_word} in travel time{unit_parenthetical}"
                    }
                subset.plot(
                    column=magnitude,
                    cmap=resolved_cmap,
                    vmin=0,
                    vmax=magnitude.max(),
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    ax=axis,
                    **plot_kwargs,
                )

            default_title = (
                "Regions with a longer journey"
                if direction == "worsened"
                else "Regions with a shorter journey"
            )

        if show_sites is not None:
            baseline_names = set(
                _select_solution(self.set_a.solution_df, **config_a).iloc[0][
                    "site_names"
                ]
            )
            candidate_names = set(
                _select_solution(self.set_b.solution_df, **config_b).iloc[0][
                    "site_names"
                ]
            )
            closed_names = baseline_names - candidate_names

            candidate_sites = site_problem.candidate_sites
            site_id_col = site_problem._candidate_sites_candidate_id_col
            closed_sites = candidate_sites[
                candidate_sites[site_id_col].isin(closed_names)
            ]

            if show_sites == "all":
                other_sites = candidate_sites[
                    ~candidate_sites[site_id_col].isin(closed_names)
                ]
                if len(other_sites) > 0:
                    other_sites.plot(ax=axis, color=site_color, markersize=site_markersize)

            if len(closed_sites) > 0:
                closed_sites.plot(
                    ax=axis,
                    color=closed_site_color,
                    marker=closed_site_marker,
                    markersize=closed_site_markersize,
                )

        axis.set_axis_off()
        if title == "default":
            title = default_title
        if title:
            axis.set_title(title)

        return fig, axis

    def plot_population_impact_by_equity_group(
        self,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
        colors=("#1b7a5e", "#b23a48"),
        figsize=(10, 5),
        title="default",
        ax=None,
    ):
        """
        Bar chart of `population_impact_by_equity_group()`'s rate-
        normalised improved/worsened proportions -- one paired bar per
        equity band, ordered most- to least-disadvantaged.

        Deliberately plots BOTH directions side by side rather than only
        the improved rate: a candidate can win on `weighted_average` while
        making some disadvantaged areas worse off, and a benefits-only
        chart would hide exactly that. Each bar is a share of THAT band's
        own population (not a share of the region-wide improved/worsened
        total), so bands with different population sizes are directly
        comparable.

        Parameters
        ----------
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_by_equity_group()`.
        colors : (str, str), default ("#1b7a5e", "#b23a48")
            `(improved_color, worsened_color)`.
        figsize : tuple, default (10, 5)
            Figure size, ignored if `ax` is given.
        title : str, default "default"
            Plot title. If "default", an automatic title is generated.
        ax : matplotlib.axes.Axes, optional
            An existing Axes to draw into instead of creating a new
            figure -- e.g. to embed this in a larger layout.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)

        Raises
        ------
        ValueError
            If no equity data is registered on the problem, or any of the
            errors `population_impact_summary()` raises -- see
            `population_impact_by_equity_group()`.
        """
        by_band = self.population_impact_by_equity_group(
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
        )

        improved_color, worsened_color = colors
        bands = [str(b) for b in by_band.index]
        x = np.arange(len(bands))
        width = 0.38

        if ax is None:
            fig, axis = plt.subplots(figsize=figsize)
        else:
            axis = ax
            fig = axis.get_figure()

        axis.bar(
            x - width / 2,
            by_band["proportion_of_band_improved"] * 100,
            width=width,
            color=improved_color,
            label="Improved",
        )
        axis.bar(
            x + width / 2,
            by_band["proportion_of_band_worsened"] * 100,
            width=width,
            color=worsened_color,
            label="Worsened",
        )

        equity_axis_label = (
            self.set_b.site_problem._equity_data_label or by_band.index.name
        )
        axis.set_xticks(x)
        axis.set_xticklabels(bands)
        axis.set_xlabel(
            f"{equity_axis_label} (most to least disadvantaged)"
            if equity_axis_label
            else "Equity band (most to least disadvantaged)"
        )
        axis.set_ylabel("% of band's own population")
        axis.legend()

        if title == "default":
            title = "Population impact by equity band"
        if title:
            axis.set_title(title)

        return fig, axis
