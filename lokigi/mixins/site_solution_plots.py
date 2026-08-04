import pandas as pd
import geopandas
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from warnings import warn
import numpy as np
import math
import plotly.express as px
from lokigi.utils import _safe_evaluate
from typing import Literal, Optional
from lokigi.utils import (
    _wrap_label,
    _select_solution,
    _get_ordinal_suffix,
    _sort_solutions_by_metric,
    _order_bins_most_to_least_disadvantaged,
    _unreachable_metrics_fragment,
    _min_max_normalize,
)
from lokigi.plot_utils import _add_plot_caption, _attach_deferred_fit_bounds
import warnings
from requests.exceptions import RequestException
import textwrap
from adjustText import adjust_text

#: Fill colour for a region with no feasible journey to any selected site
#: (`add_travel_matrix(allow_missing=True)`) on a choropleth -- distinct
#: from every sequential/categorical colormap this module uses by default,
#: so it reads as "excluded", not as an extreme value on the scale.
_UNREACHABLE_REGION_COLOUR = "lightgrey"


##################################
# MARK: Non-map plots
##################################
class NonMapPlotsMixin:
    def plot_n_best_combinations_bar(
        self,
        y_axis="weighted_average",
        n_best=10,
        interactive=True,
        sort_by=None,
        title="default",
        x_axis_label="default",
        y_axis_label="default",
        plot_names=True,
        line_breaks_x_axis_label=True,
        interactive_width=800,
        interactive_height=600,
        static_width=12,
        static_height=6,
    ):
        """
        Plot a bar chart of the top-performing site combinations.

        This method visualises the performance of the top ``n_best`` solutions
        using a bar chart, where each bar represents a solution and its value
        corresponds to the selected metric (e.g., weighted average travel time).

        Parameters
        ----------
        y_axis : str, default="weighted_average"
            Column name representing the metric to plot on the y-axis.
            This should correspond to a column in ``solution_df``.
        n_best : int, optional, default=10
            Number of top solutions to include in the plot. If None, all
            solutions are included.
        interactive : bool, default=True
            If True, generates an interactive Plotly bar chart. If False,
            generates a static Matplotlib bar chart.
        sort_by : str, optional
            Column name used to rank solutions. If provided, the best
            solutions by this metric are selected -- highest-first for
            coverage proportions, lowest-first for travel costs.
            If None, the existing order of ``solution_df`` is used.
        title : str or None, default="default"
            Title for the plot. If "default", an automatic title is generated
            based on the ranking metric or objective. If None, no title is set.
            Otherwise, the provided string is used as the title.
        plot_names: bool, default=True
            If True, plots site names. If false, plots canonical site indices.

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The generated bar chart. Returns a Plotly Figure if
            ``interactive=True``, otherwise a Matplotlib Figure.

        Notes
        -----
        The x-axis represents the selected site combinations (``site_indices``),
        which are converted to strings for display.

        When ``interactive=True``, Plotly Express is used to generate the chart.
        When ``interactive=False``, a Matplotlib figure is created and returned.

        The method assumes that lower values of the ranking metric correspond
        to better solutions when ``sort_by`` is specified.
        """

        if sort_by is not None:
            df = _sort_solutions_by_metric(self.solution_df, sort_by)
        else:
            df = self.solution_df
        if n_best is not None:
            df = df.head(n_best)
        if line_breaks_x_axis_label:
            if interactive:
                line_break = "<br>"
            else:
                line_break = "\n"
        else:
            line_break = " "

        if plot_names:
            df["site_names"] = df["site_names"].apply(
                lambda x: f",{line_break}".join([i for i in x])
            )

            x_axis = "site_names"
        else:
            df["site_indices"] = df["site_indices"].apply(
                lambda x: f",{line_break}".join([str(int(i)) for i in x])
            )

            x_axis = "site_indices"

        if interactive:
            df = df.copy()

            if sort_by is not None:
                title = f"Top {n_best} Solutions by {sort_by.replace('_', ' ').title()}"
            else:
                title = f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()}"

            fig = px.bar(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                labels={
                    x_axis: x_axis.replace("_", " ").title()
                    if x_axis_label == "default"
                    else x_axis_label,
                    y_axis: y_axis.replace("_", " ").title()
                    if y_axis_label == "default"
                    else y_axis_label,
                },
            )

            fig.update_layout(
                width=interactive_width,  # simple scaling from "inches-like" input
                height=interactive_height,
            )
        else:
            fig, ax = plt.subplots(figsize=(static_width, static_height))

            ax.bar(
                df["site_names"] if plot_names else df["site_indices"],
                df[y_axis],
            )

            if title == "default":
                if sort_by is not None:
                    ax.set_title(
                        f"Top {n_best} Solutions by {sort_by.replace('_', ' ').title()}"
                    )
                else:
                    ax.set_title(
                        f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()})"
                    )
            elif title is None:
                pass
            else:
                ax.set_title(title)

            x_label = "Site Names" if plot_names else "Site Names"
            y_label = f"{y_axis.replace('_', ' ').title()}"
            ax.set_xlabel(x_label if x_axis_label == "default" else x_axis_label)
            ax.set_ylabel(y_label if y_axis_label == "default" else y_axis_label)
            # plt.xticks(rotation=45)
            plt.tight_layout()
            plt.close(fig)

        return fig

    def plot_site_allocation_summary(
        self,
        by="demand",
        metric="proportion",
        sort_by=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
        demand=None,
        interactive=True,
        sort=True,
        cmap="Set2",
        site_color_map=None,
        title="default",
        x_axis_label="default",
        y_axis_label="default",
        interactive_width=800,
        interactive_height=600,
        static_width=10,
        static_height=6,
    ):
        """
        Bar chart of `site_allocation_summary()` for one chosen solution.

        Parameters
        ----------
        by, sort_by, solution_rank, site_names, site_indices, matrix, demand
            Passed straight through to `site_allocation_summary()`.
        metric : {"proportion", "allocated_demand", "n_regions", "average_travel_cost"}, default "proportion"
            Which `site_allocation_summary()` column to plot. "proportion"
            shows the share of demand (or of regions) closest to each site.
            "allocated_demand" shows the raw headcount allocated to each
            site instead of its share -- requires demand data (see Raises);
            use "n_regions" to count regions instead when no demand is
            registered. Note `by=` has no effect on "allocated_demand" or
            "n_regions": both are already an absolute count, not a share,
            so there is nothing for "regions" vs "demand" to switch between
            (that distinction only applies to "proportion" and
            "average_travel_cost"). "average_travel_cost" shows the average
            travel cost incurred by each site's group instead -- e.g. to
            see how much further people would have to travel if a site
            were closed. This is the comparison inspired by Gill Baker's
            work using average travel distance per patient to show that
            centralising services would roughly double typical travel
            distance (see `site_allocation_summary` for the full story).
        interactive : bool, default=True
            If True, generates an interactive Plotly bar chart. If False,
            generates a static Matplotlib bar chart.
        sort : bool, default=True
            If True, bars are ordered ascending by `metric`, so the
            smallest-catchment (or shortest-travel) site -- usually the one
            being investigated -- is at the top. If False, sites keep
            canonical site-index order, useful when lining this chart up
            against another figure built in that order.
        cmap : str, default="Set2"
            Colormap for sites closest to at least one region, matching
            `plot_best_combination(plot_site_allocation=True)`. Sites
            closest to no region are coloured grey rather than drawn from
            this colormap, since they colour no regions on that map
            either.
        site_color_map : dict, optional
            Explicit ``{site_name: color}`` mapping, e.g. one already built
            for a companion map, so the two figures share identical
            colours. Sites not present in this mapping are coloured grey.

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The generated bar chart. Returns a Plotly Figure if
            ``interactive=True``, otherwise a Matplotlib Figure.

        Notes
        -----
        With `metric="proportion"`, a zero-allocation site's bar has zero
        length and would otherwise be invisible, so every bar is always
        labelled with its percentage value -- that label is how a 0% site
        stays visible on the chart at all.

        With `metric="average_travel_cost"`, a zero-allocation site has no
        travel cost to average (`NaN` in `site_allocation_summary()`), so
        its bar is drawn at zero length but labelled "N/A" rather than a
        number -- a "0" label there would misleadingly read as "instant to
        reach" rather than "not applicable".
        """
        valid_metrics = ("proportion", "allocated_demand", "n_regions", "average_travel_cost")
        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got {metric!r}."
            )

        if metric == "allocated_demand":
            demand_col = getattr(self.site_problem, "_demand_data_demand_col", None)
            if demand is None and demand_col is None:
                raise ValueError(
                    "metric='allocated_demand' requires demand data. No "
                    "demand column is registered on this problem -- call "
                    "add_demand(), or use metric='n_regions' to count "
                    "regions instead."
                )

        _, _, _, unit, suffix = self._resolve_travel_columns(matrix)

        selected_solution = _select_solution(
            self.solution_df,
            sort_by=sort_by,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
        )
        regions_unreachable = selected_solution[f"regions_unreachable{suffix}"].values[
            0
        ]

        summary = self.site_allocation_summary(
            by=by,
            sort_by=sort_by,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
            matrix=matrix,
            demand=demand,
        ).reset_index()

        if sort:
            summary = summary.sort_values(metric, ascending=True).reset_index(drop=True)

        if site_color_map is None:
            allocated_sites = summary.loc[summary["n_regions"] > 0, "site"]
            site_color_map = self._site_allocation_color_map(allocated_sites, cmap=cmap)

        # Convert to hex up front so both the Plotly and Matplotlib branches
        # can share one representation -- to_hex() accepts both the
        # matplotlib RGBA tuples produced by `_site_allocation_color_map`
        # and named colors like "lightgrey", and Plotly is happy with hex.
        colors = [
            to_hex(site_color_map.get(site, "lightgrey")) for site in summary["site"]
        ]

        by_label = "demand" if by == "demand" else "regions"

        if metric == "proportion":
            # NaN never occurs for `proportion` -- fillna is a no-op here,
            # kept only so `plot_values` has one shared name below.
            plot_values = summary["proportion"].fillna(0)
            bar_text = [f"{v:.1%}" for v in summary["proportion"]]
            if title == "default":
                title = f"Share of {by_label} closest to each site"
                unreachable_fragment = _unreachable_metrics_fragment(
                    regions_unreachable
                )
                if unreachable_fragment:
                    title += f" ({unreachable_fragment}, excluded)"
            x_label = (
                f"Proportion of {by_label}" if x_axis_label == "default" else x_axis_label
            )
        elif metric == "allocated_demand":
            # Never NaN -- a zero-allocation site is a real 0 count.
            plot_values = summary["allocated_demand"]
            bar_text = [f"{v:,.0f}" for v in summary["allocated_demand"]]
            if title == "default":
                title = "Total demand allocated to each site"
                unreachable_fragment = _unreachable_metrics_fragment(
                    regions_unreachable
                )
                if unreachable_fragment:
                    title += f" ({unreachable_fragment}, excluded)"
            x_label = "Demand allocated" if x_axis_label == "default" else x_axis_label
        elif metric == "n_regions":
            plot_values = summary["n_regions"]
            bar_text = [f"{v:,.0f}" for v in summary["n_regions"]]
            if title == "default":
                title = "Number of regions closest to each site"
                unreachable_fragment = _unreachable_metrics_fragment(
                    regions_unreachable
                )
                if unreachable_fragment:
                    title += f" ({unreachable_fragment}, excluded)"
            x_label = "Number of regions" if x_axis_label == "default" else x_axis_label
        else:
            # A zero-allocation site is NaN in average_travel_cost (there is
            # nothing to average), which most plotting backends render as a
            # gap rather than a visible zero-length bar. Fill with 0 for the
            # bar's actual length, but keep the label text "N/A" -- see the
            # Notes above for why 0 would be a misleading label to show.
            plot_values = summary["average_travel_cost"].fillna(0)
            bar_text = [
                "N/A" if pd.isna(v) else f"{v:.1f} {unit}"
                for v in summary["average_travel_cost"]
            ]
            if title == "default":
                title = f"Average {by_label}-weighted travel cost by closest site"
                unreachable_fragment = _unreachable_metrics_fragment(
                    regions_unreachable
                )
                if unreachable_fragment:
                    title += f" ({unreachable_fragment}, excluded)"
            x_label = (
                f"Average travel cost ({unit})"
                if x_axis_label == "default"
                else x_axis_label
            )
        y_label = "Site" if y_axis_label == "default" else y_axis_label

        if interactive:
            fig = px.bar(
                summary.assign(_plot_value=plot_values),
                x="_plot_value",
                y="site",
                orientation="h",
                title=title,
                labels={"_plot_value": x_label, "site": y_label},
            )
            fig.update_traces(
                marker_color=colors,
                text=bar_text,
                textposition="outside",
            )
            fig.update_layout(
                width=interactive_width,
                height=interactive_height,
                xaxis_tickformat=".0%" if metric == "proportion" else None,
            )
            # Plotly draws horizontal bars bottom-up by default, so without
            # this the first (smallest, when sort=True) row ends up at the
            # bottom instead of the top.
            fig.update_yaxes(autorange="reversed")
        else:
            fig, ax = plt.subplots(figsize=(static_width, static_height))
            bars = ax.barh(summary["site"], plot_values, color=colors)
            ax.bar_label(bars, labels=bar_text)
            if title is not None:
                ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if metric == "proportion":
                ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            # matplotlib's barh also plots bottom-up by default -- invert so
            # the top row matches the top of the sorted (or canonically
            # ordered) DataFrame, same reasoning as the Plotly branch above.
            ax.invert_yaxis()
            plt.tight_layout()
            plt.close(fig)

        return fig

    def plot_site_capacity_summary(
        self,
        metric="allocated_utilisation_ratio",
        capacity_df=None,
        capacity_col=None,
        demand_to_capacity_rate=1.0,
        sort_by=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
        demand=None,
        interactive=True,
        sort=True,
        under_capacity_colour="#4C72B0",
        over_capacity_colour="#C44E52",
        missing_colour="lightgrey",
        show_reference_line=True,
        title="default",
        x_axis_label="default",
        y_axis_label="default",
        caption=None,
        interactive_width=800,
        interactive_height=600,
        static_width=10,
        static_height=6,
        ax=None,
    ):
        """
        Bar chart of `site_capacity_summary()` for one chosen solution.

        Parameters
        ----------
        metric : {"allocated_utilisation_ratio", "incremental_headroom_ratio"}, default "allocated_utilisation_ratio"
            Which `site_capacity_summary()` ratio to plot.
            "allocated_utilisation_ratio" assumes the allocated demand
            *replaces* today's activity entirely -- the whole-network
            reallocation `solve()` actually models.
            "incremental_headroom_ratio" assumes it lands *on top of*
            today's activity instead, and raises if no baseline load data
            (`current_load_col`/`utilisation_col`) was registered. These
            two answer genuinely different questions -- see
            `site_capacity_summary`'s Notes before picking one.
        capacity_df : pandas.DataFrame, optional
            A precomputed result from `site_capacity_summary()`. If given,
            `capacity_col`, `demand_to_capacity_rate`, and the solution
            selection arguments below are ignored entirely -- the frame is
            used exactly as supplied, matching
            `plot_accessibility(region_frame=...)`.
        capacity_col, demand_to_capacity_rate, sort_by, solution_rank,
        site_names, site_indices, matrix, demand
            Passed straight through to `site_capacity_summary()` when
            `capacity_df` is not supplied.
        interactive : bool, default True
            If True, generates an interactive Plotly bar chart. If False,
            generates a static Matplotlib bar chart.
        sort : bool, default True
            If True, bars are ordered ascending by `metric`. NaN values
            (no capacity registered for that site) sort last. If False,
            sites keep canonical site-index order.
        under_capacity_colour, over_capacity_colour : str
            Colours for bars at or below, and above, the reference value
            (`1.0` for `allocated_utilisation_ratio`; `0.0` for
            `incremental_headroom_ratio`, where negative means already
            over capacity today). Colour here is deliberately semantic
            (over/under), not categorical by site as in
            `plot_site_allocation_summary` -- the finding this chart exists
            to show is whether a site fits, not which site is which, so
            `site_color_map` is not offered.
        missing_colour : str, default "lightgrey"
            Colour for a site with no capacity registered (`NaN` ratio).
        show_reference_line : bool, default True
            If True, draws a vertical line at the reference value (see
            `under_capacity_colour` above) labelled "at capacity".
        caption : str or None, default None
            `None` prints a short "how to read this" explanation of the
            reference line and, for `metric="incremental_headroom_ratio"`,
            that this treats allocated demand as landing on top of today's
            activity. Pass `""` to suppress it, or a custom string to
            replace it. Static branch only (matches `_add_plot_caption`'s
            existing convention on `plot_accessibility()` etc.).
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw the static chart onto instead of
            creating a new figure, e.g. to embed this as one panel of a
            larger layout. Ignored if `interactive=True`. When given, the
            caller owns the figure's lifecycle: unlike the default
            (self-contained) case, this method does not call
            `plt.tight_layout()` or close the figure afterwards. `caption`
            (if not suppressed) is still placed relative to the *whole*
            figure `ax` belongs to, not just this panel -- pass
            `caption=""` if that would land awkwardly among the other
            panels of a shared figure.

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If `metric` is invalid, or if `metric="incremental_headroom_ratio"`
            but no baseline load data was registered.

        Notes
        -----
        A site with `NaN` capacity is drawn as a zero-length bar labelled
        "N/A", coloured `missing_colour` -- not `0`, which would misleadingly
        read as "empty". A site with an infinite ratio (zero registered
        capacity but nonzero allocated load) is drawn at a finite length
        (just past the largest finite bar) so the axis stays usable, but
        labelled with the infinity symbol -- the bar's length is a drawing
        convenience only, not a real value.
        """
        valid_metrics = ("allocated_utilisation_ratio", "incremental_headroom_ratio")
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got {metric!r}.")

        _, _, _, _, suffix = self._resolve_travel_columns(matrix)
        selected_solution = _select_solution(
            self.solution_df,
            sort_by=sort_by,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
        )
        regions_unreachable = selected_solution[f"regions_unreachable{suffix}"].values[
            0
        ]

        if capacity_df is None:
            capacity_df = self.site_capacity_summary(
                capacity_col=capacity_col,
                demand_to_capacity_rate=demand_to_capacity_rate,
                sort_by=sort_by,
                solution_rank=solution_rank,
                site_names=site_names,
                site_indices=site_indices,
                matrix=matrix,
                demand=demand,
            )

        if metric == "incremental_headroom_ratio" and metric not in capacity_df.columns:
            raise ValueError(
                "metric='incremental_headroom_ratio' requires baseline load "
                "data. Call add_sites(..., current_load_col=...) or "
                "add_sites(..., utilisation_col=...) so headroom can be "
                "derived."
            )

        summary = capacity_df.reset_index()

        if sort:
            summary = summary.sort_values(metric, ascending=True, na_position="last").reset_index(
                drop=True
            )

        reference_value = 1.0 if metric == "allocated_utilisation_ratio" else 0.0
        raw_values = summary[metric]
        finite_values = raw_values[np.isfinite(raw_values)]
        max_finite = finite_values.max() if len(finite_values) else None

        colors = []
        bar_text = []
        plot_values = []
        for v in raw_values:
            if pd.isna(v):
                colors.append(missing_colour)
                bar_text.append("N/A")
                plot_values.append(0.0)
            elif np.isinf(v):
                colors.append(over_capacity_colour)
                bar_text.append("∞ (zero capacity)")
                sanitized = max_finite * 1.05 if max_finite and max_finite > 0 else 1.5
                plot_values.append(sanitized)
            else:
                colors.append(
                    over_capacity_colour if v > reference_value else under_capacity_colour
                )
                bar_text.append(f"{v:.0%}")
                plot_values.append(v)

        rate_fragment = (
            f" (at {demand_to_capacity_rate:g} capacity units per unit of demand)"
            if demand_to_capacity_rate != 1.0
            else ""
        )
        metric_label = (
            "Allocated utilisation ratio"
            if metric == "allocated_utilisation_ratio"
            else "Incremental headroom ratio"
        )
        if title == "default":
            title = "Allocated demand vs capacity" + rate_fragment
            unreachable_fragment = _unreachable_metrics_fragment(regions_unreachable)
            if unreachable_fragment:
                title += f" ({unreachable_fragment}, excluded)"
        x_label = metric_label if x_axis_label == "default" else x_axis_label
        y_label = "Site" if y_axis_label == "default" else y_axis_label

        if interactive:
            fig = px.bar(
                summary.assign(_plot_value=plot_values),
                x="_plot_value",
                y="site",
                orientation="h",
                title=title,
                labels={"_plot_value": x_label, "site": y_label},
            )
            fig.update_traces(marker_color=colors, text=bar_text, textposition="outside")
            fig.update_layout(width=interactive_width, height=interactive_height)
            fig.update_yaxes(autorange="reversed")
            if show_reference_line:
                fig.add_vline(
                    x=reference_value,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text="at capacity",
                )
        else:
            owns_figure = ax is None
            if owns_figure:
                fig, ax = plt.subplots(figsize=(static_width, static_height))
            else:
                fig = ax.get_figure()

            bars = ax.barh(summary["site"], plot_values, color=colors)
            ax.bar_label(bars, labels=bar_text)
            if title is not None:
                ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if show_reference_line:
                ax.axvline(reference_value, linestyle="--", color="grey", linewidth=1)
                ax.text(
                    reference_value,
                    -0.6,
                    "at capacity",
                    color="grey",
                    fontsize=8,
                    ha="center",
                )
            ax.invert_yaxis()

            default_caption = (
                "The dashed line marks capacity exactly met. Bars past it are "
                "over capacity."
            )
            if metric == "incremental_headroom_ratio":
                default_caption += (
                    " This treats allocated demand as landing on top of "
                    "today's activity, not replacing it."
                )
            _add_plot_caption(fig, caption, default_caption)

            # Only manage the figure's lifecycle when this method created it
            # itself -- a caller-supplied ax means the figure is one panel
            # of a larger layout still being assembled, and tight_layout()/
            # closing it here would fight the caller's own layout and
            # display handling.
            if owns_figure:
                plt.tight_layout()
                plt.close(fig)

        return fig


##################################
# MARK: Maps
##################################
class MapsMixin:
    def _site_allocation_color_map(self, allocated_sites, cmap="Set2"):
        """
        Build a ``{site: color}`` mapping for `allocated_sites`, ordered by
        canonical site index (not alphabetically) so colours stay stable
        across figures that show different subsets of sites -- e.g. the
        allocation map built here and the allocation bar chart in
        `plot_site_allocation_summary` assign the same site the same
        colour as long as it's allocated in both.
        """
        master_site_order = self.site_problem.candidate_sites[
            self.site_problem._candidate_sites_candidate_id_col
        ].tolist()
        allocated = set(allocated_sites)
        ordered = [site for site in master_site_order if site in allocated]

        cmap_obj = plt.get_cmap(cmap)
        return {site: cmap_obj(i % cmap_obj.N) for i, site in enumerate(ordered)}

    def plot_allocated_utilisation(
        self,
        capacity_df=None,
        capacity_col=None,
        demand_to_capacity_rate=1.0,
        sort_by=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        matrix=None,
        demand=None,
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
        Map each selected site's `allocated_utilisation_ratio` from
        `site_capacity_summary()`.

        Site markers are coloured (and, on a static map, sized) by how
        full each site is under this solution's allocation, so pressure
        points are visible geographically rather than only in a table or
        bar chart. Unlike `plot_site_utilisation()` (today's real-world
        baseline, independent of any solve), this is solve-derived: only
        sites selected by the chosen solution are drawn, matching
        `site_capacity_summary()`'s own scope.

        Parameters
        ----------
        capacity_df : pandas.DataFrame, optional
            A precomputed result from `site_capacity_summary()`. If given,
            `capacity_col`/`demand_to_capacity_rate`/the selection
            arguments below are ignored, matching
            `plot_site_utilisation(utilisation_df=...)`.
        capacity_col, demand_to_capacity_rate, sort_by, solution_rank,
        site_names, site_indices, matrix, demand
            Passed straight through to `site_capacity_summary()` when
            `capacity_df` is not supplied.
        interactive : bool, default False
            If True, returns an interactive Folium map via `.explore()`.
            Otherwise returns a static matplotlib Axes.
        cmap : str, default "RdYlGn_r"
            Colormap for site markers. The reversed variant, matching
            `plot_site_utilisation`'s own default and for the same reason:
            a high ratio here is bad (near/at/over capacity), so red must
            map to the high end.
        missing_site_colour : str, default "lightgrey"
            Colour (and static marker size, at the smallest of
            `marker_size_range`) for a selected site with no capacity
            registered (`NaN` ratio).
        marker_size_range : tuple of (float, float), default (40, 220)
            Smallest and largest static marker size, linearly scaled by
            `allocated_utilisation_ratio`. Ignored on interactive maps.
        show_labels : bool, default False
            If True, adds text labels for each site (see `plot_sites`).
        add_basemap : bool, default True
            If True, adds a background web map.
        title : str, optional
            Axes title. Ignored on interactive maps.
        caption : str or None, default None
            `None` prints a short "how to read this" explanation; pass
            `""` to suppress it or a custom string to replace it. Static
            branch only, via `_add_plot_caption`.
        ax : matplotlib.axes.Axes, optional
        figsize : tuple, optional
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
            was never given a GeoDataFrame or lat/long columns).

        Notes
        -----
        Only sites selected by the chosen solution are drawn -- an
        unselected candidate site would otherwise have to be shown grey,
        but grey already means "no capacity registered" on this map, and
        conflating "not chosen" with "chosen but not measured" would be
        actively misleading. `plot_best_combination()` is the map for
        "which sites were chosen"; this one assumes that question is
        already answered.

        A site with an infinite ratio (zero registered capacity, nonzero
        allocated load) would otherwise flatten every other marker to a
        uniform size/colour (an infinite range swallows every finite
        value), so it is substituted with the largest finite ratio times
        1.05 for sizing/colouring purposes only -- the tooltip/popup still
        shows the true (infinite) value.
        """
        candidate_sites = self.site_problem.candidate_sites
        if not isinstance(candidate_sites, geopandas.GeoDataFrame):
            raise ValueError(
                "plot_allocated_utilisation() requires real site geometry -- "
                "add_sites() must have been given a GeoDataFrame or "
                "lat/long columns, not a bare site list inferred from the "
                "travel matrix's column names."
            )

        if capacity_df is None:
            capacity_df = self.site_capacity_summary(
                capacity_col=capacity_col,
                demand_to_capacity_rate=demand_to_capacity_rate,
                sort_by=sort_by,
                solution_rank=solution_rank,
                site_names=site_names,
                site_indices=site_indices,
                matrix=matrix,
                demand=demand,
            )

        candidate_id_col = self.site_problem._candidate_sites_candidate_id_col
        site_gdf = candidate_sites.merge(
            capacity_df.reset_index(),
            left_on=candidate_id_col,
            right_on="site",
            suffixes=("", "_y"),
        )
        site_gdf = site_gdf.drop(site_gdf.filter(regex="_y$").columns, axis=1)

        raw_ratio = site_gdf["allocated_utilisation_ratio"]
        finite_ratio = raw_ratio[np.isfinite(raw_ratio)]
        max_finite = finite_ratio.max() if len(finite_ratio) else None
        sanitized_fill = max_finite * 1.05 if max_finite and max_finite > 0 else 1.5
        display_ratio = raw_ratio.replace([np.inf], sanitized_fill)
        site_gdf["_display_ratio"] = display_ratio

        min_size, max_size = marker_size_range
        site_gdf["_marker_size"] = (
            min_size
            + _min_max_normalize(display_ratio, constant_fill=1.0) * (max_size - min_size)
        ).fillna(min_size)

        tooltip_cols = [
            c
            for c in (
                candidate_id_col,
                "allocated_demand",
                "allocated_load",
                "capacity",
                "allocated_utilisation_ratio",
                "current_load",
                "baseline_utilisation_ratio",
                "headroom",
                "incremental_headroom_ratio",
                "residual_headroom",
            )
            if c in site_gdf.columns
        ]

        if interactive:
            m = site_gdf.explore(
                column="_display_ratio",
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
            column="_display_ratio",
            cmap=cmap,
            markersize=site_gdf["_marker_size"],
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            legend=True,
            legend_kwds={"label": "Allocated utilisation ratio", "shrink": 0.6},
            missing_kwds=dict(color=missing_site_colour, label="No capacity data"),
            **kwargs,
        )

        if show_labels:
            texts = []
            for x, y, label in zip(
                site_gdf.geometry.x,
                site_gdf.geometry.y,
                site_gdf[candidate_id_col],
            ):
                wrapped_label = textwrap.fill(label, 15).title()
                texts.append(ax.text(x, y, wrapped_label))
            adjust_text(texts, force_explode=(0.05, 0.05), ax=ax)

        # geopandas' missing_kwds `label` only surfaces in a discrete
        # `scheme=` legend, not the continuous colorbar used here, so the
        # grey "no capacity data" markers would otherwise have no legend
        # entry explaining them.
        if site_gdf["_display_ratio"].isna().any():
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
                        label="No capacity data",
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
            "Markers are coloured (and sized) by each site's allocated "
            "demand relative to its capacity under this solution -- not "
            "today's real-world utilisation (see plot_site_utilisation() "
            "for that). Only sites selected by this solution are shown."
        )
        _add_plot_caption(fig, caption, default_caption)

        return ax

    def _plot_single_solution_map(
        self,
        ax,
        solution,
        show_all_locations=True,
        label_all_locations=False,
        plot_site_allocation=False,
        plot_regions_not_meeting_threshold=False,
        cmap=None,
        chosen_site_colour="black",
        unchosen_site_colour="white",
        unchosen_site_opacity=0.7,
        annotation_size=6,
        label_wrap_width=40,
        global_vmin=None,
        global_vmax=None,
        site_color_map=None,
        discrete_cmap=None,
        colors=None,
        add_legend=True,
        legend_kwargs=None,
        matrix=None,
    ):
        """
        Internal helper to plot a single solution map onto a given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot onto
        solution : pd.Series or pd.DataFrame
            Single row containing solution data
        global_vmin, global_vmax : float, optional
            For consistent color scaling across multiple plots
        site_color_map : dict, optional
            Mapping from site names to colors for consistent categorical coloring
        discrete_cmap : ListedColormap, optional
            Discrete colormap for threshold plots
        colors : list, optional
            List of colors for threshold legend
        add_legend : bool, default=True
            Whether to add a legend to this axes
        legend_kwargs : dict, optional
            Keyword arguments for legend placement
        annotation_size: int, optional
            Font size of site name annotations on the map
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. If None (default), the primary
            travel matrix's columns and unit are used.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The modified axes object
        """
        cost_col, site_col, within_col, unit, _ = self._resolve_travel_columns(matrix)

        # Merge geometry with solution data
        solution_df = (
            solution["problem_df"].values[0]
            if hasattr(solution["problem_df"], "values")
            else solution["problem_df"][0]
        )

        nearest_site_travel_gdf = pd.merge(
            self.site_problem.region_geometry_layer,
            solution_df,
            left_on=self.site_problem._region_geometry_layer_common_col,
            right_on=self.site_problem._demand_data_id_col,
            suffixes=("", "_y"),
        )

        nearest_site_travel_gdf = nearest_site_travel_gdf.drop(
            nearest_site_travel_gdf.filter(regex="_y$").columns, axis=1
        )

        # Plot regions based on selected mode
        if plot_site_allocation:
            if site_color_map is not None:
                # Use consistent global color mapping. `.map()` leaves an
                # unreachable region's colour as NaN (no key in
                # site_color_map for a missing selected_site) -- geopandas'
                # `color=` (as opposed to `column=`) doesn't support
                # `missing_kwds`, so it would otherwise draw with `alpha`
                # applied to nothing, leaving a silent hole in the map
                # rather than a visibly-explained one.
                colors_mapped = nearest_site_travel_gdf[site_col].map(
                    site_color_map
                ).fillna(_UNREACHABLE_REGION_COLOUR)
                ax = nearest_site_travel_gdf.plot(
                    color=colors_mapped,
                    legend=False,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax,
                )
            else:
                # Single plot with its own legend
                ax = nearest_site_travel_gdf.plot(
                    site_col,
                    legend=add_legend,
                    cmap=cmap,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax,
                    legend_kwds=legend_kwargs if legend_kwargs and add_legend else {},
                    missing_kwds=dict(color=_UNREACHABLE_REGION_COLOUR),
                )

        elif plot_regions_not_meeting_threshold:
            ax = nearest_site_travel_gdf.plot(
                within_col,
                legend=False,
                cmap=discrete_cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
                vmin=0,
                vmax=1,
            )

            # Add custom legend if requested and this is a single plot
            if add_legend and colors is not None:
                threshold_val = self._resolve_coverage_threshold(
                    matrix, solution["coverage_threshold"].values[0]
                )
                patches = [
                    mpatches.Patch(
                        color=colors[0],
                        label=_wrap_label(
                            f"Further than {threshold_val} {unit}",
                            width=label_wrap_width,
                        ),
                    ),
                    mpatches.Patch(
                        color=colors[1],
                        label=_wrap_label(
                            f"Within {threshold_val} {unit}",
                            width=label_wrap_width,
                        ),
                    ),
                ]
                legend_loc = (
                    legend_kwargs.get("loc", "upper right")
                    if legend_kwargs
                    else "upper right"
                )
                ax.legend(handles=patches, title="Coverage Status", loc=legend_loc)

        else:
            # Plot min_cost (or the secondary matrix's own min cost). NaN
            # (no feasible journey, see add_travel_matrix(allow_missing=
            # True)) is drawn as a distinct flat colour via missing_kwds,
            # rather than left unfilled -- geopandas draws nothing at all
            # for a missing value by default, an invisible hole in the map
            # indistinguishable from "outside the study area".
            unit_parenthetical = f" ({unit})" if unit else ""
            ax = nearest_site_travel_gdf.plot(
                cost_col,
                legend=add_legend,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
                vmin=global_vmin,
                vmax=global_vmax,
                legend_kwds={
                    "label": f"Travel time to nearest site{unit_parenthetical}"
                }
                if add_legend
                else {},
                missing_kwds=dict(color=_UNREACHABLE_REGION_COLOUR),
            )

        # Plot candidate sites if applicable. Deliberately not
        # `_candidate_sites_type == "geopandas"`: that attribute records
        # add_sites()'s *input* format, not whether candidate_sites ended up
        # with real point geometry. Tabular lat/long input is converted to a
        # GeoDataFrame internally but still leaves _candidate_sites_type ==
        # "pandas", so that check silently skipped site markers for the most
        # common add_sites() usage pattern.
        if isinstance(self.site_problem.candidate_sites, geopandas.GeoDataFrame):
            selected_site_names = (
                solution.site_names.iloc[0]
                if hasattr(solution.site_names, "iloc")
                else solution.site_names[0]
            )

            selected_sites = self.site_problem.candidate_sites[
                self.site_problem.candidate_sites[
                    self.site_problem._candidate_sites_candidate_id_col
                ].isin(selected_site_names)
            ]

            # Determine if there are required sites
            required_values = {"y", "yes", "true", "t", "required", "1"}
            has_required_sites = False
            required_sites = None

            if self.site_problem._candidate_sites_required_sites_col is not None:
                required_mask = (
                    self.site_problem.candidate_sites[
                        self.site_problem._candidate_sites_required_sites_col
                    ]
                    .astype(str)
                    .str.lower()
                    .isin(required_values)
                )
                has_required_sites = required_mask.any()
                if has_required_sites:
                    required_sites = self.site_problem.candidate_sites[required_mask]

            if show_all_locations:
                # Plot non-selected, non-required sites
                if has_required_sites:
                    non_selected_non_required = self.site_problem.candidate_sites[
                        ~self.site_problem.candidate_sites[
                            self.site_problem._candidate_sites_candidate_id_col
                        ].isin(selected_site_names)
                        & ~required_mask
                    ]
                    non_selected_non_required.plot(
                        ax=ax,
                        color=unchosen_site_colour,
                        markersize=30,
                        alpha=unchosen_site_opacity,
                    )
                else:
                    self.site_problem.candidate_sites[
                        ~self.site_problem.candidate_sites[
                            self.site_problem._candidate_sites_candidate_id_col
                        ].isin(selected_site_names)
                    ].plot(
                        ax=ax,
                        color=unchosen_site_colour,
                        markersize=30,
                        alpha=unchosen_site_opacity,
                    )

            # Plot required sites with different marker (triangle)
            if has_required_sites:
                required_sites.plot(
                    ax=ax,
                    color=chosen_site_colour,
                    markersize=60,
                    marker="^",  # Triangle marker
                )

            # Plot selected sites (non-required with circle marker)
            if has_required_sites:
                selected_non_required = selected_sites[
                    ~selected_sites[
                        self.site_problem._candidate_sites_required_sites_col
                    ]
                    .astype(str)
                    .str.lower()
                    .isin(required_values)
                ]
                selected_non_required.plot(
                    ax=ax, color=chosen_site_colour, markersize=60
                )
            else:
                selected_sites.plot(ax=ax, color=chosen_site_colour, markersize=60)
                has_required_sites = False

            # Add labels for selected sites
            for x, y, label in zip(
                selected_sites.geometry.x,
                selected_sites.geometry.y,
                selected_sites[self.site_problem._candidate_sites_candidate_id_col],
            ):
                ax.annotate(
                    _wrap_label(label, width=label_wrap_width),
                    xy=(x, y),
                    xytext=(10, 3),
                    textcoords="offset points",
                    bbox=dict(facecolor="white"),
                    size=annotation_size,
                )

            # Optionally add labels for all unselected sites
            if show_all_locations and label_all_locations:
                unselected_sites = self.site_problem.candidate_sites[
                    ~self.site_problem.candidate_sites[
                        self.site_problem._candidate_sites_candidate_id_col
                    ].isin(selected_site_names)
                ]
                for x, y, label in zip(
                    unselected_sites.geometry.x,
                    unselected_sites.geometry.y,
                    unselected_sites[
                        self.site_problem._candidate_sites_candidate_id_col
                    ],
                ):
                    ax.annotate(
                        _wrap_label(label, width=label_wrap_width),
                        xy=(x, y),
                        xytext=(10, 3),
                        textcoords="offset points",
                        bbox=dict(facecolor="white", alpha=0.6),  # More transparent
                        size=annotation_size * 0.8,  # Smaller font
                        style="italic",  # Italic to distinguish
                    )
        else:
            has_required_sites = False

        # Add basemap
        try:
            cx.add_basemap(ax, crs=nearest_site_travel_gdf.crs.to_string(), timeout=30)
        except RequestException as e:
            warnings.warn(
                f"Unable to download background map tiles ({type(e).__name__}). "
                "Continuing without a basemap.",
                stacklevel=2,
            )
        ax.axis("off")

        return ax, has_required_sites

    def plot_best_combination(
        self,
        ax=None,
        sort_by=None,
        solution_rank=1,
        site_names=None,
        site_indices=None,
        title="default",
        show_all_locations=True,
        label_all_locations=False,
        plot_site_allocation=False,
        plot_regions_not_meeting_threshold=False,
        cmap=None,
        chosen_site_colour="black",
        unchosen_site_colour="white",
        unchosen_site_opacity=0.7,
        legend_loc="upper right",
        legend_bbox_to_anchor=(1.75, 0.5),
        legend_fontsize=10,
        title_fontsize=12,
        annotation_size=6,
        label_wrap_width=40,
        height=12,
        width=6,
        site_legend_loc="best",
        matrix=None,
    ):
        """
        Plot a map of the best-performing site combination.

        This method visualises the spatial performance of the best solution in
        the solution set, including travel costs, site allocations, or coverage
        relative to a threshold. Regions are coloured according to the selected
        metric, and candidate site locations can be overlaid.

        Parameters
        ----------
        sort_by : str, optional
            Column name used to rank solutions. If provided, solutions are ranked
            best-first by this column -- highest for coverage proportions, lowest
            for travel costs. Ignored if site_names or site_indices is provided.
        solution_rank : int, default=1
            Which ranked solution to plot (1 = best, 2 = second best, etc.).
            Ignored if site_names or site_indices is provided.
        site_names : list of str, optional
            Specific site names to plot. If provided, filters solution_df to the
            row containing this exact combination of sites. Overrides sort_by and
            solution_rank.
        site_indices : list of int or np.ndarray, optional
            Specific site indices to plot. If provided, filters solution_df to the
            row containing this exact combination. Overrides sort_by, solution_rank,
            and site_names.
        title : str or None, default="default"
            Title for the plot. If "default", an automatic title is generated
            based on the objective and solution metrics. If a string is provided,
            it may be evaluated using ``_safe_evaluate`` with access to the
            selected solution. If None, no title is set.
        show_all_locations : bool, default=True
            If True, plots all candidate site locations in addition to the
            selected sites.
        plot_site_allocation : bool, default=False
            If True, regions are coloured by the assigned (nearest) site.
            Overrides other colouring options.
        plot_regions_not_meeting_threshold : bool, default=False
            If True, regions are coloured based on whether they fall within
            the coverage threshold. Overrides default cost-based colouring.
        cmap : str or matplotlib colormap, optional
            Colormap used for plotting. If None, a default is chosen:
            - Sequential colormap ("Blues") for cost-based plots
            - Qualitative colormap ("Set2") for site allocation plots
        chosen_site_colour : str, default="black"
            Colour used to plot selected site locations.
        unchosen_site_colour : str, default="grey"
            Colour used to plot unselected candidate site locations.
        unchosen_site_opacity : float, default=0.5
            Opaqueness/alpha of the points for unchosen sites
        legend_loc: str, default="upper right"
            Adjust position of the legend within the main plot. Accepts standard
            matplotlib positioning strings.
        site_legend_loc: str, default="best"
            Adjust position of the legend within the main plot. Accepts standard
            matplotlib positioning strings. Only used if required sites are present
            in the problem.
        legend_bbox_to_anchor : tuple or None, default=None
            If provided, places the legend outside the plot area. Accepts a
            2-tuple (x, y) in axes coordinates. Common examples:
            - (1.05, 1) for right side, top aligned
            - (0.5, -0.1) for bottom center
            - (1.05, 0.5) for right side, center aligned
            When set, legend_loc is used as the anchor point on the legend box.
        legend_fontsize : int or float, default=10
            Font size for legend text.
        title_fontsize : int or float, default=12
            Font size for the plot title.
        label_wrap_width : int, default=40
            Maximum character width before wrapping legend labels. Set to None
            to disable wrapping.
        annotation_size: int, optional
            Font size of site name annotations on the map
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. If provided, the plot and its
            title metrics use that matrix's columns and unit instead of the
            primary travel matrix.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib Axes object containing the generated plot.

        Raises
        ------
        ValueError
            If the region geometry layer has not been initialised in the
            ``site_problem`` instance.

        Notes
        -----
        The method requires a valid ``region_geometry_layer`` to be present in
        ``site_problem``. This should be added prior to calling this method.

        The plot is generated using GeoPandas and includes a basemap via
        ``contextily.add_basemap``.

        Depending on the flags provided, regions are coloured by:
        - "min_cost" (default): travel time/distance to the nearest site
        - "selected_site": assigned site (categorical)
        - "within_threshold": whether the region meets the coverage threshold

        Site markers require `candidate_sites` to carry real point geometry.
        This holds both for sites registered via a GeoDataFrame/geojson and
        for tabular lat/long input, which `add_sites()` converts internally
        -- it does not hold if site names were inferred from the travel
        matrix's columns without ever calling `add_sites()`.

        Titles are dynamically generated based on the objective type and may
        include metrics such as weighted/unweighted averages, maximum cost,
        and coverage proportion.

        The method assumes that lower values of the ranking metric correspond
        to better solutions.
        """
        if self.site_problem.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        if sort_by is not None:
            solution = (
                _sort_solutions_by_metric(self.solution_df, sort_by)
                .head(1)
                .reset_index()
            )
        else:
            solution = self.solution_df.head(1).reset_index()

        # Solution selection logic
        solution = _select_solution(
            self.solution_df,
            sort_by=sort_by,
            solution_rank=solution_rank,
            site_names=site_names,
            site_indices=site_indices,
        )

        # Choose appropriate colourmap if not provided
        if cmap is None:
            if plot_site_allocation:
                cmap = "Set2"
            elif plot_regions_not_meeting_threshold:
                cmap = "Set1"
            else:
                cmap = "Blues"

        # Prepare discrete colormap and colors for threshold plotting
        discrete_cmap = None
        colors = None
        if plot_regions_not_meeting_threshold:
            cmap_obj = plt.get_cmap(cmap)
            if hasattr(cmap_obj, "colors"):
                colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
            else:
                colors = [cmap_obj(0.0), cmap_obj(1.0)]
            discrete_cmap = ListedColormap(colors)

        # Set up legend kwargs
        legend_kwargs = {
            "loc": legend_loc,
            "fontsize": legend_fontsize,
        }
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

        # Create figure and plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(height, width))

        ax, has_required_sites = self._plot_single_solution_map(
            ax=ax,
            solution=solution,
            show_all_locations=show_all_locations,
            label_all_locations=label_all_locations,
            plot_site_allocation=plot_site_allocation,
            plot_regions_not_meeting_threshold=plot_regions_not_meeting_threshold,
            cmap=cmap,
            chosen_site_colour=chosen_site_colour,
            unchosen_site_colour=unchosen_site_colour,
            unchosen_site_opacity=unchosen_site_opacity,
            annotation_size=annotation_size,
            label_wrap_width=label_wrap_width,
            discrete_cmap=discrete_cmap,
            colors=colors,
            add_legend=True,
            legend_kwargs=legend_kwargs,
            matrix=matrix,
        )

        if has_required_sites:
            legend_elements = []

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=chosen_site_colour,
                    markersize=10,
                    label="Required sites",
                )
            )

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=chosen_site_colour,
                    markersize=10,
                    label="Additional selected sites",
                )
            )

            ax.legend(handles=legend_elements, loc=site_legend_loc)

        # Add title
        if title is not None:
            if title == "default":
                # Determine the description prefix based on selection method
                if site_indices is not None:
                    title_prefix = (
                        f"Solution for {self.n_sites} sites (by site indices)"
                    )
                elif site_names is not None:
                    title_prefix = f"Solution for {self.n_sites} sites (by site names)"
                elif solution_rank == 1:
                    title_prefix = f"Best solution for {self.n_sites} sites"
                else:
                    rank_suffix = _get_ordinal_suffix(solution_rank)
                    title_prefix = f"{solution_rank}{rank_suffix} best solution for {self.n_sites} sites"

                _, _, _, matrix_unit, suffix = self._resolve_travel_columns(matrix)

                # Build metrics section based on objective type
                if self.objectives == "mclp":
                    threshold_val = self._resolve_coverage_threshold(
                        matrix, solution["coverage_threshold"].values[0]
                    )
                    metrics = (
                        f"Demand covered within threshold of {threshold_val} "
                        f"{matrix_unit}: "
                        f"{solution[f'proportion_within_coverage_threshold{suffix}'].values[0]:.1%} \n"
                        f"Unweighted Average: {solution[f'unweighted_average{suffix}'].values[0]:.1f} "
                        f"{matrix_unit} \n"
                        f"Maximum: {solution[f'max{suffix}'].values[0]:.1f} "
                        f"{matrix_unit}"
                    )
                elif self.objectives in ["simple_p_median", "hybrid_simple_p_median"]:
                    metrics = (
                        f"Unweighted Average: {solution[f'unweighted_average{suffix}'].values[0]:.1f} "
                        f"{matrix_unit} \n"
                        f"Maximum: {solution[f'max{suffix}'].values[0]:.1f} "
                        f"{matrix_unit}"
                    )
                else:
                    metrics = (
                        f"Weighted Average: {solution[f'weighted_average{suffix}'].values[0]:.1f} "
                        f"{matrix_unit} \n"
                        f"Maximum: {solution[f'max{suffix}'].values[0]:.1f} "
                        f"{matrix_unit}"
                    )

                unreachable_fragment = _unreachable_metrics_fragment(
                    solution[f"regions_unreachable{suffix}"].values[0]
                )
                if unreachable_fragment:
                    metrics += f" \n{unreachable_fragment}"

                # The metrics above are picked from `self.objectives`, which
                # stops describing the ranking as soon as solve(rank_on=...)
                # is used -- so name the real one rather than leaving the
                # reader to assume the objective's default was optimised.
                ranking_line = self._ranking_metric_line(solution.iloc[0])
                if ranking_line:
                    metrics += f" \n{ranking_line}"

                title = plt.title(
                    f"{title_prefix} \n{metrics}", fontsize=title_fontsize
                )
            else:
                title = plt.title(
                    _safe_evaluate(title, solution=solution), fontsize=title_fontsize
                )

        return ax

    def plot_n_best_combinations(
        self,
        n_best=10,
        sort_by=None,
        title=None,
        subplot_title="default",
        show_all_locations=True,
        label_all_locations=False,
        plot_site_allocation=False,
        plot_regions_not_meeting_threshold=False,
        cmap=None,
        chosen_site_colour="black",
        unchosen_site_colour="white",
        unchosen_site_opacity=0.5,
        annotation_size=6,
        n_cols=None,
        n_rows=None,
        fig_size_multiplier=6,
        legend_wrap_width=40,
        site_legend_bbox_to_anchor=(0.9, 0.9),
        site_legend_loc="upper right",
        matrix=None,
    ):
        """
        Plot maps for the top-performing site combinations.

        This method generates a grid of subplots visualising the spatial
        performance of the top ``n_best`` solutions. Each subplot represents
        a single solution, showing either travel costs, site allocations, or
        coverage relative to a threshold. Candidate site locations can also
        be overlaid.

        Parameters
        ----------
        n_best : int, default=10
            Number of top solutions to plot. If greater than the number of
            available solutions, all solutions are plotted.
        sort_by : str, optional
            Column name used to rank solutions. If provided, the best
            ``n_best`` solutions by this metric are selected -- highest-first
            for coverage proportions, lowest-first for travel costs.
            If None, the existing order of ``solution_df`` is used.
        title : str or None, optional
            Title applied to each subplot. If provided, overrides
            ``subplot_title``. If None, subplot titles are controlled by
            ``subplot_title``.
        subplot_title : str or None, default="default"
            Title for each subplot. If "default", an automatic title is
            generated based on the objective and solution metrics. If a
            string is provided, it may be evaluated using
            ``_safe_evaluate`` with access to the selected solution.
            If None, no subplot titles are shown.
        show_all_locations : bool, default=True
            If True, plots all candidate site locations in addition to the
            selected sites.
        plot_site_allocation : bool, default=False
            If True, regions are coloured by the assigned (nearest) site.
            Overrides other colouring options.
        plot_regions_not_meeting_threshold : bool, default=False
            If True, regions are coloured based on whether they fall within
            the coverage threshold. Overrides default cost-based colouring.
        cmap : str or matplotlib colormap, optional
            Colormap used for plotting. If None, a default is chosen:
            - Sequential colormap (e.g., "Blues") for cost-based plots
            - Qualitative colormap (e.g., "Set2") for site allocation plots
        chosen_site_colour : str, default="black"
            Colour used to plot selected site locations.
        unchosen_site_colour : str, default="grey"
            Colour used to plot unselected candidate site locations.
        unchosen_site_opacity : float, default=0.5
            Opaqueness/alpha of the points for unchosen sites
        n_cols : int, optional
            Number of columns in the subplot grid. If None, determined
            automatically.
        n_rows : int, optional
            Number of rows in the subplot grid. If None, determined
            automatically.
        fig_size_multiplier: int, optional
            Factor to adjust overall plot size
        annotation_size: int, optional
            Font size of site name annotations on the map
        site_legend_loc: str, default="upper right"
            Adjust position of the site legend. Accepts standard
            matplotlib positioning strings. Only used if required sites are present
            in the problem.
        site_legend_bbox_to_anchor : tuple, default (0.9,0.9)
            If provided, places the legend outside the plot area. Accepts a
            2-tuple (x, y) in axes coordinates. Common examples:
            - (1.05, 1) for right side, top aligned
            - (0.5, -0.1) for bottom center
            - (1.05, 0.5) for right side, center aligned
            When set, legend_loc is used as the anchor point on the legend box.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object containing all subplots.
        ax : matplotlib.axes.Axes
            The last Axes object created in the plotting loop.

        Raises
        ------
        ValueError
            If the region geometry layer has not been initialised in the
            ``site_problem`` instance.

        Notes
        -----
        The method requires a valid ``region_geometry_layer`` to be present in
        ``site_problem``. This should be added prior to calling this method.

        Subplots are arranged in a grid determined by ``n_best``, ``n_cols``,
        and ``n_rows``. If neither ``n_cols`` nor ``n_rows`` is provided, a
        default layout with up to 5 columns is used.

        Depending on the flags provided, regions are coloured by:
        - "min_cost" (default): travel time/distance to the nearest site
        - "selected_site": assigned site (categorical)
        - "within_threshold": whether the region meets the coverage threshold

        When plotting cost-based maps, a consistent global colour scale is
        applied across all subplots.

        When plotting categorical site allocations, a consistent colour
        mapping is constructed across all subplots to ensure comparability.

        A shared legend or colourbar is added to the figure depending on the
        plotting mode.

        The method assumes that lower values of the ranking metric correspond
        to better solutions.
        """
        if self.site_problem.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        cost_col, site_col, _, matrix_unit, suffix = self._resolve_travel_columns(matrix)

        if n_best > len(self.solution_df):
            n_best = len(self.solution_df)
            warn(
                f"n_best parameter higher than number of available solutions. Returning {len(self.solution_df)} solutions."
            )

        # Set up number of rows and columns
        if n_cols is None and n_rows is None:
            max_cols = 5
            ncols = min(n_best, max_cols)
            nrows = math.ceil(n_best / ncols)
        elif n_cols is None and n_rows is not None:
            nrows = n_rows
            ncols = math.ceil(n_best / nrows)
        elif n_cols is not None and n_rows is None:
            ncols = n_cols
            nrows = math.ceil(n_best / ncols)
        else:
            nrows = n_rows
            ncols = n_cols

        # Set up subplots
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_size_multiplier * ncols, fig_size_multiplier * nrows),
        )

        # Flatten axs in case it's a 2D array
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else:
            axs = [axs]  # Single subplot case

        # Sort and select top solutions
        if sort_by is not None:
            sorted_df = (
                _sort_solutions_by_metric(self.solution_df, sort_by)
                .reset_index()
                .head(n_best)
            )
        else:
            sorted_df = self.solution_df.reset_index().head(n_best)

        # Choose appropriate colourmap if not provided
        if cmap is None:
            if plot_site_allocation:
                cmap = "Set2"
            elif plot_regions_not_meeting_threshold:
                cmap = "Set1"
            else:
                cmap = "Blues"

        # Set up consistent coloring across subplots
        global_vmin = None
        global_vmax = None
        site_color_map = None
        discrete_cmap = None
        colors = None

        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            # Calculate global color scale for min_cost (or the secondary
            # matrix's own min cost). Concatenated first so pandas' own
            # NaN-skipping min/max run once over the combined values --
            # Python's builtin min()/max() over a list of per-solution
            # `.min()`/`.max()` results is NaN-order-dependent whenever any
            # solution is entirely unreachable (all-NaN cost_col), since
            # NaN comparisons are always False.
            combined_costs = pd.concat(
                [df[cost_col] for df in sorted_df["problem_df"]]
            )
            global_vmin = combined_costs.min()
            global_vmax = combined_costs.max()

        elif plot_site_allocation:
            # Create consistent color mapping for sites
            all_allocated_sites = set()
            for df in sorted_df["problem_df"]:
                all_allocated_sites.update(df[site_col].unique())

            site_color_map = self._site_allocation_color_map(
                all_allocated_sites, cmap=cmap
            )

        elif plot_regions_not_meeting_threshold:
            # Set up discrete colormap for threshold
            cmap_obj = plt.get_cmap(cmap)
            if hasattr(cmap_obj, "colors"):
                colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
            else:
                colors = [cmap_obj(0.0), cmap_obj(1.0)]
            discrete_cmap = ListedColormap(colors)

        # Plot each solution
        for i, ax in enumerate(axs[:n_best]):
            solution = sorted_df.iloc[[i]]

            ax, has_required_sites = self._plot_single_solution_map(
                ax=ax,
                solution=solution,
                show_all_locations=show_all_locations,
                label_all_locations=label_all_locations,
                plot_site_allocation=plot_site_allocation,
                plot_regions_not_meeting_threshold=plot_regions_not_meeting_threshold,
                cmap=cmap,
                chosen_site_colour=chosen_site_colour,
                unchosen_site_colour=unchosen_site_colour,
                unchosen_site_opacity=unchosen_site_opacity,
                annotation_size=annotation_size,
                label_wrap_width=legend_wrap_width,
                global_vmin=global_vmin,
                global_vmax=global_vmax,
                site_color_map=site_color_map,
                discrete_cmap=discrete_cmap,
                colors=colors,
                add_legend=False,  # Legends added at figure level
                legend_kwargs=None,
                matrix=matrix,
            )

            # Add subplot titles
            if subplot_title is not None:
                if subplot_title == "default":
                    unreachable_fragment = _unreachable_metrics_fragment(
                        solution[f"regions_unreachable{suffix}"].values[0]
                    )
                    unreachable_suffix = (
                        f" \n{unreachable_fragment}" if unreachable_fragment else ""
                    )
                    if self.objectives == "mclp":
                        threshold_val = self._resolve_coverage_threshold(
                            matrix, solution["coverage_threshold"].values[0]
                        )
                        ax.set_title(
                            f"Demand covered within threshold of {threshold_val} {matrix_unit}: {solution[f'proportion_within_coverage_threshold{suffix}'].values[0]:.1%} \nUnweighted Average: {solution[f'unweighted_average{suffix}'].values[0]:.1f} {matrix_unit} \nMaximum: {solution[f'max{suffix}'].values[0]:.1f} {matrix_unit}{unreachable_suffix}"
                        )
                    elif self.objectives in [
                        "simple_p_median",
                        "hybrid_simple_p_median",
                    ]:
                        ax.set_title(
                            f"Unweighted Average: {solution[f'unweighted_average{suffix}'].values[0]:.1f} {matrix_unit} \nMaximum: {solution[f'max{suffix}'].values[0]:.1f} {matrix_unit}{unreachable_suffix}"
                        )
                    else:
                        ax.set_title(
                            f"Weighted Average: {solution[f'weighted_average{suffix}'].values[0]:.1f} {matrix_unit} \nMaximum: {solution[f'max{suffix}'].values[0]:.1f} {matrix_unit}{unreachable_suffix}"
                        )

                    # These metrics are chosen from `self.objectives`, which
                    # no longer describes the ranking once solve(rank_on=...)
                    # is used -- so name the metric actually optimised.
                    ranking_line = self._ranking_metric_line(solution.iloc[0])
                    if ranking_line:
                        ax.set_title(f"{ax.get_title()}\n{ranking_line}")
                else:
                    ax.set_title(_safe_evaluate(subplot_title, solution=solution))

            if title is not None:
                ax.set_title(title)

        # Hide unused subplots
        for i in range(n_best, len(axs)):
            axs[i].axis("off")

        # Add shared legends/colorbars
        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax)
            )
            sm._A = []
            unit_parenthetical = f" ({matrix_unit})" if matrix_unit else ""
            fig.colorbar(
                sm,
                ax=axs,
                fraction=0.02,
                pad=0.04,
                label=f"Travel time to nearest site{unit_parenthetical}",
            )

        elif plot_site_allocation:
            legend_patches = [
                mpatches.Patch(color=color, label=str(site))
                for site, color in site_color_map.items()
            ]
            fig.legend(
                handles=legend_patches,
                title="Allocated Sites",
                loc="center right",
                bbox_to_anchor=(1.05, 0.5),
            )

        elif plot_regions_not_meeting_threshold:
            threshold_val = self._resolve_coverage_threshold(
                matrix, sorted_df.iloc[0]["coverage_threshold"]
            )
            legend_patches = [
                mpatches.Patch(
                    color=colors[0],
                    label=_wrap_label(
                        f"Further than {threshold_val} {matrix_unit}\nfrom nearest site",
                        width=legend_wrap_width,
                    ),
                ),
                mpatches.Patch(
                    color=colors[1],
                    label=_wrap_label(
                        f"Within {threshold_val} {matrix_unit}\nof nearest site",
                        width=legend_wrap_width,
                    ),
                ),
            ]
            fig.legend(
                handles=legend_patches,
                title="Coverage Status",
                loc="center right",
                bbox_to_anchor=(1.05, 0.5),
            )

        if has_required_sites:
            legend_elements = []

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=chosen_site_colour,
                    markersize=10,
                    label="Required sites",
                )
            )

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=chosen_site_colour,
                    markersize=10,
                    label="Additional selected sites",
                )
            )

            fig.legend(
                handles=legend_elements,
                loc=site_legend_loc,
                bbox_to_anchor=site_legend_bbox_to_anchor,
                framealpha=0.9,
            )

        return fig, axs

    def plot_solution_comparison(
        self,
        solutions_config,
        figsize=(16, 8),
        title=None,
        title_fontsize=14,
        **shared_plot_kwargs,
    ):
        """
        Plot multiple solutions side-by-side for comparison.

        Parameters
        ----------
        solutions_config : list of dict
            List of configuration dictionaries for each subplot. Each dict can contain
            any parameters accepted by plot_best_combination(), e.g.:
            - 'solution_rank': int
            - 'site_names': list of str
            - 'site_indices': list of int or np.ndarray
            - 'sort_by': str
            - 'title': str (subplot title)
            - Any other plot_best_combination() parameters
        figsize : tuple, default=(16, 8)
            Figure size as (width, height) in inches.
        title : str, optional
            Overall figure title (suptitle). If None, no suptitle is added.
        title_fontsize : int, default=14
            Font size for the overall figure title.
        **shared_plot_kwargs
            Keyword arguments applied to all subplots. Individual subplot configs
            override these shared settings.

        Returns
        -------
        tuple
            (fig, axes) where fig is the Figure and axes is an array of Axes objects.

        Examples
        --------
        # Compare best vs 2nd best solution
        fig, axes = solver.plot_solution_comparison([
            {'solution_rank': 1, 'title': 'Best Solution'},
            {'solution_rank': 2, 'title': '2nd Best Solution'}
        ])

        # Compare specific site combinations
        fig, axes = solver.plot_solution_comparison([
            {'site_names': ['Site A', 'Site B', 'Site C']},
            {'site_names': ['Site D', 'Site E', 'Site F']}
        ], show_all_locations=False)

        # Three-way comparison with custom ranking
        fig, axes = solver.plot_solution_comparison([
            {'solution_rank': 1, 'sort_by': 'weighted_average'},
            {'solution_rank': 1, 'sort_by': 'max'},
            {'solution_rank': 1, 'sort_by': 'proportion_within_coverage_threshold'}
        ], figsize=(24, 8))
        """
        n_plots = len(solutions_config)

        if n_plots == 0:
            raise ValueError("solutions_config must contain at least one configuration")

        # Create subplots
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        # Handle single plot case (axes won't be an array)
        if n_plots == 1:
            axes = [axes]

        # Plot each solution
        for i, config in enumerate(solutions_config):
            # Merge shared kwargs with individual config (config takes precedence)
            plot_kwargs = {**shared_plot_kwargs, **config}

            # Get the solution based on config
            solution = _select_solution(
                self.solution_df,
                sort_by=config.get("sort_by"),
                solution_rank=config.get("solution_rank", 1),
                site_names=config.get("site_names"),
                site_indices=config.get("site_indices"),
            )

            # Extract plotting-specific parameters
            show_all_locations = plot_kwargs.pop("show_all_locations", True)
            label_all_locations = plot_kwargs.pop("label_all_locations", False)
            plot_site_allocation = plot_kwargs.pop("plot_site_allocation", False)
            plot_regions_not_meeting_threshold = plot_kwargs.pop(
                "plot_regions_not_meeting_threshold", False
            )
            cmap = plot_kwargs.pop("cmap", None)
            chosen_site_colour = plot_kwargs.pop("chosen_site_colour", "black")
            unchosen_site_colour = plot_kwargs.pop("unchosen_site_colour", "grey")
            unchosen_site_opacity = plot_kwargs.pop("unchosen_site_opacity", 0.7)
            annotation_size = plot_kwargs.pop("annotation_size", 6)
            label_wrap_width = plot_kwargs.pop("label_wrap_width", 40)
            legend_loc = plot_kwargs.pop("legend_loc", "upper right")
            legend_bbox_to_anchor = plot_kwargs.pop(
                "legend_bbox_to_anchor", (1.75, 0.5)
            )
            legend_fontsize = plot_kwargs.pop("legend_fontsize", 10)
            site_legend_loc = plot_kwargs.pop("site_legend_loc", "best")
            subplot_title = plot_kwargs.pop("title", "default")
            subplot_title_fontsize = plot_kwargs.pop("title_fontsize", 12)
            matrix = plot_kwargs.pop("matrix", None)

            # Choose colormap if not provided
            if cmap is None:
                if plot_site_allocation:
                    cmap = "Set2"
                elif plot_regions_not_meeting_threshold:
                    cmap = "Set1"
                else:
                    cmap = "Blues"

            # Prepare discrete colormap for threshold plotting
            discrete_cmap = None
            colors = None
            if plot_regions_not_meeting_threshold:
                cmap_obj = plt.get_cmap(cmap)
                if hasattr(cmap_obj, "colors"):
                    colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
                else:
                    colors = [cmap_obj(0.0), cmap_obj(1.0)]
                discrete_cmap = ListedColormap(colors)

            # Set up legend kwargs
            legend_kwargs = {
                "loc": legend_loc,
                "fontsize": legend_fontsize,
            }
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

            # Plot the solution
            axes[i], has_required_sites = self._plot_single_solution_map(
                ax=axes[i],
                solution=solution,
                show_all_locations=show_all_locations,
                label_all_locations=label_all_locations,
                plot_site_allocation=plot_site_allocation,
                plot_regions_not_meeting_threshold=plot_regions_not_meeting_threshold,
                cmap=cmap,
                chosen_site_colour=chosen_site_colour,
                unchosen_site_colour=unchosen_site_colour,
                unchosen_site_opacity=unchosen_site_opacity,
                annotation_size=annotation_size,
                label_wrap_width=label_wrap_width,
                discrete_cmap=discrete_cmap,
                colors=colors,
                add_legend=True,
                legend_kwargs=legend_kwargs,
                matrix=matrix,
            )

            # Add required sites legend if needed
            if has_required_sites:
                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="w",
                        markerfacecolor=chosen_site_colour,
                        markersize=10,
                        label="Required sites",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=chosen_site_colour,
                        markersize=10,
                        label="Additional selected sites",
                    ),
                ]
                axes[i].legend(handles=legend_elements, loc=site_legend_loc)

            # Add subplot title
            if subplot_title is not None:
                if subplot_title == "default":
                    # Generate default title (same logic as plot_best_combination)
                    if config.get("site_indices") is not None:
                        title_prefix = (
                            f"Solution for {self.n_sites} sites (by site indices)"
                        )
                    elif config.get("site_names") is not None:
                        title_prefix = (
                            f"Solution for {self.n_sites} sites (by site names)"
                        )
                    elif config.get("solution_rank", 1) == 1:
                        title_prefix = f"Best solution for {self.n_sites} sites"
                    else:
                        rank = config.get("solution_rank", 1)
                        rank_suffix = _get_ordinal_suffix(rank)
                        title_prefix = f"{rank}{rank_suffix} best solution for {self.n_sites} sites"

                    _, _, _, _, config_suffix = self._resolve_travel_columns(matrix)

                    if self.objectives == "mclp":
                        metrics = (
                            f"Demand covered: {solution[f'proportion_within_coverage_threshold{config_suffix}'].values[0]:.1%} | "
                            f"Avg: {solution[f'unweighted_average{config_suffix}'].values[0]:.1f} | "
                            f"Max: {solution[f'max{config_suffix}'].values[0]:.1f}"
                        )
                    elif self.objectives in [
                        "simple_p_median",
                        "hybrid_simple_p_median",
                    ]:
                        metrics = (
                            f"Unweighted Avg: {solution[f'unweighted_average{config_suffix}'].values[0]:.1f} | "
                            f"Max: {solution[f'max{config_suffix}'].values[0]:.1f}"
                        )
                    else:
                        metrics = (
                            f"Weighted Avg: {solution[f'weighted_average{config_suffix}'].values[0]:.1f} | "
                            f"Max: {solution[f'max{config_suffix}'].values[0]:.1f}"
                        )

                    unreachable_fragment = _unreachable_metrics_fragment(
                        solution[f"regions_unreachable{config_suffix}"].values[0]
                    )
                    if unreachable_fragment:
                        metrics += f" | {unreachable_fragment}"

                    # Chosen from `self.objectives`, which stops describing
                    # the ranking under solve(rank_on=...) -- name the real
                    # one rather than implying the default was optimised.
                    ranking_line = self._ranking_metric_line(solution.iloc[0])
                    if ranking_line:
                        metrics += f"\n{ranking_line}"

                    axes[i].set_title(
                        f"{title_prefix}\n{metrics}", fontsize=subplot_title_fontsize
                    )
                else:
                    axes[i].set_title(
                        _safe_evaluate(subplot_title, solution=solution),
                        fontsize=subplot_title_fontsize,
                    )

        # Add overall title
        if title is not None:
            fig.suptitle(title, fontsize=title_fontsize, y=0.98)

        plt.tight_layout()

        return fig, axes


##################################
# MARK: Distributions
##################################
class DistributionPlotsMixin:
    def plot_travel_time_distribution(
        self,
        top_n=1,
        bottom_n=None,
        sort_by=None,
        secondary_ranking="max",
        title="default",
        height=None,
        height_per_plot=250,
        compare_to_best=False,
        matrix=None,
        **kwargs,
    ):
        """
        Plot travel time distributions for selected solutions.

        This method generates faceted histograms of travel time (or cost)
        distributions for the top and/or bottom-performing solutions in the
        solution set. Each subplot corresponds to a single solution and includes
        summary statistics annotations.

        Parameters
        ----------
        top_n : int, optional, default=1
            Number of top-ranked solutions to include. If ``sort_by`` is provided,
            solutions are sorted before selection; otherwise, the existing order,
            which is based on the objective chosen for solving, is used.
        bottom_n : int, optional
            Number of bottom-ranked solutions to include. If provided, these are
            appended to the selected top solutions.
        sort_by : str, optional
            Column name used to rank solutions, best-first (highest for coverage
            proportions, lowest for travel costs), with ``secondary_ranking`` as
            a tie-breaker. Each column's direction is resolved independently.
            If None, no additional sorting is applied.
        secondary_ranking : str, default="max"
            Secondary column used for tie-breaking when sorting by ``sort_by``.
        title : str, default="default"
            Title for the plot. If "default", an automatic title is generated
            based on ranking and objectives.
        height : int, optional
            Total height of the figure in pixels. If None, height is determined
            dynamically using ``height_per_plot``.
        height_per_plot : int, default=250
            Height allocated per subplot when ``height`` is not specified.
        compare_to_best : bool, default=False
            If True, plots the difference in travel time relative to the best
            solution (i.e., ``min_cost_diff``). Adds a vertical reference line
            at zero. If False, plots absolute travel times (``min_cost``).
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. If provided, distributions and
            reference lines use that matrix's own travel costs and summary
            metrics instead of the primary travel matrix.
        **kwargs
            Additional keyword arguments passed to ``plotly.express.histogram``.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure containing faceted histograms of travel time
            distributions for the selected solutions.

        Notes
        -----
        When ``compare_to_best=True``, the best solution (first in the filtered
        set) is used as the reference for computing differences.

        Subplots are arranged using ``facet_row``, with annotations adjusted
        manually to align with subplot domains.

        Vertical reference lines are added for key metrics when
        ``compare_to_best=False``:
        - Weighted average
        - Unweighted average
        - 90th percentile
        - Maximum

        The method assumes that lower values of the ranking metric correspond
        to better solutions.
        """
        cost_col, _, _, _, suffix = self._resolve_travel_columns(matrix)
        weighted_average_col = f"weighted_average{suffix}"
        unweighted_average_col = f"unweighted_average{suffix}"
        percentile_90th_col = f"90th_percentile{suffix}"
        max_col = f"max{suffix}"
        regions_unreachable_col = f"regions_unreachable{suffix}"

        if sort_by is not None:
            solutions_sorted = _sort_solutions_by_metric(
                self.solution_df, [sort_by, secondary_ranking]
            ).reset_index(drop=True)
        else:
            solutions_sorted = self.solution_df.reset_index(drop=True)

        filtered_dfs = []

        if top_n is not None:
            filtered_dfs.append(solutions_sorted.head(top_n))

        if bottom_n is not None:
            filtered_dfs.append(solutions_sorted.tail(bottom_n))

        solutions_filtered = pd.concat(filtered_dfs)

        # Convert columns to be compared to tuples to allow dupliate detection
        solutions_filtered["site_indices_comp"] = (
            solutions_filtered["site_indices"]
            .apply(lambda x: list(set([int(i) for i in x])))
            .astype("str")
        )

        solutions_filtered["site_names_comp"] = (
            solutions_filtered["site_names"].apply(lambda x: list(set(x))).astype("str")
        )

        solutions_filtered = solutions_filtered.drop_duplicates(
            subset=["site_indices_comp", "site_names_comp"]
        )

        dfs = []
        if compare_to_best:
            best_df = solutions_filtered["problem_df"].iloc[0]

        for index, row in solutions_filtered.iterrows():
            df = row["problem_df"].copy()
            df["site_indices"] = ", ".join([str(int(i)) for i in row["site_indices"]])
            df["site_names"] = ", ".join(row["site_names"])
            df["weighted_average"] = row[weighted_average_col]
            df["unweighted_average"] = row[unweighted_average_col]
            df["max"] = row[max_col]
            df["90th_percentile"] = row[percentile_90th_col]
            df["regions_unreachable"] = row[regions_unreachable_col]
            if compare_to_best:
                df["min_cost_diff"] = df[cost_col] - best_df[cost_col].values
            dfs.append(df)

        dfs = pd.concat(dfs)

        # A demand location with no feasible journey (`add_travel_matrix(
        # allow_missing=True)`) has a NaN cost_col value -- plotly's
        # histogram silently excludes it from the bars/density with no
        # indication anything was left out, so the excluded count is
        # named explicitly in the label instead.
        dfs["label"] = (
            "Sites: "
            + dfs["site_names"]
            + " | Weighted Avg: "
            + dfs["weighted_average"].round(2).astype(str)
            + " | Unweighted Avg: "
            + dfs["unweighted_average"].round(2).astype(str)
            + " | P90: "
            + dfs["90th_percentile"].round(2).astype(str)
            + " | Max: "
            + dfs["max"].round(2).astype(str)
            + np.where(
                dfs["regions_unreachable"] > 0,
                " | Unreachable: " + dfs["regions_unreachable"].astype(int).astype(str),
                "",
            )
        )

        fig = px.histogram(
            dfs,
            x="min_cost_diff" if compare_to_best else cost_col,
            facet_row="label",
            nbins=30,
            histnorm="probability density",
            height=height,
            color_discrete_sequence=["lightsteelblue"],
            template="plotly_white",
            **kwargs,
        )

        # --- Subplot title annotations ---
        # Get the top of each subplot domain directly from the axes,
        # rather than trusting the facet annotation's y position.
        facet_annotations = [a for a in fig.layout.annotations if "label=" in a.text]

        num_plots = len(solutions_filtered)

        # Collect axis domains — facet_row creates yaxis, yaxis2, yaxis3 etc.
        axis_tops = []
        for i in range(
            1,
            num_plots + 1,
        ):
            axis_key = "yaxis" if i == 1 else f"yaxis{i}"
            axis = fig.layout[axis_key]
            if axis and axis.domain:
                axis_tops.append(axis.domain[1])  # domain[1] is the top of the subplot

        # Sort descending so axis_tops[0] = topmost subplot
        axis_tops = sorted(axis_tops, reverse=True)

        for i, a in enumerate(facet_annotations):
            top = axis_tops[i] if i < len(axis_tops) else a.y
            a.update(
                text=a.text.replace("label=", ""),
                textangle=0,
                xref="paper",
                yref="paper",
                x=0.0,
                xanchor="left",
                y=top,
                yanchor="bottom",
                font=dict(size=11),
            )
        # --- vline annotations ---
        if compare_to_best:
            fig.add_vline(
                x=0,
                line_color="black",
                line_width=2,
                annotation_text="Best Solution",
                annotation_position="top right",
                annotation_yshift=4,
            )
        else:
            for i, (_, row) in enumerate(solutions_filtered.iterrows()):
                subplot_row = top_n - i  # Plotly facet_row numbers bottom-to-top

                # Weighted average — label sits at the top of the line
                fig.add_vline(
                    x=row[weighted_average_col],
                    line_dash="dash",
                    line_color="darkolivegreen",
                    row=subplot_row,
                    col=1,
                    annotation_text="Weighted<br>Average",
                    annotation_position="top left",
                    annotation_yshift=5,
                    annotation_font_color="darkolivegreen",
                )

                # unweighted average — offset downward so it doesn't overlap WA
                fig.add_vline(
                    x=row[unweighted_average_col],
                    line_dash="dashdot",
                    line_color="darkcyan",
                    row=subplot_row,
                    col=1,
                    annotation_text="Unweighted<br>Average",
                    annotation_position="top right",
                    annotation_yshift=-10,
                    annotation_font_color="darkcyan",
                )

                # 90th percentile — offset downward so it doesn't overlap WA
                fig.add_vline(
                    x=row[percentile_90th_col],
                    line_dash="dot",
                    line_color="darkorange",
                    row=subplot_row,
                    col=1,
                    annotation_text="P90",
                    annotation_position="top right",
                    annotation_yshift=-15,
                    annotation_font_color="darkorange",
                )

                # 90th percentile — offset downward so it doesn't overlap WA
                fig.add_vline(
                    x=row[max_col],
                    line_dash="solid",
                    line_color="tomato",
                    row=subplot_row,
                    col=1,
                    annotation_text="Max",
                    annotation_position="top right",
                    annotation_yshift=-20,
                    annotation_font_color="tomato",
                )

        # --- Title ---
        if title == "default":
            top_n_title = f"Top {top_n}" if top_n is not None else ""
            bottom_n_title = f"Bottom {bottom_n}" if bottom_n is not None else ""

            if top_n is not None and bottom_n is not None:
                title_str = f"{top_n_title} and {bottom_n_title}"
            elif top_n is not None:
                title_str = top_n_title
            elif bottom_n is not None:
                title_str = bottom_n_title
            else:
                title_str = ""

            if sort_by is not None:
                fig.update_layout(
                    title=f"Distribution of Travel Times ({title_str} Solutions by {sort_by.replace('_', ' ').title()})"
                )
            else:
                fig.update_layout(
                    title=f"Distribution of Travel Times ({title_str} Solutions: {self.objectives.replace('_', ' ').title()})"
                )
        else:
            fig.update_layout(title=title)

        # --- Height ---
        if height is None:
            fig.update_layout(height=max(300, height_per_plot * top_n))

        return fig


##################################
# MARK: Equity
##################################
class EquityPlotsMixin:
    def check_solution_equity(
        self,
        solution_rank=1,
        return_plot=False,
        interactive=True,
        title="default",
        show_average=True,
        plot_solution_metric_as_line="weighted_average",
        sort_by=None,
        ax=None,
        colour_mode: Optional[Literal["gradient", "above_below_avg"]] = None,
        show_site_names=False,
        matrix=None,
    ):
        """
        Summarise and optionally plot equity metrics for a selected solution.

        This method computes the mean minimum cost (``min_cost``) grouped by the
        configured equity category for a given solution. It can return either the
        aggregated data or a visualisation using Plotly (interactive) or Matplotlib
        (static).

        Parameters
        ----------
        solution_rank : int, default=1
            Rank of the solution to evaluate (1-indexed).
        return_plot : bool, default=False
            If True, return a plot instead of the summary DataFrame.
        title : str, default="default"
            Title for the plot. If "default", a title is generated automatically.
        show_average : bool, default=True
            If True, display the overall average of ``min_cost`` as a horizontal
            dotted line on the plot.
        sort_by : str or None, optional
            Column name used to rank solutions before selecting the specified
            ``solution_rank``. If None, the existing order is used.
        interactive : bool, default=True
            If True, return an interactive Plotly figure. If False, return a
            Matplotlib figure.
        ax : matplotlib.axes.Axes or None, optional
            Existing Matplotlib axes to plot on when ``interactive=False``.
            If None, a new figure and axes are created.
        show_site_names : bool, default=False
            If True, the default plot title lists the selected sites by name
            (``site_names``) instead of by index (``site_indices``).
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. If provided, the summary is
            computed from that matrix's own min-cost column instead of the
            primary travel matrix.

        Returns
        -------
        pandas.DataFrame or plotly.graph_objs._figure.Figure or matplotlib.figure.Figure
            If ``return_plot=False``, returns a DataFrame with mean ``min_cost`` per
            equity group, in ascending raw-band order. Otherwise, returns a Plotly
            or Matplotlib figure depending on the ``interactive`` flag -- the
            plotted bars are reordered most- to least-disadvantaged per
            ``add_equity_data(disadvantaged_end=...)`` (ties within a tertile
            keep ascending band order, matching
            ``population_impact_by_equity_group()``'s own ordering), and the
            equity axis is labelled "(most to least disadvantaged)" so the
            direction is never left for the reader to guess. The un-plotted
            DataFrame is intentionally left in its original ascending order --
            only the chart's bar order/label changes.

        Notes
        -----
        - The equity grouping column is defined by
        ``self.site_problem._equity_data_equity_col``.
        - The plotted metric is the mean of ``min_cost`` within each group.
        - When using Matplotlib with a provided ``ax``, the plot is drawn onto the
        supplied axes and the corresponding figure is returned.
        """
        cost_col, _, _, unit, _ = self._resolve_travel_columns(matrix)
        unit_parenthetical = f" ({unit})" if unit else ""
        cost_axis_label = f"Average travel time{unit_parenthetical}"
        equity_axis_label = (
            self.site_problem._equity_data_label
            or self.site_problem._equity_data_equity_col
        )

        if sort_by is not None:
            plotting_row = _sort_solutions_by_metric(self.solution_df, sort_by).iloc[
                solution_rank - 1
            ]
        else:
            plotting_row = self.solution_df.iloc[solution_rank - 1]

        summary_equity_df = (
            plotting_row["problem_df"]
            .groupby(self.site_problem._equity_data_equity_col)[cost_col]
            .agg("mean")
            .round(2)
            .reset_index()
            .rename(columns={cost_col: "min_cost"})
        )

        if not return_plot:
            return summary_equity_df
        else:
            equity_col = self.site_problem._equity_data_equity_col
            disadvantaged_end = getattr(
                self.site_problem, "_equity_data_disadvantaged_end", None
            )
            ordered_bins = _order_bins_most_to_least_disadvantaged(
                sorted(summary_equity_df[equity_col]), disadvantaged_end
            )
            summary_equity_df = (
                summary_equity_df.set_index(equity_col).loc[ordered_bins].reset_index()
            )
            equity_axis_label = f"{equity_axis_label} (most to least disadvantaged)"

            if title == "default":
                if show_site_names:
                    sites_str = ", ".join(str(name) for name in plotting_row.site_names)
                    sites_label = f"Sites: {sites_str}"
                else:
                    sites_str = ", ".join(str(int(i)) for i in plotting_row.site_indices)
                    sites_label = f"Site Indices: {sites_str}"

                title = f"Solution equity - by {self.site_problem._equity_data_label}\nSolution Rank {solution_rank} ({sites_label})"

            if show_average:
                avg_value = plotting_row[plot_solution_metric_as_line]

            if interactive:
                import plotly.express as px

                axis_labels = {
                    equity_col: equity_axis_label,
                    "min_cost": cost_axis_label,
                }
                # category_orders forces both a categorical x-axis and the
                # most-to-least-disadvantaged row order above -- band
                # values are numeric (e.g. IMD decile), and Plotly Express
                # otherwise defaults numeric x columns to a continuous axis
                # sorted by value, silently undoing the reordering.
                category_orders = {equity_col: [str(b) for b in ordered_bins]}
                summary_equity_df = summary_equity_df.astype({equity_col: str})

                if colour_mode == "gradient":
                    fig = px.bar(
                        summary_equity_df,
                        x=equity_col,
                        y="min_cost",
                        color="min_cost",
                        color_continuous_scale=[
                            "#a8e6a3",
                            "#f28b82",
                        ],  # soft green → soft red
                        title=title,
                        labels=axis_labels,
                        category_orders=category_orders,
                    )

                elif colour_mode == "above_below_avg":
                    avg_value = summary_equity_df["min_cost"].mean()
                    summary_equity_df["_above_avg"] = (
                        summary_equity_df["min_cost"] > avg_value
                    )

                    fig = px.bar(
                        summary_equity_df,
                        x=equity_col,
                        y="min_cost",
                        color="_above_avg",
                        color_discrete_map={
                            True: "#f28b82",  # red
                            False: "#a8e6a3",  # green
                        },
                        title=title,
                        labels=axis_labels,
                        category_orders=category_orders,
                    )

                else:
                    fig = px.bar(
                        summary_equity_df,
                        x=equity_col,
                        y="min_cost",
                        title=title,
                        labels=axis_labels,
                        category_orders=category_orders,
                    )

                if show_average:
                    fig.add_hline(
                        y=avg_value,
                        line_dash="dot",
                        annotation_text=f"Avg: {avg_value:.2f}",
                        annotation_position="top right",
                    )

                return fig
            else:
                import matplotlib.pyplot as plt

                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.figure

                band_labels = [str(b) for b in summary_equity_df[equity_col]]
                x_positions = np.arange(len(band_labels))
                y_vals = summary_equity_df["min_cost"]

                if colour_mode == "gradient":
                    norm = plt.Normalize(y_vals.min(), y_vals.max())
                    cmap = plt.cm.RdYlGn_r  # green (low) → red (high)
                    colors = cmap(norm(y_vals))

                elif colour_mode == "above_below_avg":
                    avg_value = y_vals.mean()
                    colors = ["#f28b82" if v > avg_value else "#a8e6a3" for v in y_vals]

                else:
                    colors = None

                # Positional x (not the raw band values) -- band values are
                # numeric (e.g. IMD decile), and matplotlib places a bar at
                # its literal numeric x-coordinate regardless of row order,
                # so plotting directly against those values would silently
                # undo the most-to-least-disadvantaged reordering above.
                ax.bar(x_positions, y_vals, color=colors)

                ax.set_title(title)
                ax.set_xlabel(equity_axis_label)
                ax.set_ylabel(cost_axis_label)

                if show_average:
                    ax.axhline(
                        y=avg_value,
                        linestyle=":",
                        label=f"Avg: {avg_value:.2f}",
                    )
                    ax.legend()

                ax.set_xticks(x_positions)
                ax.set_xticklabels(band_labels)
                ax.tick_params(axis="x", rotation=0)
                plt.tight_layout()

                return fig

    def plot_top_n_solution_equity(
        self,
        n=4,
        sort_by=None,
        show_average=True,
        plot_solution_metric_as_line="weighted_average",
        colour_mode: Optional[Literal["gradient", "above_below_avg"]] = None,
        cols=2,
        figsize_multiplier=5,
        show_site_names=False,
        matrix=None,
    ):
        """
        Plot equity summaries for the top N solutions in a grid of subplots.

        This method generates a series of bar plots showing the distribution of
        mean minimum cost (``min_cost``) across equity groups for the top-ranked
        solutions. It reuses ``check_solution_equity`` to ensure consistent
        computation and styling.

        Parameters
        ----------
        n : int, default=4
            Number of top-ranked solutions to plot.
        sort_by : str or None, optional
            Column name used to rank solutions before selecting the top ``n``.
            If None, the existing order is used.
        show_average : bool, default=True
            If True, display the overall average of ``min_cost`` as a horizontal
            dotted line on each subplot.
        cols : int, default=2
            Number of subplot columns. The number of rows is determined automatically.
        show_site_names : bool, default=False
            If True, each subplot title lists the selected sites by name
            instead of by index.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`, passed through to
            `check_solution_equity()` for every subplot.

        Returns
        -------
        matplotlib.figure.Figure
            A Matplotlib figure containing the grid of subplots.

        Notes
        -----
        - Subplots are arranged row-wise based on the specified number of columns.
        - Any unused subplot axes are removed if ``n`` does not fill the grid.
        - All plots are generated using the non-interactive (Matplotlib) mode of
        ``check_solution_equity``.
        - Each subplot corresponds to a solution ranked from 1 to ``n``.
        """
        import matplotlib.pyplot as plt
        import math

        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(figsize_multiplier * cols, figsize_multiplier * rows)
        )
        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            self.check_solution_equity(
                solution_rank=i + 1,
                return_plot=True,
                sort_by=sort_by,
                interactive=False,
                show_average=show_average,
                plot_solution_metric_as_line=plot_solution_metric_as_line,
                colour_mode=colour_mode,
                ax=axes[i],
                show_site_names=show_site_names,
                matrix=matrix,
            )

        # Remove unused axes if n doesn't fill grid
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        return fig

    def plot_solution_by_equity():
        pass

    def plot_combination_by_equity(
        self,
        solution_rank=1,
        sort_by=None,
        ncols=3,
        cmap="Blues",
        share_colorbar=True,
        outline_color="grey",
        figsize_multiplier=4,
        groups_to_include="all",
        groupings=None,
        matrix=None,
        **kwargs,
    ):
        """
        Plot maps of travel cost for each equity group, for one solution.

        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. If provided, regions are
            coloured by that matrix's own min-cost column instead of the
            primary travel matrix.

        Notes
        -----
        Each panel's title includes the region count and population
        headcount actually in that group, e.g. "IMD decile: 1 (12 regions,
        4,821 people)". Without this, a group with very little demand looks
        identical to a well-served one -- both render as a near-empty map --
        so a reader can't tell "few areas here" from "no problem here".
        """
        cost_col, _, _, unit, _ = self._resolve_travel_columns(matrix)
        equity_col = self.site_problem._equity_data_equity_col
        demand_col = self.site_problem._demand_data_demand_col
        equity_axis_label = self.site_problem._equity_data_label or equity_col
        unit_parenthetical = f" ({unit})" if unit else ""

        # ---- Base groups from data ----
        raw_groups = sorted(
            self.solution_df["problem_df"][0][equity_col].dropna().unique()
        )

        if groups_to_include != "all":
            raw_groups = [i for i in raw_groups if i in groups_to_include]

        # ---- Handle grouping logic ----
        if groupings is None:
            # Each group stands alone
            plot_groups = {str(g): [g] for g in raw_groups}
        else:
            # Only include values that exist in the data (defensive)
            plot_groups = {
                label: [g for g in vals if g in raw_groups]
                for label, vals in groupings.items()
            }

        group_labels = list(plot_groups.keys())

        n = len(group_labels)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_multiplier * ncols, figsize_multiplier * nrows),
        )
        axes = axes.flatten()

        # ---- Select solution row ----
        if sort_by is not None:
            plotting_row = _sort_solutions_by_metric(self.solution_df, sort_by).iloc[
                solution_rank - 1
            ]
        else:
            plotting_row = self.solution_df.iloc[solution_rank - 1]

        nearest_site_travel_gdf = pd.merge(
            self.site_problem.region_geometry_layer,
            plotting_row["problem_df"],
            left_on=self.site_problem._region_geometry_layer_common_col,
            right_on=self.site_problem._demand_data_id_col,
            suffixes=("", "_y"),
        )

        nearest_site_travel_gdf = nearest_site_travel_gdf.drop(
            nearest_site_travel_gdf.filter(regex="_y$").columns, axis=1
        )

        # ---- Shared color scale ----
        if share_colorbar:
            vmin = nearest_site_travel_gdf[cost_col].min()
            vmax = nearest_site_travel_gdf[cost_col].max()
        else:
            vmin = vmax = None

        # ---- Plotting loop ----
        for i, label in enumerate(group_labels):
            ax = axes[i]
            group_values = plot_groups[label]

            subset = nearest_site_travel_gdf[
                nearest_site_travel_gdf[equity_col].isin(group_values)
            ]

            remainder = nearest_site_travel_gdf[
                ~nearest_site_travel_gdf[equity_col].isin(group_values)
            ]

            # Background
            remainder.plot(
                ax=ax,
                facecolor="none",
                edgecolor=outline_color,
                linewidth=0.5,
                alpha=0.7,
            )

            # Main layer. missing_kwds: a region with no feasible journey
            # (add_travel_matrix(allow_missing=True)) is NaN in cost_col --
            # geopandas draws nothing at all for it by default, an
            # invisible hole in the panel rather than a visibly-explained
            # one.
            subset.plot(
                column=cost_col,
                cmap=cmap,
                linewidth=0.5,
                edgecolor="black",
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                missing_kwds=dict(color=_UNREACHABLE_REGION_COLOUR),
                **kwargs,
            )

            n_regions = len(subset)
            n_people = subset[demand_col].sum()
            title = f"{equity_axis_label}: {label}\n({n_regions:,} regions, {n_people:,.0f} people)"
            unreachable_fragment = _unreachable_metrics_fragment(
                int(subset[cost_col].isna().sum())
            )
            if unreachable_fragment:
                title += f"\n{unreachable_fragment}"
            ax.set_title(title)

            try:
                cx.add_basemap(
                    ax, crs=nearest_site_travel_gdf.crs.to_string(), timeout=30
                )
            except RequestException as e:
                warnings.warn(
                    f"Unable to download background map tiles ({type(e).__name__}). "
                    "Continuing without a basemap.",
                    stacklevel=2,
                )

            ax.axis("off")

        # ---- Remove unused axes ----
        for j in range(len(group_labels), len(axes)):
            fig.delaxes(axes[j])

        # ---- Shared colorbar ----
        if share_colorbar:
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(
                sm,
                ax=axes[: len(group_labels)],
                orientation="vertical",
                fraction=0.03,
                pad=0.02,
            )
            cbar.set_label(f"Travel time to nearest site{unit_parenthetical}")

        return fig, axes
