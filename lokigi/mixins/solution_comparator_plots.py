import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lokigi.plot_utils import plot_solution_sets_comparison
from lokigi.utils import _select_solution


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
            config_1={'solution_rank': 2, 'rank_on': 'weighted_average'},
            config_2={'solution_rank': 2, 'rank_on': 'weighted_average'},
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

    def plot_population_impact_histogram(
        self,
        matrix=None,
        demand=None,
        config_a=None,
        config_b=None,
        bins=30,
        colors=("#9fb8ad", "#1b7a5e"),
        alpha=0.55,
        figsize=(9, 5),
        title="default",
        ax=None,
    ):
        """
        Overlaid before/after histogram of per-region travel cost --
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

        Bars are weighted by demand where available, so bar heights
        represent people rather than regions -- consistent with
        `population_impact_summary()`'s own demand-weighted framing.
        Falls back to an unweighted (one region = one count) histogram
        when no demand data is registered.

        Parameters
        ----------
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Plots that matrix's own
            travel costs instead of the primary matrix's.
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Weights the histogram by that
            scenario's demand instead of the primary demand data.
        config_a, config_b : dict, optional
            Passed straight through to `population_impact_summary()`.
        bins : int, default 30
            Number of histogram bins. The same bin edges (spanning both
            distributions) are used for both sides, so bar heights are
            directly comparable.
        colors : (str, str), default ("#9fb8ad", "#1b7a5e")
            `(set_a colour, set_b colour)`, used for both the histogram
            bars and their matching mean/max reference lines.
        alpha : float, default 0.55
            Opacity of the histogram bars, so the overlap between the two
            distributions stays visible.
        figsize : tuple, default (9, 5)
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
            If `demand` names an unregistered secondary demand scenario,
            or if `set_a` and `set_b`'s selected solutions were evaluated
            against different demand locations (see
            `population_impact_summary()`).
        """
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

        bin_edges = np.histogram_bin_edges(
            np.concatenate(
                [per_region["baseline_cost"].to_numpy(), per_region["current_cost"].to_numpy()]
            ),
            bins=bins,
        )

        axis.hist(
            per_region["baseline_cost"],
            bins=bin_edges,
            weights=weights,
            color=color_a,
            alpha=alpha,
            label=self.labels[0],
        )
        axis.hist(
            per_region["current_cost"],
            bins=bin_edges,
            weights=weights,
            color=color_b,
            alpha=alpha,
            label=self.labels[1],
        )

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
        axis.set_ylabel("People" if has_demand else "Regions")

        if title == "default":
            title = "Travel cost distribution: before vs after"
        if title:
            axis.set_title(title)

        return fig, axis
