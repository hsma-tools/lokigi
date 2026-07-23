from __future__ import annotations

import itertools

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import sweetpareto.vis as spv
import textwrap

from lokigi.utils import _colours_and_styles, _is_maximise_metric, _get_ordinal_suffix

from lokigi.multiobjective import ParetoMetric


##################################
# MARK: Pareto
##################################
class ParetoMixin:
    """
    Pareto front analysis for lokigi SiteProblem solutions.

    Works directly off the metrics dataframe that `problem.solve(...)` already
    produces for every candidate solution (brute-force enumeration). No new
    solve step is required -- this is purely a post-hoc filter + explainer
    layer on top of metrics you're already calculating.
    """

    def plot_simple_pareto_front_pairs(
        self,
        x_axis: str = "weighted_average",
        y_axis: str = "max",
        height=4,
        show_points=True,
        theme="whitegrid",
        maxx=None,
        maxy=None,
        **kwargs,
    ):
        """
        Plot a Pareto front for two selected solution metrics.

        This method generates a Pareto front visualisation comparing two
        performance metrics across all evaluated solutions. It highlights
        the trade-offs between objectives and optionally displays all points
        alongside the Pareto-optimal frontier.

        Parameters
        ----------
        x_axis : str, default="weighted_average"
            Column name representing the metric to plot on the x-axis. Any
            column in `solution_df` is valid, including secondary-travel-
            matrix columns suffixed `__<label>` (e.g.
            "weighted_average__public_transport").
        y_axis : str, default="max"
            Column name representing the metric to plot on the y-axis. Any
            column in `solution_df` is valid, including secondary-travel-
            matrix columns suffixed `__<label>`.
        height : float, default=4
            Height of the plot in inches.
        show_points : bool, default=True
            If True, all solutions are plotted as points in addition to the
            Pareto front.
        theme : str, default="whitegrid"
            Visual theme passed to the underlying plotting function.
        maxx : bool, default=None
            If True, the Pareto front is computed assuming the x-axis metric
            is to be maximised. If False, it is minimised.
            If None, the function automatically infers the value based on
            the metric.
        maxy : bool, default=None
            If True, the Pareto front is computed assuming the y-axis metric
            is to be maximised. If False, it is minimised.
            If None, the function automatically infers the value based on
            the metric.
        **kwargs
            Additional keyword arguments passed to ``spv.pareto_plot``.

        Returns
        -------
        object
            A Pareto plot object returned by ``spv.pareto_plot``. This is
            typically a wrapper that can be rendered or further customised.

        Notes
        -----
        The method relies on the external ``spv.pareto_plot`` function for
        computation and visualisation of the Pareto front.

        The interpretation of "optimal" depends on the ``maxx`` and ``maxy``
        flags, which determine whether each axis is treated as a maximisation
        or minimisation objective.
        """
        if maxx is None:
            maxx = _is_maximise_metric(x_axis)
        if maxy is None:
            maxy = _is_maximise_metric(y_axis)

        plot_obj = spv.pareto_plot(
            self.solution_df,
            x=x_axis,
            y=y_axis,
            maxx=maxx,
            maxy=maxy,
            show_points=show_points,
            height=height,
            theme=theme,
            **kwargs,
        )

        return plot_obj

    def plot_all_metric_pareto_front_pairs(
        self,
        height=4,
        show_points=True,
        theme="whitegrid",
        maxx=None,
        maxy=None,
        cols=3,
        **kwargs,
    ):
        """
        Plot Pareto fronts for all pairs of solution metrics.

        This method generates a grid of subplots, each showing the Pareto
        front for a pairwise combination of performance metrics. It provides
        a comprehensive view of trade-offs between all available objectives.

        Parameters
        ----------
        height : float, default=4
            Height (in inches) allocated to each subplot.
        show_points : bool, default=True
            If True, all solutions are plotted as points in addition to the
            Pareto front in each subplot.
        theme : str, default="whitegrid"
            Visual theme passed to the underlying plotting function.
        maxx : bool or None, default=None
            If True, x-axis metrics are treated as maximisation objectives
            when computing Pareto fronts. If False, they are minimised.
            If None, the direction is inferred per metric.
        maxy : bool or None, default=None
            If True, y-axis metrics are treated as maximisation objectives
            when computing Pareto fronts. If False, they are minimised.
            If None, the direction is inferred per metric.
        cols : int, default=3
            Number of columns in the subplot grid.
        **kwargs
            Additional keyword arguments passed to ``spv.pareto_plot``.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure containing all Pareto front subplots.

        Notes
        -----
        The method constructs all pairwise combinations of the following metrics:
        - "weighted_average"
        - "unweighted_average"
        - "90th_percentile"
        - "max"
        - "proportion_within_coverage_threshold" (included only if available)

        Each subplot visualises the Pareto front for a pair of metrics using
        the ``spv.pareto_plot`` function.

        Subplots are arranged in a grid with a specified number of columns,
        and rows are determined automatically.

        Any unused subplot axes (if the grid is larger than required) are
        removed from the figure.

        The figure is closed before returning to prevent duplicate display
        in some environments (e.g., Jupyter notebooks).
        """
        metrics = [
            "weighted_average",
            "unweighted_average",
            "90th_percentile",
            "max",
        ]

        if self.solution_df.coverage_threshold[0] is not None:
            metrics.append("proportion_within_coverage_threshold")

        metric_pairs = list(itertools.combinations(metrics, 2))
        num_plots = len(metric_pairs)
        cols = cols
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * height, rows * height))
        axes = axes.flatten()

        for idx, (x_metric, y_metric) in enumerate(metric_pairs):
            ax = axes[idx]
            current_maxx = _is_maximise_metric(x_metric) if maxx is None else maxx
            current_maxy = _is_maximise_metric(y_metric) if maxy is None else maxy
            plot_obj = spv.pareto_plot(
                self.solution_df,
                x=x_metric,
                y=y_metric,
                maxx=current_maxx,
                maxy=current_maxy,
                show_points=show_points,
                height=height,
                theme=theme,
                **kwargs,
            )
            _ = plot_obj.on(ax).plot()
            ax.set_title(f"{y_metric} vs {x_metric}")

        for idx in range(num_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.close(fig)
        return fig

    def compute_pareto_front(
        self,
        metrics: list[ParetoMetric],
        id_col: str = "solution_rank",
    ) -> pd.DataFrame:
        """
        Flag which solutions in `df` are Pareto-optimal across `metrics`.

        A solution A dominates solution B if A is at least as good as B on
        every metric, and strictly better on at least one. Anything not
        dominated by another solution in the set is Pareto-optimal.

        Returns a copy of `df` with two extra columns:
        - is_pareto_optimal (bool)
        - dominated_by (list of id_col values that dominate this solution;
        empty for Pareto-optimal solutions)
        """
        self.pareto_metrics = metrics.copy()

        work = self.solution_df.copy().reset_index(drop=True)

        normed = pd.DataFrame({m.column: m.normalise(work[m.column]) for m in metrics})
        values = normed.to_numpy()
        n = len(work)

        is_dominated = np.zeros(n, dtype=bool)
        dominated_by: list[list] = [[] for _ in range(n)]

        for i in range(n):
            # broadcast comparison of row i against all other rows at once
            at_least_as_good = np.all(values <= values[i], axis=1)
            strictly_better = np.any(values < values[i], axis=1)
            dominators = np.where(at_least_as_good & strictly_better)[0]
            dominators = dominators[dominators != i]
            if len(dominators) > 0:
                is_dominated[i] = True
                dominated_by[i] = work.loc[dominators, id_col].tolist()

        work["is_pareto_optimal"] = ~is_dominated
        work["dominated_by"] = dominated_by

        self.solution_df = work

    def pareto_summary(
        self,
        id_col: str = "solution_rank",
        sort_by: str | None = None,
        return_full_df: bool = False,
    ) -> pd.DataFrame:
        """
        Return just the Pareto-optimal solutions, with metric columns
        renamed to their labels, sorted by one metric for readability.
        """
        if "is_pareto_optimal" not in self.solution_df:
            raise ValueError(
                "Pareto front has not been calculated. Please run '.compute_pareto_front() first."
            )

        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        sort_by = sort_by or self.pareto_metrics[0].column
        ascending = next(
            m.direction != "higher_better"
            for m in self.pareto_metrics
            if m.column == sort_by
        )
        # Stable sort so Pareto-optimal solutions tied on `sort_by` keep a
        # consistent order between runs -- see _sort_solutions_by_metric.
        front = front.sort_values(sort_by, ascending=ascending, kind="mergesort")
        cols = [id_col] + [m.column for m in self.pareto_metrics]
        if return_full_df:
            return front.reset_index(drop=True)
        else:
            return front[cols].reset_index(drop=True)

    def describe_tradeoffs(
        self,
        id_col: str = "solution_rank",
        anchor: str | None = None,
    ) -> list[str]:
        """
        Plain-English trade-off statements comparing every Pareto-optimal
        solution back to a single anchor (by default, the solution that's
        best on the first metric in `metrics`).

        e.g. "Solution 6 improves coverage by 4.2 pts vs solution 1, but
        costs 1.8 minutes more in average weighted travel time."
        """
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        if len(front) <= 1:
            return [
                "Only one Pareto-optimal solution found -- no trade-offs to describe."
            ]

        if anchor is None:
            first = self.pareto_metrics[0]
            # Stable sort: the anchor is whichever solution lands first, so
            # an unstable sort could silently narrate a different solution
            # each run whenever the leaders tie.
            front_sorted = front.sort_values(
                first.column,
                ascending=(first.direction != "higher_better"),
                kind="mergesort",
            )
            anchor_row = front_sorted.iloc[0]
        else:
            anchor_row = front[front[id_col] == anchor].iloc[0]

        statements = []
        for _, row in front.iterrows():
            if row[id_col] == anchor_row[id_col]:
                continue
            parts = []
            for m in self.pareto_metrics:
                delta = row[m.column] - anchor_row[m.column]
                if abs(delta) < 1e-9:
                    continue
                better = (delta > 0) == (m.direction == "higher_better")
                if m.direction == "closest_to_target":
                    better = abs(row[m.column] - m.target) < abs(
                        anchor_row[m.column] - m.target
                    )
                verb = "improves" if better else "costs"
                parts.append(f"{verb} {m.label.lower()} by {abs(delta):.2f}")
            if parts:
                statements.append(
                    f"Solution {row[id_col]} vs solution {anchor_row[id_col]}: "
                    + "; ".join(parts)
                )
        return statements

    def describe_tradeoffs_for_stakeholders(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        anchor=None,
        markdown: bool = True,
    ) -> list[str]:
        """
        Stakeholder-facing version of describe_tradeoffs().

        Same underlying comparison, but written for people who don't need
        (or want) to know what "Pareto-optimal" or "dominates" mean. Every
        option is framed relative to a single reference option, in terms of
        what it gains and what it gives up -- no raw deltas, no jargon.

        For best results, write each ParetoMetric.label as a short, plain
        phrase that reads naturally in a sentence (e.g. "average journey
        time" rather than "Weighted average"), and set `unit` / `as_percentage`
        so the numbers come out in a form people recognise (minutes, %, etc).

        name_col: if you have a more meaningful label than the raw id
            (e.g. a short scenario name), pass its column name here.
            Otherwise options are labelled "Option <id>".
        markdown: if True (default), returns Quarto/markdown-friendly bullet
            blocks; if False, returns plain-text paragraphs.

        Returns a list of strings, one per non-reference option, plus a
        leading summary line.
        """
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        if len(front) <= 1:
            return [
                "One option clearly stands out here -- every other combination we "
                "looked at was equalled or beaten on every measure that matters."
            ]

        if anchor is None:
            first = self.pareto_metrics[0]
            # Stable sort: the anchor is whichever solution lands first, so
            # an unstable sort could silently narrate a different solution
            # each run whenever the leaders tie.
            front_sorted = front.sort_values(
                first.column,
                ascending=(first.direction != "higher_better"),
                kind="mergesort",
            )
            anchor_row = front_sorted.iloc[0]
        else:
            anchor_row = front[front[id_col] == anchor].iloc[0]

        def option_name(row) -> str:
            return str(row[name_col]) if name_col else f"Option {row[id_col]}"

        anchor_name = option_name(anchor_row)
        metric_list = ", ".join(m.label.lower() for m in self.pareto_metrics)

        statements = [
            f"We found {len(front)} genuinely different options -- each one is the best "
            f"choice on at least one measure, so no single option wins on everything. "
            f"The trade-offs are between {metric_list}."
        ]

        for _, row in front.iterrows():
            if row[id_col] == anchor_row[id_col]:
                continue
            gains, tradeoffs = [], []
            for m in self.pareto_metrics:
                delta = row[m.column] - anchor_row[m.column]
                if abs(delta) < 1e-9:
                    continue
                better = (delta > 0) == (m.direction == "higher_better")
                if m.direction == "closest_to_target":
                    better = abs(row[m.column] - m.target) < abs(
                        anchor_row[m.column] - m.target
                    )
                (gains if better else tradeoffs).append(m.phrase(delta, better))

            name = option_name(row)
            if markdown:
                block = [f"**{name}** (compared with {anchor_name})"]
                if gains:
                    block.append("- Gains: " + "; ".join(gains))
                if tradeoffs:
                    block.append("- Costs: " + "; ".join(tradeoffs))
                statements.append("\n".join(block))
            else:
                sentence = f"{name}, compared with {anchor_name}: "
                if gains:
                    sentence += " and ".join(gains)
                if gains and tradeoffs:
                    sentence += ", but "
                if tradeoffs:
                    sentence += " and ".join(tradeoffs)
                statements.append(sentence + ".")

        return statements

    def plot_pareto_summary(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        height: float = 5,
        width_multiplier: float = 2,
        theme: str = "whitegrid",
        other_color: str = "lightgray",
        palette: str = "tab10",
        show_value_labels: bool = True,
        title: str | None = None,
        caption: str | None = None,
        ax=None,
    ):
        """
        Plot every solution as a line across all `metrics` at once, each axis
        independently rescaled so "up" always means "better" regardless of
        whether the underlying metric is minimised, maximised, or has a
        target value.

        This is the genuinely multi-objective counterpart to a grid of
        pairwise 2D Pareto plots. A pairwise grid can be misleading once
        you have 3+ objectives: a solution can look non-dominated in every
        single 2D projection shown, while still being dominated once all
        metrics are considered together (because the dominating solution's
        advantage sits on a third axis that isn't in that particular pair).
        Equally, a solution can look beaten in one 2D pair by something
        that is actually worse overall once every metric is weighed in.
        A parallel coordinates plot avoids both failure modes because
        every axis is shown simultaneously on every line.

        Pareto-optimal solutions (is_pareto_optimal == True, as produced by
        pareto.compute_pareto_front) are drawn in colour and labelled;
        dominated solutions are drawn as thin grey background lines, so the
        front stands out against the full brute-forced solution space.

        palette: any matplotlib/seaborn colormap name. Qualitative palettes
            (tab10, Set2, ...) look cleanest for small fronts; if the number
            of Pareto-optimal options exceeds the palette's distinct colours,
            colours are cycled and combined with a different linestyle/marker
            per cycle so lines stay distinguishable. Continuous palettes
            (e.g. "husl", "viridis") sidestep this by generating a fresh
            colour per line regardless of how many there are.
        show_value_labels: if True (default), annotate each point on a
            closest_to_target axis with its signed distance from target
            (e.g. "+0.09"). Turn off for a cleaner chart once you've got
            more than a handful of options and the labels start to crowd.
        title / caption: shown above/below the chart. Sensible stakeholder-
            facing defaults are used if not provided; pass "" to suppress
            either entirely.
        """
        sns.set_style(theme)
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(len(self.pareto_metrics) * width_multiplier, height)
            )
        else:
            fig = ax.figure

        x_positions = list(range(len(self.pareto_metrics)))

        # normalise every metric to 0-1 where 1 is always "better", using the
        # full solution set's range (not just the front) so axes reflect the
        # true spread of what was evaluated
        normed = {}
        raw_range = {}
        for m in self.pareto_metrics:
            col = self.solution_df[m.column]
            lo, hi = col.min(), col.max()
            raw_range[m.column] = (lo, hi)
            span = (hi - lo) or 1.0  # guard against a constant column
            if m.direction == "higher_better":
                normed[m.column] = (col - lo) / span
            elif m.direction == "lower_better":
                normed[m.column] = 1 - (col - lo) / span
            else:  # closest_to_target
                dev = (col - m.target).abs()
                max_dev = dev.max() or 1.0
                normed[m.column] = 1 - dev / max_dev
        normed_df = pd.DataFrame(normed, index=self.solution_df.index)

        # background: dominated solutions, unlabelled
        for idx in rest.index:
            ax.plot(
                x_positions,
                normed_df.loc[idx, [m.column for m in self.pareto_metrics]],
                color=other_color,
                alpha=0.5,
                linewidth=1,
                zorder=1,
            )

        # foreground: the true multi-objective front, colour-coded and labelled
        styles = _colours_and_styles(max(len(front), 1), palette)
        target_metric_cols = {
            m.column for m in self.pareto_metrics if m.direction == "closest_to_target"
        }
        for (colour, linestyle, marker), (_, row) in zip(styles, front.iterrows()):
            label = str(row[name_col]) if name_col else f"Option {row[id_col]}"
            y_vals = normed_df.loc[row.name, [m.column for m in self.pareto_metrics]]
            ax.plot(
                x_positions,
                y_vals,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.2,
                markersize=5,
                label=label,
                zorder=3,
            )
            # closest_to_target axes: position alone can't show which side of the
            # target a solution sits on (above and below collapse to the same
            # distance-from-target height), so label each point with its signed
            # deviation explicitly
            if not show_value_labels:
                continue
            for m in self.pareto_metrics:
                if m.column not in target_metric_cols:
                    continue
                deviation = row[m.column] - m.target
                sign = (
                    "+" if deviation > 0 else ("\u2212" if deviation < 0 else "\u00b1")
                )
                x = x_positions[self.pareto_metrics.index(m)]
                ax.annotate(
                    f"{sign}{abs(deviation):.2f}",
                    (x, y_vals[m.column]),
                    textcoords="offset points",
                    xytext=(8, 0),
                    fontsize=7.5,
                    color=colour,
                    va="center",
                )

        for x, m in zip(x_positions, self.pareto_metrics):
            ax.axvline(x, color="gray", linewidth=0.8, alpha=0.4, zorder=0)
            if m.direction == "closest_to_target":
                # top of axis is always "at target" here; bottom is the largest
                # deviation observed in either direction, so labelling it with a
                # bare number would misleadingly imply a single direction
                label_note = (
                    "diff. from target shown per point"
                    if show_value_labels
                    else "closer = better"
                )
                ax.text(
                    x,
                    1.06,
                    f"target = {m.target:g}\n({label_note})",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="gray",
                )
                ax.text(
                    x,
                    -0.06,
                    "furthest from target \u2192",
                    ha="center",
                    va="top",
                    fontsize=7.5,
                    color="gray",
                )
            else:
                lo, hi = raw_range[m.column]
                worse_val, better_val = (
                    (lo, hi) if m.direction == "higher_better" else (hi, lo)
                )
                ax.text(
                    x,
                    -0.06,
                    f"{worse_val:.2f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="gray",
                )
                ax.text(
                    x,
                    1.06,
                    f"{better_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="gray",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [m.label for m in self.pareto_metrics], rotation=15, ha="right"
        )
        ax.set_yticks([])
        ax.set_ylim(-0.14, 1.16)
        ax.set_ylabel("Better \u2191 (rescaled per metric)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)

        if title is None:
            title = "Comparing the shortlisted options"
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=14, loc="left")

        if caption is None:
            caption = (
                "How to read this chart: each coloured line is one option that isn't beaten on every "
                "measure by any other -- there's no single best choice. Higher always means better on "
                "every axis, even though the measures use different units, so a line near the top of an "
                "axis is doing well on that measure and a line near the bottom is doing poorly. Grey lines "
                "are options that were beaten everywhere by another combination, shown only for context. "
                "For a 'target' measure, height shows how close an option is to the ideal value; the small "
                "label next to each point shows whether it sits above (+) or below (\u2212) that target."
            )
        if caption:
            import textwrap

            wrapped = "\n".join(textwrap.wrap(caption, width=105))
            fig.text(
                0.01,
                -0.02,
                wrapped,
                ha="left",
                va="top",
                fontsize=8.5,
                color="dimgray",
                wrap=True,
            )

        fig.tight_layout()
        return fig

    def plot_pareto_facets(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        show_site_names: bool = True,
        ncols: int = 1,
        height_per_row: float = 4.0,
        width_multiplier: float = 2.2,
        theme: str = "whitegrid",
        other_color: str = "lightgray",
        palette: str = "tab10",
        show_value_labels: bool = False,  # Off by default in small multiples to prevent clutter
        show_dominated: bool = True,
        title: str | None = None,
        wrap_at: int = 110,
        show_raw_labels: bool = True,
        rank_scope: str = "all",
    ):
        """
        Plot each non-dominated (Pareto-optimal) solution on its own subplot facet.

        Each facet highlights one specific Pareto-optimal solution while fading the
        other front solutions to the background, allowing side-by-side trade-off analysis.

        Subplots are dynamically annotated with:
        - The solution ID & custom name
        - Included sites (if `sites_col` is provided)
        - The count of overall alternatives this specific solution dominates
        - Its key relative strengths and trade-offs compared to the Pareto average, each
          annotated with its rank (e.g. "Weighted average travel time (1st of 18)")

        Parameters
        ----------
        rank_scope : str, default "all"
            Scope used to rank each metric for the Strengths/Sacrifices annotations.
            "all" ranks a solution's metric against every enumerated solution
            (dominated + Pareto-optimal). "pareto_front" ranks it only against the
            other non-dominated solutions shown in this plot.
        """
        if rank_scope not in ("all", "pareto_front"):
            raise ValueError(
                f"rank_scope must be one of 'all' or 'pareto_front', got {rank_scope!r}"
            )
        sns.set_style(theme)
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()

        n_front = len(front)
        if n_front == 0:
            raise ValueError("No Pareto-optimal solutions found to plot.")

        # Calculate grid dimensions
        nrows = math.ceil(n_front / ncols)
        fig_width = len(self.pareto_metrics) * width_multiplier * ncols
        fig_height = nrows * height_per_row

        # wrap_at is tuned for a full-width single-column figure; with ncols>1 each
        # facet is proportionally narrower, so scale the wrap width down to match or
        # the wrapped site list can overflow past its own subplot
        effective_wrap_at = max(20, wrap_at // ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
        )
        # Ensure axes is a flat array even if 1D or single subplot
        axes = np.array(axes).ravel()

        # Normalise every metric to 0-1 (where 1 is always "better")
        normed = {}
        raw_range = {}
        for m in self.pareto_metrics:
            col = self.solution_df[m.column]
            lo, hi = col.min(), col.max()
            raw_range[m.column] = (lo, hi)
            span = (hi - lo) or 1.0
            if m.direction == "higher_better":
                normed[m.column] = (col - lo) / span
            elif m.direction == "lower_better":
                normed[m.column] = 1 - (col - lo) / span
            else:  # closest_to_target
                dev = (col - m.target).abs()
                max_dev = dev.max() or 1.0
                normed[m.column] = 1 - dev / max_dev

        normed_df = pd.DataFrame(normed, index=self.solution_df.index)
        x_positions = list(range(len(self.pareto_metrics)))

        # Compute average performance across the Pareto front for trade-off reference
        front_normed = normed_df.loc[front.index]
        front_mean = front_normed.mean()

        # Rank each metric (per rank_scope) for the Strengths/Sacrifices annotations.
        # normed_df values are already oriented so higher = better regardless of the
        # metric's original direction, so a single ranking rule covers all of them.
        rank_source = normed_df if rank_scope == "all" else front_normed
        rank_n = len(rank_source)
        rank_df = rank_source.rank(method="min", ascending=False)

        # Build a persistent mapping of front-solution index to styling (color, marker, style)
        styles = _colours_and_styles(max(n_front, 1), palette)
        style_map = {idx: style for style, (idx, _) in zip(styles, front.iterrows())}

        target_metric_cols = {
            m.column for m in self.pareto_metrics if m.direction == "closest_to_target"
        }

        # Generate each subplot
        for i, (idx, current_row) in enumerate(front.iterrows()):
            ax = axes[i]

            # Plot the faint background dominated space (optional)
            if show_dominated:
                for rest_idx in rest.index:
                    ax.plot(
                        x_positions,
                        normed_df.loc[
                            rest_idx, [m.column for m in self.pareto_metrics]
                        ],
                        color=other_color,
                        alpha=0.15,  # Kept very faint to keep visual focus on the front
                        linewidth=0.8,
                        zorder=1,
                    )

            # Plot the other Pareto-optimal solutions as faded lines
            for other_idx, other_row in front.iterrows():
                if other_idx == idx:
                    continue
                colour, linestyle, marker = style_map[other_idx]
                ax.plot(
                    x_positions,
                    normed_df.loc[other_idx, [m.column for m in self.pareto_metrics]],
                    color=colour,
                    linestyle=linestyle,
                    marker=marker,
                    alpha=0.15,  # Faded background
                    linewidth=1.2,
                    markersize=3,
                    zorder=2,
                )

            # Highlight the current primary solution
            curr_colour, curr_linestyle, curr_marker = style_map[idx]
            curr_y_vals = normed_df.loc[idx, [m.column for m in self.pareto_metrics]]

            ax.plot(
                x_positions,
                curr_y_vals,
                color=curr_colour,
                linestyle=curr_linestyle,
                marker=curr_marker,
                linewidth=3.5,
                markersize=7,
                zorder=4,
            )

            # --- ADDED: Annotate each point on the highlighted line with its RAW value ---
            if show_raw_labels:
                for m in self.pareto_metrics:
                    raw_val = current_row[m.column]
                    x = x_positions[self.pareto_metrics.index(m)]
                    y = curr_y_vals[m.column]

                    # Smart-format: Keep integers clean, float decimals concise, and add commas
                    if isinstance(raw_val, (int, np.integer)):
                        val_str = f"{raw_val:,}"
                    elif isinstance(raw_val, (float, np.floating)):
                        val_str = (
                            f"{raw_val:,.2f}".rstrip("0").rstrip(".")
                            if not raw_val.is_integer()
                            else f"{int(raw_val):,}"
                        )
                    else:
                        val_str = str(raw_val)

                    ax.annotate(
                        val_str,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),  # Positioned 8 points directly above the marker
                        fontsize=8,
                        color="black",  # Dark text is easiest to read in a white bubble
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                        zorder=5,  # Keeps labels on top of all lines
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="white",
                            ec=curr_colour,
                            alpha=0.85,
                            lw=1,
                        ),
                    )

            # Value annotations if toggled (useful for target tracking)
            if show_value_labels:
                for m in self.pareto_metrics:
                    if m.column not in target_metric_cols:
                        continue
                    deviation = current_row[m.column] - m.target
                    sign = (
                        "+"
                        if deviation > 0
                        else ("\u2212" if deviation < 0 else "\u00b1")
                    )
                    x = x_positions[self.pareto_metrics.index(m)]
                    ax.annotate(
                        f"{sign}{abs(deviation):.2f}",
                        (x, curr_y_vals[m.column]),
                        textcoords="offset points",
                        xytext=(8, 3),
                        fontsize=7,
                        color=curr_colour,
                        fontweight="bold",
                        va="bottom",
                    )

            # Draw gray vertical guidelines
            for x, m in zip(x_positions, self.pareto_metrics):
                ax.axvline(x, color="gray", linewidth=0.8, alpha=0.3, zorder=0)

            # --- Subplot Title & Meta Calculations ---

            # Dominance math: A candidate is dominated by us if our metrics are >= theirs
            # on all dimensions, and strictly > on at least one.
            curr_norm_vals = normed_df.loc[idx]
            dominates_mask = (curr_norm_vals >= normed_df).all(axis=1) & (
                curr_norm_vals > normed_df
            ).any(axis=1)
            num_dominated = dominates_mask.sum()

            def _label_with_rank(m):
                rank_val = int(rank_df.loc[idx, m.column])
                rank_str = f"{rank_val}{_get_ordinal_suffix(rank_val)} of {rank_n}"
                return f"{m.label} ({rank_str})"

            # Relative Strengths and Weaknesses (vs the average of the Pareto Front)
            strengths = []
            trade_offs = []
            for m in self.pareto_metrics:
                val = curr_norm_vals[m.column]
                avg = front_mean[m.column]
                diff = val - avg
                if diff > 0.05:  # Noticeably above front average
                    strengths.append(_label_with_rank(m))
                elif diff < -0.05:  # Noticeably below front average
                    trade_offs.append(_label_with_rank(m))

            # Fallbacks if metrics are entirely clustered around the mean
            if not strengths:
                strengths = [
                    _label_with_rank(m)
                    for m in self.pareto_metrics
                    if curr_norm_vals[m.column] >= curr_norm_vals.median()
                ][:2]
            if not trade_offs:
                trade_offs = [
                    _label_with_rank(m)
                    for m in self.pareto_metrics
                    if curr_norm_vals[m.column] < curr_norm_vals.median()
                ][:2]

            # Label
            label = (
                str(current_row[name_col])
                if name_col
                else f"Option {current_row[id_col]}"
            )

            if num_dominated > 0:
                label = label + f": beats {num_dominated} designs"

            # Site summary
            sites_str = ""
            if show_site_names:
                sites_val = current_row["site_names"]
                if isinstance(sites_val, (list, set, tuple)):
                    sites_str = ", ".join(map(str, sites_val))
                else:
                    sites_str = str(sites_val)
                # Wrap long site lists to keep layout clean
                sites_str = textwrap.fill(sites_str, width=effective_wrap_at)

            # Compile information into a neat, blocky subplot title
            label = label.replace(" ", r"\ ")
            title_lines = [rf"$\bf{{{label}}}$"]
            # Now add additional info to the title, not in bold
            if sites_str:
                title_lines.append(f"Included: {sites_str}")

            meta_parts = []
            if strengths:
                meta_parts.append(f"---\nStrengths: {', '.join(strengths[:2])}")
            if trade_offs:
                meta_parts.append(f"\nSacrifices: {', '.join(trade_offs[:2])}")

            title_lines.append("".join(meta_parts))
            ax.set_title(
                "\n".join(title_lines),
                fontsize=8.5,
                loc="left",
                pad=20,  # Leaves headroom for the raw-value labels near the top of the axes
                linespacing=1.3,
            )

        # Clean up structural axes ticks & labels
        for idx_ax, ax in enumerate(axes[:n_front]):
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [m.label for m in self.pareto_metrics],
                rotation=15,
                ha="right",
                fontsize=8,
            )

            ax.tick_params(labelbottom=True)

            # Y-axis labelling
            if idx_ax % ncols == 0:
                ax.set_ylabel("Better \u2192 (scaled)", fontsize=9)
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(["Worse", "Mid", "Best"], fontsize=8)
            else:
                ax.set_ylabel("")
                ax.yaxis.set_tick_params(labelleft=False)

            # Label bounds on the first row's axes for clean context
            if idx_ax < ncols:
                for x, m in zip(x_positions, self.pareto_metrics):
                    if m.direction == "closest_to_target":
                        ax.text(
                            x,
                            1.08,
                            f"tgt={m.target:g}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="gray",
                        )
                    else:
                        lo, hi = raw_range[m.column]
                        better_val = hi if m.direction == "higher_better" else lo
                        ax.text(
                            x,
                            1.08,
                            f"{better_val:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="gray",
                        )

            ax.set_ylim(-0.1, 1.15)

        # Hide any empty unused subplot cells
        for j in range(n_front, len(axes)):
            fig.delaxes(axes[j])

        # Universal title
        if title is None:
            title = "Individual Analysis of Pareto-Optimal Choices"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

        fig.tight_layout()
        return fig
