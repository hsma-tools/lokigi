from __future__ import annotations

import itertools

import math
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import sweetpareto.vis as spv
import textwrap

from lokigi.plot_utils import _add_plot_caption
from lokigi.utils import _colours_and_styles, _is_maximise_metric, _get_ordinal_suffix

from lokigi.multiobjective import Metric


def _format_site_diff_suffix(added, removed):
    """
    Compact "+X, Y; -Z" representation of a site-name diff, for a legend
    label or subplot title where `show_solutions_summary()`'s full "Sites
    added"/"Sites removed" columns would be too verbose. `added`/
    `removed` are the comma-joined strings `_resolve_site_diff()` returns
    per row; an empty string in either means nothing changed in that
    direction (or, for the reference row itself under `diff_against=
    "rank_1"`, nothing changed at all).

    The two clauses are joined with "; ", not ", " -- with commas
    throughout, "+Site_A, Site_C, -Site_B, Site_D" reads as four
    separately-signed items rather than two added and two removed; the
    semicolon marks where the added list ends and the removed one begins.
    """
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return "; ".join(parts)


def _wrap_legend_label(label, wrap_at, newline):
    """Wrap a legend label at `wrap_at` characters (None = leave it alone),
    joining lines with `newline` -- "\\n" for matplotlib, "<br>" for plotly."""
    if not wrap_at:
        return label
    return newline.join(textwrap.wrap(label, width=wrap_at, break_long_words=False))


def _site_names_list(sites_val):
    """A `site_names` cell as a list of strings, whether it holds a
    list/set/tuple of names or a single pre-joined value."""
    if isinstance(sites_val, (list, set, tuple)):
        return [str(site) for site in sites_val]
    return [str(sites_val)]


def _format_raw_value(raw_val):
    """Smart-format a raw metric value for an on-chart label: keep integers
    clean, float decimals concise, and add thousands separators."""
    if isinstance(raw_val, (int, np.integer)):
        return f"{raw_val:,}"
    if isinstance(raw_val, (float, np.floating)):
        return (
            f"{raw_val:,.2f}".rstrip("0").rstrip(".")
            if not raw_val.is_integer()
            else f"{int(raw_val):,}"
        )
    return str(raw_val)


# matplotlib linestyle/marker -> plotly equivalent, for the styles
# _colours_and_styles() can hand back
_PLOTLY_DASH = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}
_PLOTLY_SYMBOL = {
    "o": "circle",
    "s": "square",
    "^": "triangle-up",
    "D": "diamond",
    "v": "triangle-down",
    "P": "cross",
    "X": "x",
}

# Empty hole at the middle of a radar chart, in normalised (0-1) units. The
# worst value seen on a metric normalises to exactly 0, so without a hole
# every option that's worst on some metric pinches in to a single shared
# centre point, which reads as a drawing glitch rather than "worst".
_RADAR_HOLE = 0.25

# Radial range padding on the interactive radar, in normalised units -- see
# radial_range in _plot_pareto_radar_interactive
_PLOTLY_RADIAL_PAD = 0.04


def _radar_angles(n):
    """Spoke angles (radians) for n metrics, starting at 12 o'clock once
    `_style_radar_axes` has set the polar offset/direction."""
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def _closed(values):
    """Repeat the first point at the end so a radar polygon closes."""
    values = list(values)
    return values + values[:1]


def _style_radar_axes(ax, angles, labels, fontsize, pad):
    """Shared polar-axes setup for the radar plots: first metric at 12
    o'clock running clockwise, a 0-1 radial range with an empty centre, no
    radial tick numbers (the scale is rescaled per metric, so they mean
    nothing), and spoke labels pushed away from the plot on either side."""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_rorigin(-_RADAR_HOLE)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis="x", pad=pad)
    # Centred labels on the left/right spokes overhang back across the
    # plot; anchor them on the side facing away from it instead
    for tick_label, angle in zip(ax.get_xticklabels(), angles):
        x = np.cos(np.pi / 2 - angle)
        tick_label.set_horizontalalignment(
            "left" if x > 0.1 else ("right" if x < -0.1 else "center")
        )


def _radar_offset(angle, distance, turn=0.0):
    """(dx, dy) in points for nudging a vertex label off its marker on the
    spoke at `angle`: radially outward when `turn=0`, or rotated `turn`
    radians anticlockwise from there (e.g. +/-pi/2 to sit beside the spoke
    rather than on it)."""
    display_angle = np.pi / 2 - angle + turn
    return (distance * np.cos(display_angle), distance * np.sin(display_angle))


def _signed_deviation(deviation):
    sign = "+" if deviation > 0 else ("\u2212" if deviation < 0 else "\u00b1")
    return f"{sign}{abs(deviation):.2f}"


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
        metrics: list[Metric],
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

        For best results, write each Metric.label as a short, plain
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

    def _require_pareto_front(self):
        if "is_pareto_optimal" not in self.solution_df:
            raise ValueError(
                "Pareto front has not been calculated. Please run '.compute_pareto_front() first."
            )

    def _pareto_normalised(self, scale_to="all"):
        """
        Rescale every Pareto metric to 0-1 where 1 is always "better".

        `scale_to="all"` (the default) uses the full solution set's range, not
        just the front, so axes reflect the true spread of what was
        evaluated. `scale_to="pareto_front"` uses only the Pareto-optimal
        solutions' range, which stretches the front across the whole axis;
        dominated solutions can then fall outside 0-1 (a Pareto-optimal
        solution never does). They are left as they are here, so dominance
        comparisons stay exact, and callers draw them with `np.clip(.., 0, 1)`.

        Returns `(normed_df, raw_range)`: `normed_df` is indexed like
        `solution_df` with one column per metric, and `raw_range` maps each
        metric column to the raw `(min, max)` it was scaled over.
        """
        scoped = (
            self.solution_df[self.solution_df["is_pareto_optimal"]]
            if scale_to == "pareto_front"
            else self.solution_df
        )
        normed = {}
        raw_range = {}
        for m in self.pareto_metrics:
            col = self.solution_df[m.column]
            lo, hi = scoped[m.column].min(), scoped[m.column].max()
            raw_range[m.column] = (lo, hi)
            span = (hi - lo) or 1.0  # guard against a constant column
            if m.direction == "higher_better":
                normed[m.column] = (col - lo) / span
            elif m.direction == "lower_better":
                normed[m.column] = 1 - (col - lo) / span
            else:  # closest_to_target
                dev = (col - m.target).abs()
                max_dev = (scoped[m.column] - m.target).abs().max() or 1.0
                normed[m.column] = 1 - dev / max_dev
        return pd.DataFrame(normed, index=self.solution_df.index), raw_range

    @staticmethod
    def _require_scale_to(scale_to):
        if scale_to not in ("all", "pareto_front"):
            raise ValueError(
                f"scale_to must be one of 'all' or 'pareto_front', got {scale_to!r}"
            )

    def _scale_phrase(self, scale_to):
        """The set of options a radar's best/worst and ranks are over, as a
        phrase to follow "of" / "across"."""
        if scale_to == "pareto_front":
            n = int(self.solution_df["is_pareto_optimal"].sum())
            return f"the {n} Pareto-optimal options"
        return f"all {len(self.solution_df)} options"

    @staticmethod
    def _pareto_option_label(row, id_col, name_col, site_diff):
        """Legend label for one option: `name_col`/"Option <id>", plus a
        compact site diff suffix when `site_diff` is given."""
        label = str(row[name_col]) if name_col else f"Option {row[id_col]}"
        if site_diff is not None:
            suffix = _format_site_diff_suffix(
                site_diff["added"].loc[row.name], site_diff["removed"].loc[row.name]
            )
            if suffix:
                label = f"{label} ({suffix})"
        return label

    def _pareto_facet_title(
        self,
        idx,
        row,
        normed_df,
        front_mean,
        rank_df,
        rank_n,
        id_col,
        name_col,
        show_site_names,
        site_diff,
        wrap_at,
        meta_wrap_at=None,
    ):
        """
        Multi-line facet title for one Pareto-optimal option: bold name (and
        how many designs it beats), included sites, optional site diff, and
        its strengths/sacrifices relative to the front average, each with
        its rank. `wrap_at` wraps the site and diff lines; the
        strengths/sacrifices lines are only wrapped if `meta_wrap_at` is
        given.
        """
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
        label = str(row[name_col]) if name_col else f"Option {row[id_col]}"

        if num_dominated > 0:
            label = label + f": beats {num_dominated} designs"

        # Site summary
        sites_str = ""
        if show_site_names:
            sites_str = ", ".join(_site_names_list(row["site_names"]))
            # Wrap long site lists to keep layout clean
            sites_str = textwrap.fill(sites_str, width=wrap_at)

        # Compile information into a neat, blocky subplot title
        label = label.replace(" ", r"\ ")
        title_lines = [rf"$\bf{{{label}}}$"]
        # Now add additional info to the title, not in bold
        if sites_str:
            title_lines.append(f"Included: {sites_str}")

        if site_diff is not None:
            diff_suffix = _format_site_diff_suffix(
                site_diff["added"].loc[idx], site_diff["removed"].loc[idx]
            )
            if diff_suffix:
                diff_line = f"Diff (vs {site_diff['label']}): {diff_suffix}"
                title_lines.append(textwrap.fill(diff_line, width=wrap_at))

        # Narrow facets (the radar ones) also wrap these, or long metric labels
        # run on into the neighbouring subplot
        def _wrap_meta(line):
            return textwrap.fill(line, width=meta_wrap_at) if meta_wrap_at else line

        meta_parts = []
        if strengths:
            strengths_line = f"Strengths: {', '.join(strengths[:2])}"
            meta_parts.append(f"---\n{_wrap_meta(strengths_line)}")
        if trade_offs:
            trade_offs_line = f"Sacrifices: {', '.join(trade_offs[:2])}"
            meta_parts.append(f"\n{_wrap_meta(trade_offs_line)}")

        title_lines.append("".join(meta_parts))
        return "\n".join(title_lines)

    def plot_pareto_summary(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        diff_against: str | None = None,
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
        diff_against: str or None, default None
            If given, appends a compact "+added, -removed" site-name diff
            to each Pareto-optimal line's legend label -- the same
            per-row diff `show_solutions_summary(diff_against=...)`
            computes, see `_resolve_site_diff` for the exact values
            accepted ("default", "rank_1", "previous_rank",
            "required_sites") and what each one means. `None` (the
            default) leaves labels as `name_col`/`id_col` only, matching
            prior behaviour. Since this is computed over just the
            Pareto-optimal `front` rows, "previous_rank" diffs each
            option against the next-best Pareto-optimal option, not
            against a dominated solution that happens to sit between them
            in rank.
        title / caption: shown above/below the chart. Sensible stakeholder-
            facing defaults are used if not provided; pass "" to suppress
            either entirely.
        """
        sns.set_style(theme)
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()
        site_diff = self._resolve_site_diff(front, diff_against) if diff_against else None

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(len(self.pareto_metrics) * width_multiplier, height)
            )
        else:
            fig = ax.figure

        x_positions = list(range(len(self.pareto_metrics)))

        # Rotated at a shallow angle, a long label overhangs a fair way to the
        # left of its tick. tight_layout() (called below) treats that overhang
        # as space the axes must yield, and since the overhang is a roughly
        # fixed number of inches regardless of the figure's own width, a
        # narrow figure (few metrics, small width_multiplier) can end up
        # ceding nearly all of it -- collapsing the plot itself down to a
        # sliver. Wrapping long labels onto multiple lines keeps that overhang
        # bounded to roughly one width_multiplier's worth of characters.
        metric_label_wrap_width = max(int(width_multiplier * 9), 12)
        metric_labels = [
            textwrap.fill(m.label, width=metric_label_wrap_width)
            for m in self.pareto_metrics
        ]

        normed_df, raw_range = self._pareto_normalised()

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
            label = self._pareto_option_label(row, id_col, name_col, site_diff)
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
        ax.set_xticklabels(metric_labels, rotation=15, ha="right")
        ax.set_yticks([])
        ax.set_ylim(-0.14, 1.16)
        ax.set_ylabel("Better \u2191 (rescaled per metric)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)

        if title is None:
            title = "Comparing the shortlisted options"
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=14, loc="left")

        default_caption = (
            "How to read this chart: each coloured line is one option that isn't beaten on every "
            "measure by any other -- there's no single best choice. Higher always means better on "
            "every axis, even though the measures use different units, so a line near the top of an "
            "axis is doing well on that measure and a line near the bottom is doing poorly. Grey lines "
            "are options that were beaten everywhere by another combination, shown only for context. "
            "For a 'target' measure, height shows how close an option is to the ideal value; the small "
            "label next to each point shows whether it sits above (+) or below (\u2212) that target."
        )
        _add_plot_caption(fig, caption, default_caption, width=105)

        fig.tight_layout()
        return fig

    def plot_pareto_facets(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        show_site_names: bool = True,
        diff_against: str | None = None,
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
        diff_against : str or None, default None
            If given, adds a "Diff (vs <reference>): +added, -removed"
            line to each facet's title -- the same per-row site-name diff
            `show_solutions_summary(diff_against=...)` computes, see
            `_resolve_site_diff` for the exact values accepted ("default",
            "rank_1", "previous_rank", "required_sites"). `None` (the
            default) omits it, matching prior behaviour. Computed over
            just the Pareto-optimal `front` rows, so "previous_rank"
            diffs each facet against the next-best Pareto-optimal option,
            not a dominated solution that happens to sit between them in
            rank.
        """
        if rank_scope not in ("all", "pareto_front"):
            raise ValueError(
                f"rank_scope must be one of 'all' or 'pareto_front', got {rank_scope!r}"
            )
        sns.set_style(theme)
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()
        site_diff = self._resolve_site_diff(front, diff_against) if diff_against else None

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

        # Rotated at a shallow angle, a long label overhangs a fair way to the
        # left of its tick, which tight_layout() (called below) treats as
        # space the axes must yield -- see the matching comment in
        # plot_pareto_summary for why that can collapse a narrow facet down
        # to a sliver. Wrapping bounds the overhang to roughly one
        # width_multiplier's worth of characters per line.
        metric_label_wrap_width = max(int(width_multiplier * 9), 12)
        metric_labels = [
            textwrap.fill(m.label, width=metric_label_wrap_width)
            for m in self.pareto_metrics
        ]

        normed_df, raw_range = self._pareto_normalised()
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
                    x = x_positions[self.pareto_metrics.index(m)]
                    y = curr_y_vals[m.column]

                    ax.annotate(
                        _format_raw_value(current_row[m.column]),
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

            ax.set_title(
                self._pareto_facet_title(
                    idx,
                    current_row,
                    normed_df,
                    front_mean,
                    rank_df,
                    rank_n,
                    id_col,
                    name_col,
                    show_site_names,
                    site_diff,
                    effective_wrap_at,
                ),
                fontsize=8.5,
                loc="left",
                pad=20,  # Leaves headroom for the raw-value labels near the top of the axes
                linespacing=1.3,
            )

        # Clean up structural axes ticks & labels
        for idx_ax, ax in enumerate(axes[:n_front]):
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                metric_labels,
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

    def _require_radar_metrics(self):
        n = len(self.pareto_metrics)
        if n < 3:
            raise ValueError(
                f"A radar chart needs at least 3 Pareto metrics to draw a shape, but "
                f"compute_pareto_front() was given {n}. Use plot_pareto_summary() or "
                f"plot_pareto_facets() instead."
            )

    def _radar_spoke_detail(self, m, raw_range, scale_to="all"):
        """Short note of what the outer ring and centre mean on one spoke.
        It names the set the range was taken over (see `_pareto_normalised`),
        otherwise "best"/"worst" reads as best/worst of the shapes shown."""
        if m.direction == "closest_to_target":
            return f"target = {m.target:g}"
        lo, hi = raw_range[m.column]
        worse_val, better_val = (lo, hi) if m.direction == "higher_better" else (hi, lo)
        return (
            f"best {_format_raw_value(better_val)}, worst {_format_raw_value(worse_val)}\n"
            f"of {self._scale_phrase(scale_to)}"
        )

    def _resolve_initially_shown(self, front, initially_shown, id_col):
        """Index labels of the `front` rows to show initially. Validates
        `initially_shown` whether or not the plot is interactive, since a bad
        value is a mistake either way."""
        if isinstance(initially_shown, str):
            if initially_shown == "all":
                return set(front.index)
            if initially_shown == "first":
                return set(front.index[:1])
            raise ValueError(
                f"initially_shown must be 'all', 'first' or a list of {id_col} values, "
                f"got {initially_shown!r}"
            )
        try:
            wanted = list(initially_shown)
        except TypeError:
            raise ValueError(
                f"initially_shown must be 'all', 'first' or a list of {id_col} values, "
                f"got {initially_shown!r}"
            ) from None
        if not wanted:
            raise ValueError(
                "initially_shown is empty, so no options would be shown. Pass "
                "'all', 'first' or at least one id."
            )
        valid = set(front[id_col])
        missing = [i for i in wanted if i not in valid]
        if missing:
            dominated = [
                i for i in missing if i in set(self.solution_df[id_col]) - valid
            ]
            detail = (
                f" ({', '.join(map(str, dominated))} exist but "
                f"{'is' if len(dominated) == 1 else 'are'} not Pareto-optimal)"
                if dominated
                else ""
            )
            raise ValueError(
                f"initially_shown has {id_col} values that aren't Pareto-optimal "
                f"options: {', '.join(map(str, missing))}{detail}. "
                f"Valid values: {', '.join(map(str, sorted(valid)))}."
            )
        return set(front.index[front[id_col].isin(wanted)])

    def plot_pareto_radar(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        diff_against: str | None = None,
        interactive: bool = False,
        initially_shown="all",
        legend_wrap_at: int | None = None,
        scale_to: str = "all",
        figsize: tuple = (9, 7),
        theme: str = "whitegrid",
        other_color: str = "lightgray",
        palette: str = "tab10",
        show_dominated: bool = False,
        fill: bool = True,
        show_value_labels: bool = False,
        title: str | None = None,
        caption: str | None = None,
        ax=None,
    ):
        """
        Plot every Pareto-optimal solution as a shape on a radar (spider)
        chart, one spoke per metric, each rescaled so further from the
        centre always means "better" regardless of whether the underlying
        metric is minimised, maximised, or has a target value.

        This shows the same data as `plot_pareto_summary()`, with the axes
        wrapped into a circle instead of laid side by side. That's
        analytically no better (arguably worse), but the "bigger shape =
        doing well" reading is often easier for stakeholders to pick up
        than a parallel coordinates plot.

        Area is not a fair overall score. A shape's area depends on which
        metrics sit next to each other: reordering the metrics passed to
        `compute_pareto_front()` can change which option *looks* biggest
        without any value changing. Compare options spoke by spoke. The
        default caption says so.

        Needs at least 3 metrics. With fewer, use `plot_pareto_summary()`.

        Parameters
        ----------
        id_col, name_col, diff_against
            As in `plot_pareto_summary()`.
        interactive : bool, default False
            If True, returns an interactive Plotly figure: hover a point
            for its raw value and rank, click a legend entry to hide that
            option, or double-click one to show it alone.
        initially_shown : "all", "first" or list of ids, default "all"
            Which options are visible when an interactive figure first
            opens. The rest start greyed out in the legend, one click from
            being shown -- useful when a large front makes the full overlay
            a tangle. "all" shows every option, "first" only the first
            option in the legend (the best-ranked Pareto-optimal one), and
            a list, tuple or set picks specific options by their `id_col`
            value (e.g. `[1, 17]`), even when `name_col` is set. Ignored if
            `interactive=False`, though an invalid value still raises.
        legend_wrap_at : int, optional
            Wrap legend labels onto multiple lines at this many characters.
            `None` (the default) leaves them on one line. Worth setting when
            `diff_against` makes labels long, since a wide legend squeezes
            the chart. Only the legend wraps: the interactive hover title
            keeps the full label on one line.
        scale_to : {"all", "pareto_front"}, default "all"
            Which options each spoke's best and worst values are taken
            from, and so what the outer ring and centre mean. "all" scales
            over every option compared, dominated ones included, so the
            chart shows how good the front is against everything that was
            evaluated. "pareto_front" scales over just the Pareto-optimal
            options, which stretches them across the whole spoke and makes
            the differences between them easier to see, at the cost of
            saying nothing about how they compare with the rest. The spoke
            labels, hover ranks and caption name whichever is used. With
            "pareto_front", dominated outlines (`show_dominated=True`) can
            fall outside the chart and are clipped to its edge.
        figsize : tuple, default (9, 7)
            Passed to `plt.subplots()` if `ax` is not supplied. Ignored if
            `interactive=True`.
        theme : str, default "whitegrid"
            Seaborn style. Ignored if `interactive=True`.
        other_color : str, default "lightgray"
            Colour of the dominated solutions when `show_dominated=True`.
        palette : str, default "tab10"
            As in `plot_pareto_summary()`.
        show_dominated : bool, default False
            If True, also draw every dominated solution as a thin
            background outline. Off by default, unlike
            `plot_pareto_summary()`: a brute-forced solution set can hold
            hundreds of them, and on a radar they pile up into a grey disc
            that hides the shapes that matter.
        fill : bool, default True
            Lightly shade the inside of each option's shape.
        show_value_labels : bool, default False
            Label each point on a closest_to_target spoke with its signed
            distance from target (e.g. "+0.09"), since distance from the
            centre alone can't show which side of the target it's on. Off
            by default, unlike `plot_pareto_summary()`: every option's
            label lands on the same spoke, so beyond a handful of options
            they pile into an unreadable cluster. Interactive figures
            always show this on hover instead.
        title, caption : str, optional
            Shown above/below the chart. Stakeholder-facing defaults are
            used if not provided; pass "" to suppress either.
        ax : matplotlib.axes.Axes, optional
            Existing *polar* axes to plot onto, e.g. from
            `plt.subplots(subplot_kw={"projection": "polar"})`. Ignored if
            `interactive=True`.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
        """
        self._require_pareto_front()
        self._require_radar_metrics()
        self._require_scale_to(scale_to)
        if not interactive and ax is not None and ax.name != "polar":
            raise ValueError(
                "plot_pareto_radar() needs polar axes, e.g. from "
                "plt.subplots(subplot_kw={'projection': 'polar'})."
            )

        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()
        site_diff = self._resolve_site_diff(front, diff_against) if diff_against else None
        shown_index = self._resolve_initially_shown(front, initially_shown, id_col)
        if legend_wrap_at is not None and (
            isinstance(legend_wrap_at, bool)
            or not isinstance(legend_wrap_at, int)
            or legend_wrap_at < 1
        ):
            raise ValueError(
                f"legend_wrap_at must be a positive whole number of characters or "
                f"None, got {legend_wrap_at!r}"
            )
        normed_df, raw_range = self._pareto_normalised(scale_to)
        scale_phrase = self._scale_phrase(scale_to)
        metric_cols = [m.column for m in self.pareto_metrics]
        styles = _colours_and_styles(max(len(front), 1), palette)
        has_target = any(
            m.direction == "closest_to_target" for m in self.pareto_metrics
        )

        if title is None:
            title = "Comparing the shortlisted options"

        default_caption = (
            "How to read this chart: each coloured shape is one option that isn't beaten on "
            "every measure by any other -- there's no single best choice. Each spoke is one "
            "measure, rescaled so that further from the centre is always better: the outer "
            f"ring is the best value on that measure across {scale_phrase}"
            + (
                " compared, not just the ones shown in colour"
                if scale_to == "all"
                else ", so it only compares the shapes shown with each other"
            )
            + ", and the inner ring the worst. A bigger shape is not automatically better "
            "overall -- its area depends on which measures happen to sit next to each "
            "other -- so compare options spoke by spoke."
        )
        if show_dominated:
            default_caption += (
                " Grey outlines are options that were beaten everywhere by another "
                "combination, shown only for context"
                + (
                    "."
                    if scale_to == "all"
                    else "; any that fall outside the chart are drawn at its edge."
                )
            )
        if has_target:
            default_caption += (
                " For a 'target' measure, distance from the centre shows how close an option "
                "is to the ideal value"
            )
            if interactive:
                default_caption += (
                    "; hover over a point to see whether it sits above (+) or below "
                    "(\u2212) that target."
                )
            elif show_value_labels:
                default_caption += (
                    "; the small label next to each point shows whether it sits above (+) "
                    "or below (\u2212) that target."
                )
            else:
                default_caption += "."
        if interactive and len(shown_index) < len(front):
            default_caption += (
                " Only some options are shown to start with -- click a greyed-out "
                "option in the legend to add it."
            )

        if interactive:
            return self._plot_pareto_radar_interactive(
                front,
                rest,
                site_diff,
                normed_df,
                raw_range,
                styles,
                id_col,
                name_col,
                other_color,
                show_dominated,
                shown_index,
                legend_wrap_at,
                scale_to,
                fill,
                title,
                default_caption if caption is None else caption,
            )

        sns.set_style(theme)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        else:
            fig = ax.figure

        angles = _radar_angles(len(self.pareto_metrics))
        closed_angles = _closed(angles)
        spoke_labels = [
            f"{textwrap.fill(m.label, width=20)}\n"
            f"({self._radar_spoke_detail(m, raw_range, scale_to)})"
            for m in self.pareto_metrics
        ]

        if show_dominated:
            for idx in rest.index:
                ax.plot(
                    closed_angles,
                    # Clipped: under scale_to="pareto_front" a dominated
                    # option can fall outside the chart
                    _closed(np.clip(normed_df.loc[idx, metric_cols], 0, 1)),
                    color=other_color,
                    alpha=0.5,
                    linewidth=0.8,
                    zorder=1,
                )

        for option_i, ((colour, linestyle, marker), (idx, row)) in enumerate(
            zip(styles, front.iterrows())
        ):
            r = _closed(normed_df.loc[idx, metric_cols])
            ax.plot(
                closed_angles,
                r,
                color=colour,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.2,
                markersize=5,
                label=_wrap_legend_label(
                    self._pareto_option_label(row, id_col, name_col, site_diff),
                    legend_wrap_at,
                    "\n",
                ),
                zorder=3,
                clip_on=False,
            )
            if fill:
                ax.fill(closed_angles, r, color=colour, alpha=0.08, zorder=2)
            if not show_value_labels:
                continue
            for angle, m in zip(angles, self.pareto_metrics):
                if m.direction != "closest_to_target":
                    continue
                # Every option's point sits on the same spoke, so beside it
                # rather than along it, alternating sides so neighbouring
                # options' labels don't stack on each other
                side = np.pi / 2 if option_i % 2 == 0 else -np.pi / 2
                ax.annotate(
                    _signed_deviation(row[m.column] - m.target),
                    (angle, normed_df.loc[idx, m.column]),
                    textcoords="offset points",
                    xytext=_radar_offset(angle, 9, turn=side),
                    fontsize=7.5,
                    color=colour,
                    ha="center",
                    va="center",
                    zorder=4,
                    # Every point is inside the axes by construction, but
                    # matplotlib's inside-the-axes check doesn't account for
                    # the polar rorigin hole, so leaving clipping on silently
                    # drops these labels
                    annotation_clip=False,
                )

        _style_radar_axes(ax, angles, spoke_labels, fontsize=9, pad=14)
        ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), frameon=False, fontsize=9)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=30, loc="left")
        _add_plot_caption(fig, caption, default_caption, width=110)

        fig.tight_layout()
        return fig

    def _plot_pareto_radar_interactive(
        self,
        front,
        rest,
        site_diff,
        normed_df,
        raw_range,
        styles,
        id_col,
        name_col,
        other_color,
        show_dominated,
        shown_index,
        legend_wrap_at,
        scale_to,
        fill,
        title,
        caption,
    ):
        metric_cols = [m.column for m in self.pareto_metrics]
        theta = _closed(
            [
                textwrap.fill(m.label, width=20).replace("\n", "<br>")
                + "<br><span style='font-size:10px;color:gray'>"
                + self._radar_spoke_detail(m, raw_range, scale_to).replace("\n", "<br>")
                + "</span>"
                for m in self.pareto_metrics
            ]
        )
        # Ranked over the same set the spokes are scaled over
        rank_source = (
            normed_df.loc[front.index] if scale_to == "pareto_front" else normed_df
        )
        rank_df = rank_source.rank(method="min", ascending=False)
        scale_phrase = self._scale_phrase(scale_to)

        fig = go.Figure()

        if show_dominated:
            for idx in rest.index:
                fig.add_trace(
                    go.Scatterpolar(
                        r=_closed(np.clip(normed_df.loc[idx, metric_cols], 0, 1)),
                        theta=theta,
                        mode="lines",
                        line=dict(color=mcolors.to_hex(other_color), width=0.8),
                        opacity=0.5,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Plotly hands a hover inside overlapping fills to the topmost (last
        # drawn) one containing the cursor, so draw the largest shapes first:
        # where several overlap, the hover then names the smallest -- usually
        # the outline nearest the cursor -- rather than whichever option
        # happens to come last. legendrank keeps the legend in option order.
        #
        # The empty centre comes from starting the radial range below 0 (r=0
        # lands at the same 20% of the radius as the static version's
        # rorigin), not from polar.hole: with a hole, plotly's hover zone is
        # a ring whose seam runs along 0 degrees, so every vertex on an
        # east-pointing spoke (any 4 or 8 metrics) was unhoverable. The top
        # is padded past 1 because plotly also drops the hover once the
        # cursor's radius leaves the range, which left vertices exactly on
        # the outer ring only hoverable on their inner half.
        radial_range = [
            -_RADAR_HOLE * (1 + _PLOTLY_RADIAL_PAD),
            1 + _PLOTLY_RADIAL_PAD,
        ]

        def drawn_area(idx):
            # Polygon area up to a constant factor, since the spokes are
            # evenly spaced. Offset by the empty centre, so areas compare the
            # shapes as drawn.
            radii = normed_df.loc[idx, metric_cols].to_numpy() - radial_range[0]
            return float(np.sum(radii * np.roll(radii, -1)))

        def short_name(row):
            return str(row[name_col]) if name_col else f"Option {row[id_col]}"

        options = list(zip(styles, front.iterrows()))
        draw_order = sorted(
            range(len(options)), key=lambda i: -drawn_area(options[i][1][0])
        )

        for option_i in draw_order:
            (colour, linestyle, marker), (idx, row) = options[option_i]
            # One per line, since a joined list of real site names makes for
            # a very wide tooltip. Shown both on each vertex's hover and on
            # the shape's own (fill) hover.
            sites_html = ""
            if "site_names" in row.index:
                sites_html = "<br><br><b>Sites</b><br>" + "<br>".join(
                    f"\u2022 {site}" for site in _site_names_list(row["site_names"])
                )
            name = self._pareto_option_label(row, id_col, name_col, site_diff)
            customdata = []
            for m in self.pareto_metrics:
                rank_val = int(rank_df.loc[idx, m.column])
                context = f"{rank_val}{_get_ordinal_suffix(rank_val)} of {scale_phrase}"
                if m.direction == "closest_to_target":
                    context += (
                        f"; {_signed_deviation(row[m.column] - m.target)} vs "
                        f"target {m.target:g}"
                    )
                # Options tied on a metric share the exact same vertex, and
                # plotly can only name one of them on hover -- so whichever
                # it picks, list the others
                tied = [
                    short_name(other)
                    for other_idx, other in front.iterrows()
                    if other_idx != idx and np.isclose(other[m.column], row[m.column])
                ]
                if tied:
                    context += f"<br>Same value: {', '.join(tied)}"
                customdata.append(
                    [m.label, _format_raw_value(row[m.column]), context, sites_html, name]
                )
            r, g, b, _ = mcolors.to_rgba(colour)
            fig.add_trace(
                go.Scatterpolar(
                    r=_closed(normed_df.loc[idx, metric_cols]),
                    theta=theta,
                    customdata=_closed(customdata),
                    # Only the legend label wraps; the hover shows `name` from
                    # customdata so its title stays on one line
                    name=_wrap_legend_label(name, legend_wrap_at, "<br>"),
                    legendrank=option_i + 1,
                    # "legendonly" starts the trace hidden but keeps its
                    # legend entry, so one click brings it back. Hidden
                    # traces don't take part in hover.
                    visible=True if idx in shown_index else "legendonly",
                    mode="lines+markers",
                    line=dict(
                        color=mcolors.to_hex(colour),
                        width=2.2,
                        dash=_PLOTLY_DASH[linestyle],
                    ),
                    marker=dict(symbol=_PLOTLY_SYMBOL[marker], size=7),
                    fill="toself" if fill else "none",
                    fillcolor=f"rgba({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f}, 0.1)",
                    # Hovering inside a shape, away from any vertex, shows
                    # the option and its sites. A fill hover displays a
                    # scalar `text` (it can't use hovertemplate); `text` is
                    # never drawn, since mode has no "text"
                    hoveron="points+fills" if fill else "points",
                    text=f"<b>{name}</b>{sites_html}",
                    hovertemplate=(
                        "<b>%{customdata[4]}</b><br>"
                        "%{customdata[0]}: %{customdata[1]}<br>"
                        "%{customdata[2]}%{customdata[3]}<extra></extra>"
                    ),
                )
            )

        # Explicit pixel sizing throughout: plotly's 450px default height
        # leaves only a sliver for the chart once the title and caption
        # margins come out of it, which shrinks the circle, scrolls the
        # legend, and runs the bottom spoke label into the caption. Spoke
        # labels are drawn outside the polar area, so it's inset from the
        # sides to keep them clear of the legend, and the caption sits a
        # fixed distance below it rather than a fraction of the plot height.
        # Plotly can't re-wrap an annotation to the figure's width, so the
        # caption is pre-wrapped narrow enough (~560px) to fit a typical
        # notebook output, and shifted left into the 80px margin (which the
        # left spoke's label needs, so it stays) to start near the edge.
        chart_height = 560
        caption_lines = textwrap.wrap(caption, width=100) if caption else []
        # The right margin holds the legend, which plotly otherwise shifts
        # left over the chart once its labels outgrow the default 80px
        # (diff_against labels do, unless wrapped by legend_wrap_at)
        longest_label_line = max(
            (
                len(line)
                for t in fig.data
                if t.showlegend is not False
                for line in t.name.split("<br>")
            ),
            default=0,
        )
        margin = dict(
            t=135,
            b=95 + 16 * len(caption_lines),
            l=80,
            r=80 if longest_label_line <= 12 else int(7.5 * longest_label_line) + 50,
        )
        layout = dict(
            template="plotly_white",
            height=chart_height + margin["t"] + margin["b"],
            polar=dict(
                domain=dict(x=[0.15, 0.85]),
                # See radial_range above for why this, not polar.hole
                radialaxis=dict(
                    range=radial_range,
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    showticklabels=False,
                ),
                # Invisible ticks, purely to push the spoke labels clear of
                # markers sitting on the outer ring
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    ticks="outside",
                    ticklen=10,
                    tickcolor="rgba(0,0,0,0)",
                ),
            ),
            legend=dict(x=1.02, y=1),
            margin=margin,
            # Down from plotly's 20px, so a vertex's detail only takes over
            # when the cursor is actually on it; elsewhere the shape it's
            # inside answers instead of the nearest vertex of any option
            hoverdistance=10,
        )
        if title:
            layout["title"] = dict(text=f"<b>{title}</b>", x=0.02, xanchor="left")
        if caption_lines:
            layout["annotations"] = [
                dict(
                    text="<br>".join(caption_lines),
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0,
                    xshift=-50,
                    yshift=-80,  # clears the bottom spoke's (up to 4-line) label
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    align="left",
                    font=dict(size=11, color="dimgray"),
                )
            ]
        fig.update_layout(**layout)
        return fig

    def plot_pareto_radar_facets(
        self,
        id_col: str = "solution_rank",
        name_col: str | None = None,
        show_site_names: bool = True,
        diff_against: str | None = None,
        ncols: int = 3,
        size_per_facet: float = 4.0,
        theme: str = "whitegrid",
        other_color: str = "lightgray",
        palette: str = "tab10",
        show_dominated: bool = False,
        show_raw_labels: bool = True,
        show_value_labels: bool = False,
        title: str | None = None,
        caption: str | None = None,
        wrap_at: int | None = None,
        rank_scope: str = "all",
        scale_to: str = "all",
    ):
        """
        Plot each Pareto-optimal solution on its own radar (spider) chart.

        The radar counterpart to `plot_pareto_facets()`: each facet
        highlights one option as a filled shape, with the other
        Pareto-optimal options as faint outlines behind it, and the same
        title block (included sites, how many designs it beats, and its
        strengths and sacrifices with their ranks). Further from the
        centre is always better on every spoke.

        As with `plot_pareto_radar()`, a shape's area depends on the order
        of the metrics, so it is not a fair overall score. Needs at least 3
        metrics; with fewer, use `plot_pareto_facets()`.

        Parameters
        ----------
        id_col, name_col, show_site_names, diff_against, theme, other_color,
        palette, show_raw_labels, show_value_labels, rank_scope
            As in `plot_pareto_facets()`.
        wrap_at : int, optional
            Characters per line in each facet's title block. Unlike
            `plot_pareto_facets()` this is per facet, not per figure, so it
            isn't divided by `ncols`. Defaults to roughly what fits in
            `size_per_facet` inches.
        ncols : int, default 3
            Facets per row (capped at the number of options). Defaults
            higher than `plot_pareto_facets()` because square radar panels
            tile side by side more naturally than wide line charts.
        size_per_facet : float, default 4.0
            Width of each facet in inches. Each row is taller than this,
            to leave room for the title block.
        show_dominated : bool, default False
            If True, also draw every dominated solution as a very faint
            outline. Off by default -- see `plot_pareto_radar()`.
        title : str, optional
            Figure title. A default is used if not provided; pass "" to
            suppress it.
        caption : str, optional
            Explanatory text below the chart. A stakeholder-facing default
            is used if not provided; pass "" to suppress it.
        scale_to : {"all", "pareto_front"}, default "all"
            Which options each spoke is scaled over, and so what the outer
            ring and centre mean -- see `plot_pareto_radar()`. Separate from
            `rank_scope`, which only sets the set the Strengths/Sacrifices
            ranks in each title are over.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if rank_scope not in ("all", "pareto_front"):
            raise ValueError(
                f"rank_scope must be one of 'all' or 'pareto_front', got {rank_scope!r}"
            )
        self._require_scale_to(scale_to)
        self._require_pareto_front()
        self._require_radar_metrics()
        sns.set_style(theme)
        front = self.solution_df[self.solution_df["is_pareto_optimal"]].copy()
        rest = self.solution_df[~self.solution_df["is_pareto_optimal"]].copy()
        site_diff = self._resolve_site_diff(front, diff_against) if diff_against else None

        n_front = len(front)
        if n_front == 0:
            raise ValueError("No Pareto-optimal solutions found to plot.")

        ncols = min(ncols, n_front)
        nrows = math.ceil(n_front / ncols)
        if wrap_at is None:
            # ~15 characters per inch at the title's 8.5pt
            wrap_at = max(20, int(size_per_facet * 15))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            # Taller than wide to leave room for the title block; polar axes
            # are always square, so any more than that is dead space
            figsize=(ncols * size_per_facet, nrows * size_per_facet * 1.2),
            subplot_kw={"projection": "polar"},
        )
        axes = np.array(axes).ravel()

        normed_df, raw_range = self._pareto_normalised(scale_to)
        metric_cols = [m.column for m in self.pareto_metrics]
        front_normed = normed_df.loc[front.index]
        front_mean = front_normed.mean()
        rank_source = normed_df if rank_scope == "all" else front_normed
        rank_n = len(rank_source)
        rank_df = rank_source.rank(method="min", ascending=False)

        styles = _colours_and_styles(max(n_front, 1), palette)
        style_map = {idx: style for style, (idx, _) in zip(styles, front.iterrows())}

        angles = _radar_angles(len(self.pareto_metrics))
        closed_angles = _closed(angles)
        spoke_labels = [textwrap.fill(m.label, width=14) for m in self.pareto_metrics]

        for i, (idx, current_row) in enumerate(front.iterrows()):
            ax = axes[i]

            if show_dominated:
                for rest_idx in rest.index:
                    ax.plot(
                        closed_angles,
                        _closed(np.clip(normed_df.loc[rest_idx, metric_cols], 0, 1)),
                        color=other_color,
                        alpha=0.15,
                        linewidth=0.8,
                        zorder=1,
                    )

            for other_idx in front.index:
                if other_idx == idx:
                    continue
                colour, linestyle, _ = style_map[other_idx]
                ax.plot(
                    closed_angles,
                    _closed(normed_df.loc[other_idx, metric_cols]),
                    color=colour,
                    linestyle=linestyle,
                    alpha=0.2,
                    linewidth=1.2,
                    zorder=2,
                )

            curr_colour, curr_linestyle, curr_marker = style_map[idx]
            curr_r = normed_df.loc[idx, metric_cols]
            ax.plot(
                closed_angles,
                _closed(curr_r),
                color=curr_colour,
                linestyle=curr_linestyle,
                marker=curr_marker,
                linewidth=3,
                markersize=6,
                zorder=4,
                clip_on=False,
            )
            ax.fill(closed_angles, _closed(curr_r), color=curr_colour, alpha=0.2, zorder=3)

            for angle, m in zip(angles, self.pareto_metrics):
                if show_raw_labels:
                    ax.annotate(
                        _format_raw_value(current_row[m.column]),
                        (angle, curr_r[m.column]),
                        textcoords="offset points",
                        xytext=_radar_offset(angle, 11),
                        fontsize=7.5,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=5,
                        annotation_clip=False,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="white",
                            ec=curr_colour,
                            alpha=0.85,
                            lw=1,
                        ),
                    )
                if show_value_labels and m.direction == "closest_to_target":
                    ax.annotate(
                        _signed_deviation(current_row[m.column] - m.target),
                        (angle, curr_r[m.column]),
                        textcoords="offset points",
                        # Beside the spoke, clear of the raw-value box
                        # sitting radially outward
                        xytext=_radar_offset(angle, 10, turn=-np.pi / 2),
                        fontsize=7,
                        color=curr_colour,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        annotation_clip=False,
                    )

            # Extra pad clears the raw-value boxes sitting just outside r=1
            _style_radar_axes(ax, angles, spoke_labels, fontsize=8, pad=20)
            ax.set_title(
                self._pareto_facet_title(
                    idx,
                    current_row,
                    normed_df,
                    front_mean,
                    rank_df,
                    rank_n,
                    id_col,
                    name_col,
                    show_site_names,
                    site_diff,
                    wrap_at,
                    meta_wrap_at=wrap_at,
                ),
                fontsize=8.5,
                loc="left",
                pad=28,
                linespacing=1.3,
            )

        for j in range(n_front, len(axes)):
            fig.delaxes(axes[j])

        if title is None:
            title = "Individual Analysis of Pareto-Optimal Choices"
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

        default_caption = (
            "How to read this chart: each panel highlights one option that isn't beaten on "
            "every measure by any other, with the rest of the shortlist as faint outlines "
            "behind it. Further from the centre is always better on every spoke: the outer ring "
            f"is the best value on that measure across {self._scale_phrase(scale_to)}"
            + (", not just the shortlist" if scale_to == "all" else "")
            + ", and the centre the worst. The boxed numbers are "
            "the option's actual values. A bigger shape is not automatically better overall -- "
            "its area depends on which measures happen to sit next to each other."
        )
        _add_plot_caption(fig, caption, default_caption, width=40 * ncols)

        fig.tight_layout()
        return fig
