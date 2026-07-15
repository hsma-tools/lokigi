import sweetpareto.vis as spv
import itertools
import matplotlib.pyplot as plt
import math
from typing import Literal


##################################
# MARK: Pareto
##################################
class ParetoMixin:
    def plot_simple_pareto_front_pairs(
        self,
        x_axis: Literal[
            "weighted_average",
            "unweighted_average",
            "90th_percentile",
            "max",
            "proportion_within_coverage_threshold",
        ] = "weighted_average",
        y_axis: Literal[
            "weighted_average",
            "unweighted_average",
            "90th_percentile",
            "max",
            "proportion_within_coverage_threshold",
        ] = "max",
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
        x_axis : {"weighted_average", "unweighted_average", "90th_percentile", \
                "max", "proportion_within_coverage_threshold"}, \
                default="weighted_average"
            Column name representing the metric to plot on the x-axis.
        y_axis : {"weighted_average", "unweighted_average", "90th_percentile", \
                "max", "proportion_within_coverage_threshold"}, \
                default="max"
            Column name representing the metric to plot on the y-axis.
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
            maxx = x_axis == "proportion_within_coverage_threshold"
        if maxy is None:
            maxy = y_axis == "proportion_within_coverage_threshold"

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
            current_maxx = (
                (x_metric == "proportion_within_coverage_threshold")
                if maxx is None
                else maxx
            )
            current_maxy = (
                (y_metric == "proportion_within_coverage_threshold")
                if maxy is None
                else maxy
            )
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
