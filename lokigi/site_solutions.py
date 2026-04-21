import pandas as pd
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from warnings import warn
import numpy as np
import math
import plotly.express as px
from lokigi.utils import _safe_evaluate
import sweetpareto.vis as spv
import itertools
from typing import Literal


class EvaluatedCombination:
    """
    Container for results and summary metrics of an evaluated site combination.

    This class stores the outcome of evaluating a candidate solution (i.e.,
    a set of selected sites) against a demand dataset, and computes a range
    of summary statistics based on the minimum cost (e.g., travel time or
    distance) from demand locations to the selected sites.

    Parameters
    ----------
    solution_type : str
        Label describing the type of solution (e.g., optimisation method or scenario).
    site_names : list of str
        Names of the selected sites in the solution.
    site_indices : list of int
        Indices of the selected sites corresponding to the original site list.
    evaluated_combination_df : pandas.DataFrame
        DataFrame containing evaluation results for each demand point. Must include:
        - "min_cost": Minimum cost from each demand point to the selected sites.
        - "within_threshold": Boolean indicator of whether the demand point is
          within the specified coverage threshold.
        - A demand column specified by ``site_problem._demand_data_demand_col``.
    site_problem : object
        Object containing problem configuration and metadata, including the name
        of the demand column via ``_demand_data_demand_col``.
    coverage_threshold : float, optional
        Threshold used to determine whether a demand point is considered covered.
        If provided, used to compute the proportion of demand points within coverage.

    Attributes
    ----------
    solution_type : str
        Type or label of the solution.
    site_names : list of str
        Names of the selected sites.
    site_indices : list of int
        Indices of the selected sites.
    evaluated_combination_df : pandas.DataFrame
        DataFrame containing per-demand-point evaluation results.
    site_problem : object
        Problem definition object.
    coverage_threshold : float or None
        Coverage threshold used in evaluation.

    weighted_average : float
        Demand-weighted average of the minimum cost.
    unweighted_average : float
        Simple (unweighted) average of the minimum cost.
    percentile_90th : float
        90th percentile of the minimum cost distribution.
    max : float
        Maximum minimum cost across all demand points.
    proportion_within_coverage_threshold : float
        Proportion of demand points that fall within the coverage threshold.

    Notes
    -----
    The weighted average is computed using demand values as weights.
    """

    def __init__(
        self,
        solution_type,
        site_names,
        site_indices,
        evaluated_combination_df,
        site_problem,
        coverage_threshold=None,
    ):
        self.solution_type = solution_type
        self.site_names = site_names
        self.site_indices = site_indices
        self.evaluated_combination_df = evaluated_combination_df
        self.site_problem = site_problem
        self.weighted_average = np.average(
            self.evaluated_combination_df["min_cost"],
            weights=self.evaluated_combination_df[
                self.site_problem._demand_data_demand_col
            ],
        )
        self.unweighted_average = np.average(self.evaluated_combination_df["min_cost"])
        self.percentile_90th = np.percentile(
            self.evaluated_combination_df["min_cost"], q=90
        )
        self.max = np.max(self.evaluated_combination_df["min_cost"])
        self.coverage_threshold = coverage_threshold
        self.proportion_within_coverage_threshold = np.sum(
            self.evaluated_combination_df["within_threshold"]
        ) / len(self.evaluated_combination_df)

    def show_result_df(self):
        return self.evaluated_combination_df

    def return_solution_metrics(self):
        # Return weighted average
        return {
            "site_names": self.site_names,
            "site_indices": self.site_indices,
            "coverage_threshold": self.coverage_threshold,
            "weighted_average": self.weighted_average,
            "unweighted_average": self.unweighted_average,
            "90th_percentile": self.percentile_90th,
            "max": self.max,
            "proportion_within_coverage_threshold": self.proportion_within_coverage_threshold,
            "problem_df": self.evaluated_combination_df,
        }


class SiteSolutionSet:
    """
    Container for a set of evaluated site selection solutions.

    This class stores and provides convenient access to a collection of
    candidate solutions from a brute-force or heuristic search,
    along with their associated evaluation metrics. It supports returning
    and plotting details of the best-performing solutions.

    Parameters
    ----------
    solution_df : pandas.DataFrame
        DataFrame containing one row per evaluated solution. Typically includes:
        - "site_indices": Indices of selected sites for the solution.
        - One or more objective/metric columns (e.g., "weighted_average",
          "unweighted_average", "90th_percentile", etc.).
        The DataFrame is reset to a zero-based index upon initialisation.
    site_problem : object
        The originating problem instance used to generate and evaluate
        the solutions.
    objectives : str or list of str
        Objective(s) used to evaluate and rank the solutions (e.g.,
        "weighted_average", "mclp").
    n_sites : int, optional
        Number of sites selected in each solution (e.g., ``p`` in a p-median
        or p-center problem).

    Attributes
    ----------
    solution_df : pandas.DataFrame
        DataFrame of evaluated solutions with metrics.
    site_problem : object
        Problem definition associated with the solutions.
    objectives : str or list of str
        Objective(s) used in evaluation.
    n_sites : int or None
        Number of sites in each solution.

    Notes
    -----
    Solutions are typically pre-sorted before being passed to this class
    (e.g., by objective value and tie-breakers). The optional ``rank_on``
    argument in methods allows overriding this ordering dynamically.

    The structure of ``solution_df`` is assumed to be consistent with the
    outputs of the optimisation or search routine that generated it.
    """

    def __init__(self, solution_df, site_problem, objectives, n_sites=None):
        """
        Initialise a SiteSolutionSet instance.

        Parameters
        ----------
        solution_df : pandas.DataFrame
            DataFrame containing evaluated solutions. Each row represents a
            candidate solution and typically includes columns such as
            "site_indices", "site_names", and one or more objective metrics.
            The index is reset to a zero-based integer index on initialisation.
        site_problem : object
            The originating problem instance used to generate and evaluate
            the solutions.
        objectives : str or list of str
            Objective(s) used to evaluate the solutions.
        n_sites : int, optional
            Number of sites selected in each solution.

        Notes
        -----
        The input DataFrame is copied with its index reset to ensure consistent
        positional indexing for downstream operations.
        """
        self.solution_df = solution_df.reset_index(drop=True)
        self.site_problem = site_problem
        self.objectives = objectives
        self.n_sites = n_sites

    def show_solutions(self, rounding=2):
        """
        Return the solution DataFrame with rounded values.

        Parameters
        ----------
        rounding : int, default=2
            Number of decimal places to round numeric columns to.

        Returns
        -------
        pandas.DataFrame
            The solution DataFrame with numeric values rounded to the specified
            precision.

        Notes
        -----
        This method does not modify the underlying DataFrame; it returns a
        rounded copy.
        """
        if rounding is None:
            return self.solution_df
        else:
            return round(self.solution_df, rounding)

    def return_best_combination_details(self, rank_on=None, top_n=1):
        """
        Return details of the top-ranked solution(s).

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, solutions are
            sorted in ascending order before selecting the top entries.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.
        top_n : int, default=1
            Number of top solutions to return.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the top ``top_n`` solutions, including all
            available columns. The index is reset in the returned DataFrame.

        Notes
        -----
        Sorting is performed in ascending order, so lower values are assumed
        to represent better solutions for the specified ranking metric.
        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on).head(top_n).reset_index()
        else:
            return self.solution_df.head(top_n).reset_index()

    def return_best_combination_site_indices(self, rank_on=None):
        """
        Return the site indices for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, the solution
            with the lowest value in this column is selected.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_indices" column for the best solution.
            Typically a list or array of site indices.

        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on)["site_indices"].iloc[0]
        else:
            return self.solution_df["site_indices"].iloc[0]

    def return_best_combination_site_names(self, rank_on=None):
        """
        Return the site names for the best-performing solution.

        Parameters
        ----------
        rank_on : str, optional
            Column name to sort the solutions by. If provided, the solution
            with the lowest value in this column is selected.
            If None, the existing order of ``solution_df``, which is based on the
            objective selected, is used.

        Returns
        -------
        object
            The value of the "site_indices" column for the best solution.
            Typically a list or array of site indices.

        """
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on)["site_names"].iloc[0]
        else:
            return self.solution_df["site_names"].iloc[0]

    def plot_travel_time_distribution(
        self,
        top_n=1,
        bottom_n=None,
        rank_on=None,
        secondary_ranking="max",
        title="default",
        height=None,
        height_per_plot=250,
        compare_to_best=False,
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
            Number of top-ranked solutions to include. If ``rank_on`` is provided,
            solutions are sorted before selection; otherwise, the existing order,
            which is based on the objective chosen for solving, is used.
        bottom_n : int, optional
            Number of bottom-ranked solutions to include. If provided, these are
            appended to the selected top solutions.
        rank_on : str, optional
            Column name used to rank solutions. Sorting is performed in ascending
            order using this column, with ``secondary_ranking`` as a tie-breaker.
            If None, no additional sorting is applied.
        secondary_ranking : str, default="max"
            Secondary column used for tie-breaking when sorting by ``rank_on``.
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
        if rank_on is not None:
            solutions_sorted = self.solution_df.sort_values(
                [rank_on, secondary_ranking]
            ).reset_index(drop=True)
        else:
            solutions_sorted = self.solution_df.reset_index(drop=True)

        filtered_dfs = []

        if top_n is not None:
            filtered_dfs.append(solutions_sorted.head(top_n))

        if bottom_n is not None:
            filtered_dfs.append(temp_bottom=solutions_sorted.tail(bottom_n))

        solutions_filtered = pd.concat(filtered_dfs)

        # Convert columns to be compared to tuples to allow dupliate detection
        solutions_filtered["site_indices_comp"] = solutions_filtered[
            "site_indices"
        ].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

        solutions_filtered["site_names_comp"] = solutions_filtered["site_names"].apply(
            lambda x: tuple(x) if isinstance(x, np.ndarray) else x
        )

        solutions_filtered = solutions_filtered.drop_duplicates(
            subset=["site_indices_comp", "site_names_comp"]
        )

        dfs = []
        if compare_to_best:
            best_df = solutions_filtered["problem_df"].iloc[0]

        for index, row in solutions_filtered.iterrows():
            df = row["problem_df"].copy()
            df["site_indices"] = str(row["site_indices"])
            df["weighted_average"] = row["weighted_average"]
            df["unweighted_average"] = row["unweighted_average"]
            df["max"] = row["max"]
            df["90th_percentile"] = row["90th_percentile"]
            if compare_to_best:
                df["min_cost_diff"] = df["min_cost"] - best_df["min_cost"].values
            dfs.append(df)

        dfs = pd.concat(dfs)
        dfs["label"] = (
            "Sites: "
            + dfs["site_indices"].astype(str)
            + " | Weighted Avg: "
            + dfs["weighted_average"].round(2).astype(str)
            + " | Unweighted Avg: "
            + dfs["unweighted_average"].round(2).astype(str)
            + " | P90: "
            + dfs["90th_percentile"].round(2).astype(str)
            + " | Max: "
            + dfs["max"].round(2).astype(str)
        )

        fig = px.histogram(
            dfs,
            x="min_cost_diff" if compare_to_best else "min_cost",
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
                annotation_text="Best",
                annotation_position="top right",
                annotation_yshift=4,
            )
        else:
            for i, (_, row) in enumerate(solutions_filtered.iterrows()):
                subplot_row = top_n - i  # Plotly facet_row numbers bottom-to-top

                # Weighted average — label sits at the top of the line
                fig.add_vline(
                    x=row["weighted_average"],
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
                    x=row["unweighted_average"],
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
                    x=row["90th_percentile"],
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
                    x=row["max"],
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

            if rank_on is not None:
                fig.update_layout(
                    title=f"Distribution of Travel Times ({title_str} Solutions by {rank_on.replace('_', ' ').title()})"
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

    def summary_table(self):
        pass

    def plot_best_combination(
        self,
        rank_on=None,
        title="default",
        show_all_locations=True,
        plot_site_allocation=False,
        plot_regions_not_meeting_threshold=False,
        cmap=None,
        chosen_site_colour="black",
        unchosen_site_colour="grey",
        legend_loc="upper right",
    ):
        """
        Plot a map of the best-performing site combination.

        This method visualises the spatial performance of the best solution in
        the solution set, including travel costs, site allocations, or coverage
        relative to a threshold. Regions are coloured according to the selected
        metric, and candidate site locations can be overlaid.

        Parameters
        ----------
        rank_on : str, optional
            Column name used to rank solutions. If provided, the solution with
            the lowest value in this column is selected. If None, the first row
            of ``solution_df`` is used.
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
        legend_loc: str, default="upper right"
            Adjust position of the legend within the main plot. Accepts standard
            matplotlib positioning strings.

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

        When plotting site locations, only GeoPandas-based candidate sites
        are currently supported.

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

        if rank_on is not None:
            solution = self.solution_df.sort_values(rank_on).head().reset_index()
        else:
            solution = self.solution_df.head().reset_index()

        nearest_site_travel_gdf = pd.merge(
            self.site_problem.region_geometry_layer,
            solution["problem_df"][0],
            left_on=self.site_problem._region_geometry_layer_common_col,
            right_on=self.site_problem._demand_data_id_col,
        )

        # Choose appropriate colourmap if a colourmap is not provided
        if cmap is None:
            if plot_site_allocation:
                # If plotting site allocation, use a categorical (qualitative) colourmap
                cmap = "Set2"
            elif plot_regions_not_meeting_threshold:
                cmap = "Set1"

                cmap_obj = plt.get_cmap(cmap)
                # Extract the first two colors correctly
                if hasattr(cmap_obj, "colors"):
                    colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
                else:
                    colors = [cmap_obj(0.0), cmap_obj(1.0)]

                discrete_cmap = ListedColormap(colors)
            else:
                # If plotting travel time/distance/other cost, use a sequential colourmap
                cmap = "Blues"

        if plot_site_allocation:
            ax = nearest_site_travel_gdf.plot(
                "selected_site",
                legend=True,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
                legend_kwds={"loc": legend_loc},
            )
        elif plot_regions_not_meeting_threshold:
            # nearest_site_travel_gdf["within_threshold"] = nearest_site_travel_gdf[
            #     "within_threshold_str"
            # ].apply(
            #     lambda x: (
            #         f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nof nearest site"
            #         if x is True
            #         else f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nfrom nearest site"
            #     )
            # )
            ax = nearest_site_travel_gdf.plot(
                "within_threshold",
                legend=False,
                cmap=discrete_cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
                vmin=0,
                vmax=1,
            )

            # Build custom legend
            patches = [
                mpatches.Patch(
                    color=colors[0],
                    label=f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}",
                ),
                mpatches.Patch(
                    color=colors[1],
                    label=f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}",
                ),
            ]
            ax.legend(handles=patches, title="Coverage Status", loc=legend_loc)
        else:
            ax = nearest_site_travel_gdf.plot(
                "min_cost",
                legend=True,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
            )

        if self.site_problem._candidate_sites_type == "geopandas":
            selected_sites = self.site_problem.candidate_sites.iloc[
                solution.site_indices.iloc[0]
            ]

            if show_all_locations:
                all_site_points = self.site_problem.candidate_sites.plot(
                    ax=ax, color=unchosen_site_colour, markersize=30, alpha=0.3
                )

            selected_site_points = selected_sites.plot(
                ax=ax, color=chosen_site_colour, markersize=60
            )

            for x, y, label in zip(
                selected_sites.geometry.x,
                selected_sites.geometry.y,
                selected_sites[self.site_problem._candidate_sites_candidate_id_col],
            ):
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(10, 3),
                    textcoords="offset points",
                    bbox=dict(facecolor="white"),
                )

        cx.add_basemap(
            ax,
            crs=nearest_site_travel_gdf.crs.to_string(),
        )

        ax.axis("off")

        if title is not None:
            if title == "default":
                if self.objectives == "mclp":
                    title = plt.title(
                        f"Best solution for {self.n_sites} sites \nCoverage within threshold of {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}: {solution['proportion_within_coverage_threshold'].values[0]:.1%} \nUnweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )

                elif self.objectives in [
                    "simple_p_median",
                    "hybrid_simple_p_median",
                ]:
                    title = plt.title(
                        f"Best solution for {self.n_sites} sites \nUnweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )

                else:
                    title = plt.title(
                        f"Best solution for {self.n_sites} sites \nWeighted Average: {solution['weighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )
            else:
                title = plt.title(_safe_evaluate(title, solution=solution))

        return ax

    def plot_n_best_combinations(
        self,
        n_best=10,
        rank_on=None,
        title=None,
        subplot_title="default",
        show_all_locations=True,
        plot_site_allocation=False,
        plot_regions_not_meeting_threshold=False,
        cmap=None,
        chosen_site_colour="black",
        unchosen_site_colour="grey",
        n_cols=None,
        n_rows=None,
        fig_size_multiplier=6,
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
        rank_on : str, optional
            Column name used to rank solutions. If provided, solutions are
            sorted in ascending order and the top ``n_best`` are selected.
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
        n_cols : int, optional
            Number of columns in the subplot grid. If None, determined
            automatically.
        n_rows : int, optional
            Number of rows in the subplot grid. If None, determined
            automatically.
        fig_size_multiplier: int, optional
            Factor to adjust overall plot size

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

        if n_best > len(self.solution_df):
            n_best = len(self.solution_df)
            warn(
                f"n_best parameter higher than number of available solutions. Returning {len(self.solution_df)} solutions."
            )

        # Set up number of rows and columns if not specified
        if n_cols is None and n_rows is None:
            max_cols = 5
            ncols = min(n_best, max_cols)
            nrows = math.ceil(n_best / ncols)

        # Set up number of rows and columns if only one value specified
        elif n_cols is None and n_rows is not None:
            nrows = n_rows
            ncols = min(n_best, max_cols)

        elif n_cols is not None and n_rows is None:
            ncols = n_cols
            nrows = math.ceil(n_best / ncols)
        # If none of these conditions are met, then can assume both have been passed

        # Set up subplots for plotting onto
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_size_multiplier * ncols, fig_size_multiplier * nrows),
        )

        # flatten axs in case it's a 2D array
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()

        if rank_on is not None:
            sorted_df = self.solution_df.sort_values(rank_on).reset_index().head(n_best)
        else:
            sorted_df = self.solution_df.reset_index().head(n_best)

        # Choose appropriate colourmap if a colourmap is not provided
        if cmap is None:
            if plot_site_allocation:
                # If plotting site allocation, use a categorical (qualitative) colourmap
                cmap = "Set2"
            elif plot_regions_not_meeting_threshold:
                cmap = "Set1"
            else:
                # If plotting travel time/distance/other cost, use a sequential colourmap
                cmap = "Blues"

        # Set up a consistent legend that will be shared across all subplots
        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            # Calculate global color scale boundaries
            global_vmin = min(df["min_cost"].min() for df in sorted_df["problem_df"])
            global_vmax = max(df["min_cost"].max() for df in sorted_df["problem_df"])
        elif plot_site_allocation:
            master_site_order = self.site_problem.candidate_sites[
                self.site_problem._candidate_sites_candidate_id_col
            ].tolist()

            # Extract all unique selected sites across all top solutions for consistent mapping
            all_allocated_sites = set()
            for df in sorted_df["problem_df"]:
                all_allocated_sites.update(df["selected_site"].unique())
            all_allocated_sites = sorted(list(all_allocated_sites))

            sorted_allocated_sites = [
                site for site in master_site_order if site in all_allocated_sites
            ]

            # Create a consistent color mapping for categorical site data
            cmap_obj = plt.get_cmap(cmap)
            # Handles qualitative maps seamlessly (wrapping around if too many sites)
            site_color_map = {
                site: cmap_obj(i % cmap_obj.N)
                for i, site in enumerate(sorted_allocated_sites)
            }
        elif plot_regions_not_meeting_threshold:
            cmap_obj = plt.get_cmap(cmap)
            if hasattr(cmap_obj, "colors"):
                colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
            else:
                colors = [cmap_obj(0.0), cmap_obj(1.0)]

            discrete_cmap = ListedColormap(colors)

        for i, ax in enumerate(fig.axes):
            solution = sorted_df.iloc[[i]]
            solution_df = solution["problem_df"].values[0]

            nearest_site_travel_gdf = pd.merge(
                self.site_problem.region_geometry_layer,
                solution_df,
                left_on=self.site_problem._region_geometry_layer_common_col,
                right_on=self.site_problem._demand_data_id_col,
            )

            if plot_site_allocation:
                # Map the site IDs to their consistent global color
                colors = nearest_site_travel_gdf["selected_site"].map(site_color_map)
                ax = nearest_site_travel_gdf.plot(
                    color=colors,
                    legend=False,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax,
                )

            elif plot_regions_not_meeting_threshold:
                # nearest_site_travel_gdf["within_threshold_str"] = (
                #     nearest_site_travel_gdf["within_threshold"].apply(
                #         lambda x: (
                #             f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nof nearest site"
                #             if x is True
                #             else f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nfrom nearest site"
                #         )
                #     )
                # )

                ax = nearest_site_travel_gdf.plot(
                    "within_threshold",
                    legend=False,
                    cmap=discrete_cmap,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    figsize=(12, 6),
                    ax=ax,
                    vmin=0,
                    vmax=1,
                )
            # Otherwise plot the min cost (travel/cost to closest site from region)
            else:
                ax = nearest_site_travel_gdf.plot(
                    "min_cost",
                    legend=False,
                    cmap=cmap,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    figsize=(12, 6),
                    ax=ax,
                    vmin=global_vmin,
                    vmax=global_vmax,
                )

            if self.site_problem._candidate_sites_type == "geopandas":
                selected_sites = self.site_problem.candidate_sites.iloc[
                    solution.site_indices.iloc[0]
                ]

                if show_all_locations:
                    all_site_points = self.site_problem.candidate_sites.plot(
                        ax=ax, color=unchosen_site_colour, markersize=30, alpha=0.3
                    )

                selected_site_points = selected_sites.plot(
                    ax=ax, color=chosen_site_colour, markersize=60
                )

                for x, y, label in zip(
                    selected_sites.geometry.x,
                    selected_sites.geometry.y,
                    selected_sites[self.site_problem._candidate_sites_candidate_id_col],
                ):
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(10, 3),
                        textcoords="offset points",
                        bbox=dict(facecolor="white"),
                    )

            cx.add_basemap(
                ax,
                crs=nearest_site_travel_gdf.crs.to_string(),
            )

            ax.axis("off")

            if subplot_title is not None:
                if subplot_title == "default":
                    if self.objectives == "mclp":
                        ax.set_title(
                            f"Coverage within threshold of {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}: {solution['proportion_within_coverage_threshold'].values[0]:.1%} \nUnweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                        )

                    elif self.objectives in [
                        "simple_p_median",
                        "hybrid_simple_p_median",
                    ]:
                        ax.set_title(
                            f"Unweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                        )

                    else:
                        ax.set_title(
                            f"Weighted Average: {solution['weighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                        )
                else:
                    ax.set_title(_safe_evaluate(subplot_title, solution=solution))
            if title is not None:
                ax.set_title(title)

        # Set up appropriate shared legends
        # If plotting min travel time/cost per region
        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            # Create a single colorbar based on the global scale and chosen colormap
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax)
            )
            sm._A = []  # Empty array for the scalar mappable

            # Add the colorbar to the figure
            fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.04, label="Min Cost")

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
            # Create legend for threshold coverage
            if hasattr(cmap_obj, "colors"):
                # Get the actual defined color list
                colors = [cmap_obj.colors[0], cmap_obj.colors[1]]
            else:
                # Fallback for continuous maps: sample at 0 and 1
                colors = [cmap_obj(0.0), cmap_obj(1.0)]

            # 3. Create a discrete colormap for the plot to ensure it matches the legend
            # This ensures 0 maps to colors[0] and 1 maps to colors[1] exactly
            discrete_cmap = ListedColormap(colors)

            # Matplotlib assigns colors to boolean values: False=0, True=1
            legend_patches = [
                mpatches.Patch(
                    color=colors[0],  # Color for False (Index 0)
                    label=f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nfrom nearest site",
                ),
                mpatches.Patch(
                    color=colors[1],  # Color for True (Index 1)
                    label=f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nof nearest site",
                ),
            ]

            fig.legend(
                handles=legend_patches,
                title="Coverage Status",
                loc="center right",
                bbox_to_anchor=(1.05, 0.5),
            )

        return fig, axs

    def plot_n_best_combinations_bar(
        self,
        y_axis="weighted_average",
        n_best=10,
        interactive=True,
        rank_on=None,
        title="default",
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
        rank_on : str, optional
            Column name used to rank solutions. If provided, solutions are
            sorted in ascending order before selecting the top ``n_best``.
            If None, the existing order of ``solution_df`` is used.
        title : str or None, default="default"
            Title for the plot. If "default", an automatic title is generated
            based on the ranking metric or objective. If None, no title is set.
            Otherwise, the provided string is used as the title.

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
        to better solutions when ``rank_on`` is specified.
        """

        if rank_on is not None:
            df = self.solution_df.sort_values(rank_on)
        else:
            df = self.solution_df
        if n_best is not None:
            df = df.head(n_best)

        if interactive:
            df = df.copy()
            df["site_indices"] = df["site_indices"].astype("str")
            if rank_on is not None:
                title = f"Top {n_best} Solutions by {rank_on.replace('_', ' ').title()}"
            else:
                title = f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()}"
            fig = px.bar(
                df,
                x="site_indices",
                y=y_axis,
                title=title,
            )
        else:
            fig, ax = plt.subplots()
            ax.bar(
                df["site_indices"].astype(str),
                df[y_axis],
            )
            if title == "default":
                if rank_on is not None:
                    ax.set_title(
                        f"Top {n_best} Solutions by {rank_on.replace('_', ' ').title()}"
                    )
                else:
                    ax.set_title(
                        f"Top {n_best} Solutions: {self.objectives.replace('_', ' ').title()})"
                    )
            elif title is None:
                pass
            else:
                ax.set_title(title)

            ax.set_xlabel("Site Indices")
            ax.set_ylabel(f"{y_axis.replace('_', ' ').title()}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.close(fig)

        return fig

    def plot_simple_pareto_front(
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

    def plot_all_metric_pareto_front(
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
