import pandas as pd
import contextily as cx
import textwrap
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from warnings import warn
import numpy as np
import math
import plotly.express as px
from lokigi.utils import _safe_evaluate
import sweetpareto.vis as spv


class EvaluatedCombination:
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
    def __init__(self, solution_df, site_problem, objectives, n_sites=None):
        self.solution_df = solution_df.reset_index(drop=True)
        self.site_problem = site_problem
        self.objectives = objectives
        self.n_sites = n_sites

    def show_solutions(self, rounding=2):
        return round(self.solution_df, rounding)

    def return_best_combination_details(self, rank_on=None, top_n=1):
        if rank_on is not None:
            return self.solution_df.sort_values(rank_on).head(top_n).reset_index()
        else:
            return self.solution_df.head(top_n).reset_index()

    def return_best_combination_site_indices(self, rank_on=None):
        if rank_on is not None:
            return (
                self.solution_df.sort_values(rank_on)
                .head(1)["site_indices"]
                .reset_index()[0]
            )
        else:
            return self.solution_df.head(1)["site_indices"].reset_index()[0]

    def return_best_combination_site_names(self, rank_on=None):
        if rank_on is not None:
            return (
                self.solution_df.sort_values(rank_on)
                .head(1)["site_names"]
                .reset_index()[0]
            )
        else:
            return self.solution_df.head(1)["site_names"].reset_index()[0]

    def plot_travel_time_distribution(
        self,
        top_n=1,
        rank_on=None,
        secondary_ranking="max",
        title="default",
        height=None,
        height_per_plot=250,
        compare_to_best=False,
        **kwargs,
    ):
        if rank_on is not None:
            solutions_filtered = (
                self.solution_df.sort_values([rank_on, secondary_ranking])
                .reset_index(drop=True)
                .head(top_n)
            )
        else:
            solutions_filtered = self.solution_df.reset_index(drop=True).head(top_n)

        dfs = []
        if compare_to_best:
            best_df = solutions_filtered.head(1)["problem_df"][0]

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
            + " | Weighted Average: "
            + dfs["weighted_average"].round(2).astype(str)
            + " |Unweighted Average: "
            + dfs["unweighted_average"].round(2).astype(str)
            + " | 90th percentile: "
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
            **kwargs,
        )

        fig.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace("label=", ""),
                textangle=0,
                x=a.x - 0.98,  # small shift left
                y=a.y + 0.092,  # move above subplot
                xanchor="left",
                yanchor="bottom",
            )
        )

        if title == "default":
            if rank_on is not None:
                fig.update_layout(
                    title=f"Distribution of Travel Times (Top {top_n} Solutions by {rank_on.replace('_', ' ').title()})"
                )
            else:
                fig.update_layout(
                    title=f"Distribution of Travel Times (Top {top_n} Solutions: {self.objectives.replace('_', ' ').title()})"
                )
        else:
            fig.update_layout(title=title)

        if compare_to_best:
            fig.add_vline(
                x=0,
                line_color="black",
                line_width=2,
                annotation_text="Best",
            )
        else:
            for i, (_, row) in enumerate(solutions_filtered.iterrows()):
                fig.add_vline(
                    x=row["weighted_average"],
                    line_dash="dash",
                    annotation_text="WA",
                    row=i + 1,
                    col=1,
                )

                fig.add_vline(
                    x=row["90th_percentile"],
                    line_dash="dot",
                    annotation_text="P90",
                    row=i + 1,
                    col=1,
                )

        if height is None:
            fig_height = max(300, height_per_plot * top_n)

            fig.update_layout(height=fig_height)

            # fig.update_layout(
            #     margin=dict(t=80, b=40, l=40, r=40),
            # )

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
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
    ):
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
            if not plot_site_allocation:
                # If plotting travel time/distance/other cost, use a sequential colourmap
                cmap = "Blues"
            else:
                # If plotting site allocation, use a categorical (qualitative) colourmap
                cmap = "Set2"

        if plot_site_allocation:
            ax = nearest_site_travel_gdf.plot(
                "selected_site",
                legend=True,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
            )
        elif plot_regions_not_meeting_threshold:
            nearest_site_travel_gdf["within_threshold"] = nearest_site_travel_gdf[
                "within_threshold"
            ].apply(
                lambda x: (
                    f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nof nearest site"
                    if x is True
                    else f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nfrom nearest site"
                )
            )
            ax = nearest_site_travel_gdf.plot(
                "within_threshold",
                legend=True,
                cmap=cmap,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                figsize=(12, 6),
            )
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
                    plt.title(
                        f"Best solution for {self.n_sites} sites \nCoverage within threshold of {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}: {solution['proportion_within_coverage_threshold'].values[0]:.1%} \nUnweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )

                elif self.objectives in [
                    "simple_p_median",
                    "hybrid_simple_p_median",
                ]:
                    plt.title(
                        f"Best solution for {self.n_sites} sites \nUnweighted Average: {solution['unweighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )

                else:
                    plt.title(
                        f"Best solution for {self.n_sites} sites \nWeighted Average: {solution['weighted_average'].values[0]:.1f} {self.site_problem._travel_matrix_unit} \nMaximum: {solution['max'].values[0]:.1f} {self.site_problem._travel_matrix_unit}"
                    )
            else:
                plt.title(_safe_evaluate(title, solution=solution))

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
        chosen_site_colour="magenta",
        unchosen_site_colour="grey",
        n_cols=None,
        n_rows=None,
    ):
        if n_best > len(self.solution_df):
            n_best = len(self.solution_df)

        if n_cols is None and n_rows is None:
            max_cols = 5
            ncols = min(n_best, max_cols)
            nrows = math.ceil(n_best / ncols)

        elif n_cols is None and n_rows is not None:
            nrows = n_rows
            ncols = min(n_best, max_cols)

        else:
            ncols = n_cols
            nrows = math.ceil(n_best / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

        # flatten axs in case it's a 2D array
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()

        if self.site_problem.region_geometry_layer is None:
            raise ValueError(
                "The region data has not been initialised in the problem class."
                "Please run add_region_geometry_layer() first."
            )

        if rank_on is not None:
            sorted_df = self.solution_df.sort_values(rank_on).reset_index().head(n_best)
        else:
            sorted_df = self.solution_df.reset_index().head(n_best)

        # Choose appropriate colourmap if a colourmap is not provided
        if cmap is None:
            if not plot_site_allocation:
                # If plotting travel time/distance/other cost, use a sequential colourmap
                cmap = "Blues"
            else:
                # If plotting site allocation, use a categorical (qualitative) colourmap
                cmap = "Set2"

        # Set up a consistent legend that will be shared across all subplots
        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            # Calculate global color scale boundaries
            global_vmin = min(df["min_cost"].min() for df in sorted_df["problem_df"])
            global_vmax = max(df["min_cost"].max() for df in sorted_df["problem_df"])
        else:
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
                nearest_site_travel_gdf["within_threshold"] = nearest_site_travel_gdf[
                    "within_threshold"
                ].apply(
                    lambda x: (
                        f"Within {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nof nearest site"
                        if x is True
                        else f"Further than {solution['coverage_threshold'].values[0]} {self.site_problem._travel_matrix_unit}\nfrom nearest site"
                    )
                )
                ax = nearest_site_travel_gdf.plot(
                    "within_threshold",
                    legend=False,
                    cmap=cmap,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    figsize=(12, 6),
                    ax=ax,
                )
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

        if not plot_site_allocation and not plot_regions_not_meeting_threshold:
            # Create a single colorbar based on the global scale and chosen colormap
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax)
            )
            sm._A = []  # Empty array for the scalar mappable

            # Add the colorbar to the figure
            fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.04, label="Min Cost")
        else:
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

        return fig, ax

    def plot_n_best_combinations_bar(
        self,
        y_axis="weighted_average",
        n_best=10,
        interactive=True,
        rank_on=None,
        title="default",
    ):
        if rank_on is not None:
            df = self.solution_df.sort_values(rank_on)
        else:
            df = self.solution_df
        if n_best is not None:
            df = df.head(n_best)

        if interactive:
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
    ):
        fig = spv.pareto_plot(
            self.solution_df,
            x=x_axis,
            y=y_axis,
            maxx=False,
            maxy=True,
            show_points=True,
            height=4,
            theme="whitegrid",
        )

        return fig

    def plot_all_metric_pareto_front(self):
        metrics = [
            "weighted_average",
            "unweighted_average",
            "90th_percentile",
            "max",
            "proportion_within_coverage_threshold",
        ]

        metric_pairs = list(itertools.combinations(metrics, 2))
        num_plots = len(metric_pairs)

        # Calculate grid size (e.g., 5 metrics = 10 plots -> 3x4 grid)
        cols = 3
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()  # Flatten to easily iterate over a 1D array of axes

        for idx, (x_metric, y_metric) in enumerate(metric_pairs):
            ax = axes[idx]
            self.plot_simple_pareto_front(x_axis=x_metric, y_axis=y_metric, ax=ax)
            ax.set_title(f"{y_metric} vs {x_metric}")

        # Hide any unused subplots (if your grid is larger than your pair count)
        for idx in range(num_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig
