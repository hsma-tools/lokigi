from lokigi.utils import _safe_evaluate, _select_solution
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def plot_solution_sets_comparison(
    solution_sets,
    solutions_config,
    figsize=(16, 8),
    title=None,
    title_fontsize=14,
    **shared_plot_kwargs,
):
    """
    Plot solutions from different SiteSolutionSet instances side-by-side.

    This standalone function allows comparison of solutions across different
    solution sets (e.g., comparing car vs public transport accessibility).

    Parameters
    ----------
    solution_sets : list of SiteSolutionSet
        List of SiteSolutionSet instances to plot from. Must have same length
        as solutions_config.
    solutions_config : list of dict
        List of configuration dictionaries for each subplot. Each dict can contain:
        - 'solution_rank': int
        - 'site_names': list of str
        - 'site_indices': list of int or np.ndarray
        - 'rank_on': str
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

    Raises
    ------
    ValueError
        If solution_sets and solutions_config have different lengths.

    Examples
    --------
    # Compare car vs public transport for same site configuration
    fig, axes = plot_solution_sets_comparison(
        solution_sets=[car_solution_set, pt_solution_set],
        solutions_config=[
            {'site_indices': [0, 5, 10], 'title': 'Car Travel'},
            {'site_indices': [0, 5, 10], 'title': 'Public Transport'}
        ]
    )

    # Compare best solutions from different optimization runs
    fig, axes = plot_solution_sets_comparison(
        solution_sets=[run1, run2, run3],
        solutions_config=[
            {'solution_rank': 1, 'title': 'Run 1 (p-median)'},
            {'solution_rank': 1, 'title': 'Run 2 (MCLP)'},
            {'solution_rank': 1, 'title': 'Run 3 (hybrid)'}
        ],
        figsize=(24, 8)
    )
    """
    if len(solution_sets) != len(solutions_config):
        raise ValueError(
            f"solution_sets and solutions_config must have the same length. "
            f"Got {len(solution_sets)} solution sets and {len(solutions_config)} configs."
        )

    n_plots = len(solution_sets)

    if n_plots == 0:
        raise ValueError("Must provide at least one solution set and config")

    # Create subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    # Plot each solution from its respective solution set
    for i, (solution_set, config) in enumerate(zip(solution_sets, solutions_config)):
        # Merge shared kwargs with individual config
        plot_kwargs = {**shared_plot_kwargs, **config}

        # Get the solution
        solution = _select_solution(
            solution_set.solution_df,
            rank_on=config.get("rank_on"),
            solution_rank=config.get("solution_rank", 1),
            site_names=config.get("site_names"),
            site_indices=config.get("site_indices"),
        )

        # Extract plotting parameters (same as before)
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
        legend_bbox_to_anchor = plot_kwargs.pop("legend_bbox_to_anchor", (1.75, 0.5))
        legend_fontsize = plot_kwargs.pop("legend_fontsize", 10)
        site_legend_loc = plot_kwargs.pop("site_legend_loc", "best")
        subplot_title = plot_kwargs.pop("title", "default")
        subplot_title_fontsize = plot_kwargs.pop("title_fontsize", 12)

        # Choose colormap
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
        axes[i], has_required_sites = solution_set._plot_single_solution_map(
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
                # Generate default title
                if config.get("site_indices") is not None:
                    title_prefix = (
                        f"Solution for {solution_set.n_sites} sites (by site indices)"
                    )
                elif config.get("site_names") is not None:
                    title_prefix = (
                        f"Solution for {solution_set.n_sites} sites (by site names)"
                    )
                elif config.get("solution_rank", 1) == 1:
                    title_prefix = f"Best solution for {solution_set.n_sites} sites"
                else:
                    rank = config.get("solution_rank", 1)
                    rank_suffix = solution_set._get_ordinal_suffix(rank)
                    title_prefix = f"{rank}{rank_suffix} best solution for {solution_set.n_sites} sites"

                if solution_set.objectives == "mclp":
                    metrics = (
                        f"Coverage: {solution['proportion_within_coverage_threshold'].values[0]:.1%} | "
                        f"Avg: {solution['unweighted_average'].values[0]:.1f} | "
                        f"Max: {solution['max'].values[0]:.1f}"
                    )
                elif solution_set.objectives in [
                    "simple_p_median",
                    "hybrid_simple_p_median",
                ]:
                    metrics = (
                        f"Unweighted Avg: {solution['unweighted_average'].values[0]:.1f} | "
                        f"Max: {solution['max'].values[0]:.1f}"
                    )
                else:
                    metrics = (
                        f"Weighted Avg: {solution['weighted_average'].values[0]:.1f} | "
                        f"Max: {solution['max'].values[0]:.1f}"
                    )

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
