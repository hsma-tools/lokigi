from lokigi.plot_utils import plot_solution_sets_comparison


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
