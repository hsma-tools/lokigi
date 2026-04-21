import numpy as np

from lokigi.mixins.site_solution_plots import (
    MapsMixin,
    NonMapPlotsMixin,
    ParetoPlotsMixin,
    DistributionPlotsMixin,
    EquityPlotsMixin,
)


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

        # Weighted average code modified from
        # https://github.com/health-data-science-OR/healthcare-logistics/blob/8d03b890a8ce861b64f6f834710dc50f2d85f68e/optimisation/metapy/evolutionary/evolutionary.py#L722
        # Credit for original code to Dr Tom Monks
        # Licence reproduced below in line with reuse conditions
        # MIT License
        #
        # Copyright (c) 2020 health-data-science-OR
        #
        # Permission is hereby granted, free of charge, to any person obtaining a copy
        # of this software and associated documentation files (the "Software"), to deal
        # in the Software without restriction, including without limitation the rights
        # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        # copies of the Software, and to permit persons to whom the Software is
        # furnished to do so, subject to the following conditions:
        #
        # The above copyright notice and this permission notice shall be included in all
        # copies or substantial portions of the Software.
        #
        # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        # SOFTWARE.
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


class SiteSolutionSet(
    MapsMixin,
    NonMapPlotsMixin,
    ParetoPlotsMixin,
    DistributionPlotsMixin,
    EquityPlotsMixin,
):
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

    def summary_table(self):
        pass
