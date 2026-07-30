import numpy as np
import pandas as pd
from warnings import warn
from lokigi.utils import (
    _add_rank_column,
    _is_maximise_metric,
    _select_solution,
    _population_impact_metrics,
    _split_bins_into_tertiles,
)


class SolutionComparatorMethodsMixin:
    def compare_top_results(self, n=1):
        """
        Compares the top N solutions from both sets.
        """
        top_a = self.set_a.solution_df.head(n).copy()
        top_b = self.set_b.solution_df.head(n).copy()

        top_a["origin"] = self.labels[0]
        top_b["origin"] = self.labels[1]

        return pd.concat([top_a, top_b]).reset_index(drop=True)

    def get_metric_summary(self, objective="weighted_average"):
        """
        Returns a comparison of descriptive statistics for a specific objective.
        """

        stats_a = self.set_a.solution_df[objective].describe()
        stats_b = self.set_b.solution_df[objective].describe()

        summary = pd.DataFrame({self.labels[0]: stats_a, self.labels[1]: stats_b})
        summary["difference"] = summary[self.labels[0]] - summary[self.labels[1]]
        return summary

    def compare_site_allocation(
        self,
        by="demand",
        metric="proportion",
        config_a=None,
        config_b=None,
        matrix=None,
        demand=None,
    ):
        """
        Compare `site_allocation_summary()` between `set_a` and `set_b`,
        e.g. a 2-site solution against a 3-site solution, to see whose
        demand a new site actually took, or how much further people would
        have to travel if a site were closed.

        Parameters
        ----------
        by : {"demand", "regions"}, default "demand"
            Passed to both sets' `site_allocation_summary()`.
        metric : {"proportion", "average_travel_cost"}, default "proportion"
            Which `site_allocation_summary()` column to compare.
            "proportion" answers "whose demand moved?"; "average_travel_cost"
            answers "how much further (or less far) do the people closest to
            this site now have to travel?" -- the comparison inspired by
            Gill Baker's work using average travel distance per patient to
            show that centralising services would roughly double typical
            travel distance, while a third site added little further
            benefit (see `site_allocation_summary`).
        config_a, config_b : dict, optional
            Keyword arguments forwarded to `set_a.site_allocation_summary()`
            and `set_b.site_allocation_summary()` respectively (e.g.
            ``{"solution_rank": 2}``), selecting which solution from each
            set to summarise. Default to ``{"solution_rank": 1}``.
        matrix : str, optional
            Passed to both sets' `site_allocation_summary()`.
        demand : str, optional
            Passed to both sets' `site_allocation_summary()`.

        Returns
        -------
        pandas.DataFrame
            Indexed by the union of both solutions' site names, in
            canonical site-index order. Columns are `self.labels[0]`,
            `self.labels[1]`, and `difference` (`labels[0] - labels[1]`,
            matching `get_metric_summary`'s direction).

        Notes
        -----
        With `metric="proportion"` (the default), `NaN` and `0.0` mean
        different things and are not interchangeable: `NaN` means the site
        is not in that solution at all (not opened), while `0.0` means the
        site is opened but is closest to no region. Collapsing the two
        would erase the point of the comparison -- e.g. comparing a 2-site
        solution against a 3-site one, the new site's row is `NaN` in the
        2-site column, not `0.0`.

        With `metric="average_travel_cost"`, this distinction does not
        apply: a site that is opened but closest to nothing has no travel
        cost to average either, so both "not opened" and "opened but
        empty" show up as `NaN`.
        """
        if metric not in ("proportion", "average_travel_cost"):
            raise ValueError(
                f"metric must be 'proportion' or 'average_travel_cost', got {metric!r}."
            )

        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        summary_a = self.set_a.site_allocation_summary(
            by=by, matrix=matrix, demand=demand, **config_a
        )
        summary_b = self.set_b.site_allocation_summary(
            by=by, matrix=matrix, demand=demand, **config_b
        )

        master_site_order = self.set_a.site_problem.candidate_sites[
            self.set_a.site_problem._candidate_sites_candidate_id_col
        ].tolist()
        union_sites = set(summary_a.index) | set(summary_b.index)
        index = [site for site in master_site_order if site in union_sites]

        label_a, label_b = self.labels
        comparison = pd.DataFrame(
            {
                label_a: summary_a[metric].reindex(index),
                label_b: summary_b[metric].reindex(index),
            }
        )
        comparison["difference"] = comparison[label_a] - comparison[label_b]
        return comparison

    def site_reallocation_matrix(
        self, matrix=None, demand=None, config_a=None, config_b=None, by="demand"
    ):
        """
        Cross-tabulate each demand location's closest site under `set_a`
        against its closest site under `set_b` -- "if a region's nearest
        site changed, which site did it move from, and which did it move
        to?"

        A single table answers both directions of "reallocation": read a
        row (a `set_a` site) across to see where that site's demand ended
        up in `set_b` -- e.g. a closed site's row shows exactly which
        remaining sites picked up its patients. Read a column (a `set_b`
        site) down to see which `set_a` sites its demand came from -- e.g.
        a newly opened site's column shows which existing sites it drew
        demand away from. The diagonal (same site name in both) is the
        demand that didn't move at all.

        Parameters
        ----------
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Cross-tabulates each side's
            `selected_site__<label>` column instead of the primary
            matrix's `selected_site`.
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Weights by that scenario's demand
            instead of the primary demand data (only relevant for
            `by="demand"`).
        config_a, config_b : dict, optional
            Passed to `_select_solution()` to choose which solution from
            `set_a`/`set_b` to compare (e.g. ``{"solution_rank": 2}``).
            Default to ``{"solution_rank": 1}``.
        by : {"demand", "regions"}, default "demand"
            "demand" sums the registered demand weight moving between
            each (from-site, to-site) pair; "regions" counts demand
            locations instead, ignoring demand weighting entirely --
            the only option available when no demand data is registered.

        Returns
        -------
        pandas.DataFrame
            Rows = `set_a`'s selected sites, columns = `set_b`'s selected
            sites. Cell (i, j) = the demand (or region count) whose
            closest site was i under `set_a` and is j under `set_b`. Every
            selected site gets a full row/column even if it captures no
            reallocated demand at all -- those cells are 0.0, not `NaN`,
            since "nothing moved this way" is a real, meaningful answer
            (matching `site_allocation_summary()`'s own "explicit 0"
            convention). The index/column names are
            `self.labels[0]`/`self.labels[1]`.

            Each axis is ordered persisting sites first (present on both
            sides, in canonical site-index order), then that axis's own
            "changed" sites last (closed sites at the bottom of the row
            axis; newly opened sites at the right of the column axis).
            Otherwise an unrelated unchanged site could sort after the
            genuinely closed/opened ones purely by candidate index,
            burying the actual reallocation in the middle of the table
            instead of grouping it where it's easy to scan.

        Raises
        ------
        ValueError
            If `by` is not "demand" or "regions"; if `by="demand"` but no
            demand data is registered (see `Notes`); if `demand` names an
            unregistered secondary demand scenario; or if `set_a` and
            `set_b`'s selected solutions were evaluated against different
            demand locations (via `_population_impact_frame`).

        Notes
        -----
        To see only what the review persona actually asked for -- "closed
        sites down the left, the sites they moved to across the top" --
        slice the result to just the rows/columns that differ between the
        two solutions, e.g. using the already-available
        `sites_closed_vs_baseline`/`sites_added_vs_baseline` columns:
        ``result.loc[closed_sites]`` for "where did each closed site's
        demand go?", or ``result[added_sites]`` for "which sites did each
        added site draw demand from?". The full matrix is returned rather
        than pre-filtered so that reallocation between two sites that are
        BOTH still open after the change (e.g. a new site partially
        stealing an existing site's catchment) isn't hidden either.
        """
        if by not in ("demand", "regions"):
            raise ValueError(f"by must be 'demand' or 'regions', got {by!r}.")

        frame = self._population_impact_frame(matrix, demand, config_a, config_b)
        problem_df_a = frame["problem_df_a"].reindex(frame["problem_df_b"].index)
        problem_df_b = frame["problem_df_b"]
        demand_series = frame["demand_series"]

        if by == "demand" and demand_series is None:
            raise ValueError(
                "by='demand' requires demand data. No demand column is "
                "registered on this problem -- call add_demand(), or pass "
                "by='regions' to count demand locations equally."
            )

        selected_site_col = self.set_b._resolve_travel_columns(matrix)[1]

        if by == "demand":
            # fillna(0.0): pivot_table (which crosstab delegates to once
            # values/aggfunc are given) otherwise leaves NaN for a (from,
            # to) pair that never co-occurs -- indistinguishable from "no
            # data" when it actually means "zero regions moved this way".
            crosstab = pd.crosstab(
                problem_df_a[selected_site_col],
                problem_df_b[selected_site_col],
                values=demand_series,
                aggfunc="sum",
            ).fillna(0.0)
        else:
            crosstab = pd.crosstab(
                problem_df_a[selected_site_col], problem_df_b[selected_site_col]
            )

        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}
        sites_a = list(
            _select_solution(self.set_a.solution_df, **config_a)["site_names"].iloc[0]
        )
        sites_b = list(
            _select_solution(self.set_b.solution_df, **config_b)["site_names"].iloc[0]
        )

        unexpected_rows = set(crosstab.index) - set(sites_a)
        unexpected_cols = set(crosstab.columns) - set(sites_b)
        if unexpected_rows or unexpected_cols:
            # Mirrors site_allocation_summary()'s own guard: a selected_site
            # value not in the solution's own site_names means problem_df
            # and site_names have gone out of step -- reindexing below
            # would otherwise silently drop these rather than flagging it.
            warn(
                f"site_reallocation_matrix: selected_site columns contain "
                f"site(s) not in the corresponding solution's site_names -- "
                f"{sorted(unexpected_rows)} in set_a, "
                f"{sorted(unexpected_cols)} in set_b. These are excluded "
                "from the result.",
                stacklevel=2,
            )

        master_site_order = self.set_b.site_problem.candidate_sites[
            self.set_b.site_problem._candidate_sites_candidate_id_col
        ].tolist()

        def _ordered(sites, other_sites):
            # Persisting sites (present on both sides) first, in canonical
            # order, then the sites unique to THIS side (closed, for rows;
            # opened, for columns) at the end -- so the actual reallocation
            # is grouped together at the bottom/right instead of scattered
            # wherever its candidate index happens to fall, which is what a
            # plain canonical-order listing would otherwise do (an unrelated
            # unchanged site sorting last, purely by site ID, is easy to
            # mistake for "the interesting one").
            other = set(other_sites)
            persisting = [s for s in master_site_order if s in sites and s in other]
            changed = [s for s in master_site_order if s in sites and s not in other]
            return persisting + changed

        row_order = _ordered(sites_a, sites_b)
        col_order = _ordered(sites_b, sites_a)

        result = crosstab.reindex(index=row_order, columns=col_order, fill_value=0.0)
        result.index.name = self.labels[0]
        result.columns.name = self.labels[1]
        return result

    def _population_impact_frame(self, matrix, demand, config_a, config_b):
        """
        Shared frame-building logic for `population_impact_summary()` and
        `population_impact_by_equity_group()`: selects each set's solution,
        resolves the cost column, aligns baseline vs current cost by
        demand-location ID (raising if the two solutions don't cover the
        same set of locations), and resolves the demand series to weight
        by. Factored out so both methods build this frame identically
        rather than risking two diverging implementations.

        Returns
        -------
        dict
            `baseline_cost`/`current_cost` (`pandas.Series`, indexed by
            demand-location ID); `demand_series` (`pandas.Series` or
            `None`); `problem_df_a`/`problem_df_b` (each set's full
            per-region frame, both indexed by demand-location ID -- carry
            any other column, e.g. an equity band or `within_threshold`,
            for further breakdowns); `within_col` (the resolved
            `within_threshold`/`within_threshold__<label>` column name for
            `matrix`, present or not depending on whether a coverage
            threshold was ever set -- callers check for its presence in
            `problem_df_a`/`problem_df_b`'s columns before using it).

        Raises
        ------
        ValueError
            If `demand` names an unregistered secondary demand scenario, or
            if `set_a` and `set_b`'s selected solutions were evaluated
            against different demand locations.
        """
        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        solution_a = _select_solution(self.set_a.solution_df, **config_a)
        solution_b = _select_solution(self.set_b.solution_df, **config_b)

        cost_col, _, within_col, _, _ = self.set_b._resolve_travel_columns(matrix)

        id_col = self.set_b.site_problem._demand_data_id_col
        problem_df_a = solution_a["problem_df"].iloc[0].set_index(id_col)
        problem_df_b = solution_b["problem_df"].iloc[0].set_index(id_col)

        ids_a = set(problem_df_a.index)
        ids_b = set(problem_df_b.index)
        if ids_a != ids_b:
            only_a = sorted(map(str, ids_a - ids_b))
            only_b = sorted(map(str, ids_b - ids_a))
            raise ValueError(
                "set_a and set_b's selected solutions cover different demand "
                f"locations -- {len(only_a)} only in set_a (e.g. {only_a[:5]}), "
                f"{len(only_b)} only in set_b (e.g. {only_b[:5]}). Both "
                "solutions must be evaluated against the exact same demand "
                "locations."
            )

        # Best-effort sanity check for the gross coverage-transition metrics
        # (population_impact_summary()'s demand_newly_covered/uncovered):
        # those diff each side's within_threshold flags directly, which is
        # only meaningful if both were assessed against the same coverage
        # threshold. "coverage_threshold" always reflects the PRIMARY
        # matrix's threshold, so this can't catch a mismatch specific to a
        # secondary matrix passed via matrix= -- it's a heuristic, not a
        # guarantee, but catches the common case of comparing two solutions
        # built with different threshold_for_coverage values by mistake.
        threshold_a = solution_a["coverage_threshold"].iloc[0]
        threshold_b = solution_b["coverage_threshold"].iloc[0]
        if (
            not pd.isna(threshold_a)
            and not pd.isna(threshold_b)
            and threshold_a != threshold_b
        ):
            warn(
                "set_a and set_b were evaluated with different "
                f"threshold_for_coverage values ({threshold_a} vs "
                f"{threshold_b}). Coverage-based comparisons (e.g. "
                "population_impact_summary()'s newly-covered/newly-"
                "uncovered counts) would then diff 'covered' flags computed "
                "against two different cutoffs, which is misleading.",
                UserWarning,
                stacklevel=3,
            )

        current_cost = problem_df_b[cost_col]
        baseline_cost = problem_df_a[cost_col].reindex(current_cost.index)

        if demand is None:
            demand_col = getattr(
                self.set_b.site_problem, "_demand_data_demand_col", None
            )
        else:
            registered_demand_labels = getattr(
                self.set_b.site_problem, "secondary_demand_matrices", {}
            )
            if demand not in registered_demand_labels:
                raise ValueError(
                    f"Unknown secondary demand scenario '{demand}'. "
                    f"Registered labels: {sorted(registered_demand_labels)}."
                )
            demand_col = f"demand__{demand}"

        has_demand = demand_col is not None and demand_col in problem_df_b.columns
        demand_series = problem_df_b[demand_col].astype(float) if has_demand else None

        return {
            "baseline_cost": baseline_cost,
            "current_cost": current_cost,
            "demand_series": demand_series,
            "problem_df_a": problem_df_a,
            "problem_df_b": problem_df_b,
            "within_col": within_col,
        }

    def population_impact_summary(
        self,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
        return_per_region=False,
        as_dict=False,
    ):
        """
        How many people's journey actually changed between `set_a` and
        `set_b`, and by how much -- a per-demand-location diff, rather
        than only the region-wide shift in `weighted_average`. A
        `weighted_average` shift dilutes a real, large, local effect
        across everyone else who is unaffected by it; this answers "how
        many people are better/worse off, and by how much" directly.

        **`self.set_a` is treated as the reference/baseline and `self.set_b`
        as the candidate** -- e.g. `SolutionComparator(baseline, candidate)`,
        where `baseline` is typically built with
        `SiteProblem.evaluate_baseline()`. This is unlike
        `get_metric_summary()`/`compare_site_allocation()`'s `difference`
        column, which is `set_a - set_b` with no particular baseline/
        candidate relationship implied -- but it does not conflict with it:
        every value returned here is a positive magnitude, with direction
        carried by the bucket name (`_improved`/`_worsened`) rather than by
        sign, so there is no ambiguous sign convention to reconcile.

        Parameters
        ----------
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Diffs that matrix's own
            `min_cost__<label>` column instead of the primary matrix's.
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Weights the diff by that scenario's
            demand instead of the primary demand data.
        meaningful_change_threshold : float, default 0.0
            A region's cost must move by strictly more than
            `max(meaningful_change_threshold, 1e-9)` to count as improved
            or worsened; anything smaller (including floating-point noise
            at the default 0.0) is `unchanged`.
        config_a, config_b : dict, optional
            Keyword arguments forwarded to `_select_solution()` for
            `set_a`/`set_b` respectively (e.g. ``{"solution_rank": 2}``),
            selecting which solution from each set to compare. Default to
            ``{"solution_rank": 1}``, matching `compare_site_allocation()`.
        return_per_region : bool, default False
            If True, also return a per-region DataFrame (`baseline_cost`,
            `current_cost`, `demand` if available, `delta`, `bucket`) for
            drill-down, indexed by demand-location ID.
        as_dict : bool, default False
            If False (the default), the summary is returned as a single-
            column `pandas.DataFrame` (index = metric name, column =
            `"value"`) -- pandas' own display formatting keeps this
            readable in a notebook, unlike a bare dict (numpy >=2.0 reprs
            its float scalars as e.g. `np.float64(46907.0)`, which shows
            through verbatim on a plain dict). Pass `as_dict=True` for the
            original `dict`, e.g. to pull out a single value with
            `impact["demand_improved"]` for further computation.

        Returns
        -------
        pandas.DataFrame (or dict if `as_dict=True`), or a 2-tuple of
        (summary, per-region DataFrame) if `return_per_region=True`
            `regions_improved`/`regions_worsened`/`regions_unchanged`
            (counts); `demand_improved`/`demand_worsened`/
            `demand_unchanged` (`NaN` if no demand data is registered);
            `proportion_demand_improved`/`proportion_demand_worsened`;
            `total_demand`; `mean_reduction_among_improved`/
            `mean_increase_among_worsened` (demand-weighted, positive
            magnitudes); `max_reduction`/`max_increase` (positive). See
            `lokigi.utils._population_impact_metrics` for the full
            definition -- this method is a thin wrapper around it that
            handles solution selection and demand-location alignment.

            Also, whenever a coverage threshold was assessed on both `set_a`
            and `set_b` (i.e. `within_threshold` isn't all-NaN on either
            side): `demand_newly_covered`/`demand_newly_uncovered` and
            `regions_newly_covered`/`regions_newly_uncovered` -- the GROSS
            number of people/regions crossing the coverage threshold in
            each direction (not the net change in
            `proportion_within_coverage_threshold`, which can mask
            simultaneous gains and losses). `demand_*` are `NaN` if no
            demand data is registered; the whole group of four keys is
            simply absent (not `NaN`) if no coverage threshold was ever
            assessed.

        Raises
        ------
        ValueError
            If `demand` names an unregistered secondary demand scenario, or
            if `set_a` and `set_b`'s selected solutions were evaluated
            against different demand locations (their `problem_df`s must
            share the exact same demand-location ID set).
        """
        frame = self._population_impact_frame(matrix, demand, config_a, config_b)
        baseline_cost = frame["baseline_cost"]
        current_cost = frame["current_cost"]
        demand_series = frame["demand_series"]
        has_demand = demand_series is not None

        impact = _population_impact_metrics(
            current=current_cost.to_numpy(),
            baseline=baseline_cost.to_numpy(),
            demand=None if demand_series is None else demand_series.to_numpy(),
            meaningful_change_threshold=meaningful_change_threshold,
        )

        # Gross coverage transitions: who crossed the coverage threshold in
        # each direction, not just how their travel cost changed -- e.g. a
        # region can get a shorter journey (population_impact's "improved")
        # without that being enough to newly cross threshold_for_coverage,
        # or vice versa near the boundary. Only added when a coverage
        # threshold was actually assessed on BOTH sides (an all-NaN
        # within_col means "never assessed", per _coverage_stats'
        # convention); silently omitted otherwise, matching this method's
        # existing "absent, not NaN" pattern for optional metrics.
        within_col = frame["within_col"]
        problem_df_a = frame["problem_df_a"]
        problem_df_b = frame["problem_df_b"]
        if within_col in problem_df_a.columns and within_col in problem_df_b.columns:
            within_a = problem_df_a[within_col].reindex(current_cost.index)
            within_b = problem_df_b[within_col]
            if not (within_a.isna().all() or within_b.isna().all()):
                covered_a = within_a.astype(bool)
                covered_b = within_b.astype(bool)
                newly_covered = covered_b & ~covered_a
                newly_uncovered = covered_a & ~covered_b

                impact["regions_newly_covered"] = int(newly_covered.sum())
                impact["regions_newly_uncovered"] = int(newly_uncovered.sum())
                if has_demand:
                    impact["demand_newly_covered"] = float(
                        (newly_covered.astype(float) * demand_series).sum()
                    )
                    impact["demand_newly_uncovered"] = float(
                        (newly_uncovered.astype(float) * demand_series).sum()
                    )
                else:
                    impact["demand_newly_covered"] = np.nan
                    impact["demand_newly_uncovered"] = np.nan

        # dtype="object" (rather than letting pandas infer a single float64
        # column) keeps each metric's own native type on display -- the
        # region-count metrics stay integers (e.g. "61") instead of being
        # upcast to float64 alongside the float-valued metrics ("61.0"),
        # matching how the same values already render inside a dict.
        summary = (
            impact
            if as_dict
            else pd.DataFrame({"value": pd.Series(impact, dtype="object")}).rename_axis(
                "metric"
            )
        )

        if not return_per_region:
            return summary

        per_region = pd.DataFrame(
            {"baseline_cost": baseline_cost, "current_cost": current_cost}
        )
        if has_demand:
            per_region["demand"] = demand_series
        per_region["delta"] = per_region["current_cost"] - per_region["baseline_cost"]
        tol = max(meaningful_change_threshold, 1e-9)
        per_region["bucket"] = np.select(
            [per_region["delta"] < -tol, per_region["delta"] > tol],
            ["improved", "worsened"],
            default="unchanged",
        )
        return summary, per_region

    def population_impact_worst_affected(
        self,
        n=10,
        direction="worsened",
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
    ):
        """
        The `n` demand locations hit hardest by the change between `set_a`
        and `set_b` -- naming names, rather than leaving a reader to infer
        "9,669 people worsened" affects *somewhere in particular*.
        `population_impact_summary()`/`population_impact_phrase()` give the
        region-wide total; this is the drill-down, built from the same
        per-region diff (`return_per_region=True`).

        Parameters
        ----------
        n : int, default 10
            Number of locations to return. If fewer than `n` locations fall
            in `direction`, all of them are returned (no error).
        direction : {"worsened", "improved"}, default "worsened"
            Which locations to rank. "worsened" (the default) answers "who
            is worst off?" -- the question that matters most for a
            proposed closure. "improved" answers "who benefits most?".
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_summary()`.

        Returns
        -------
        pandas.DataFrame
            Indexed by demand-location ID, ordered from most- to least-
            affected. Columns: `Before`, `After`, `Change` (signed --
            positive means longer/worse, negative means shorter/better),
            each suffixed with the travel matrix's registered unit in
            parentheses (e.g. `Before (minutes)`) if one was registered,
            and `People affected` (that location's demand weight) if
            demand data is registered, else omitted.

        Raises
        ------
        ValueError
            If `direction` isn't "worsened" or "improved", or any of the
            errors `population_impact_summary()` raises.
        """
        if direction not in ("worsened", "improved"):
            raise ValueError(
                f'direction must be "worsened" or "improved", got {direction!r}.'
            )

        _, per_region = self.population_impact_summary(
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
            return_per_region=True,
        )

        affected = per_region[per_region["bucket"] == direction]
        affected = affected.sort_values(
            "delta", ascending=(direction == "improved")
        ).head(n)

        unit = self.set_b._resolve_travel_columns(matrix)[3]
        unit_parenthetical = f" ({unit})" if unit else ""

        id_col = self.set_b.site_problem._demand_data_id_col
        result = pd.DataFrame(
            {
                f"Before{unit_parenthetical}": affected["baseline_cost"].round(1),
                f"After{unit_parenthetical}": affected["current_cost"].round(1),
                f"Change{unit_parenthetical}": affected["delta"].round(1),
            },
            index=affected.index,
        ).rename_axis(id_col)
        if "demand" in affected.columns:
            result["People affected"] = affected["demand"]

        return result

    def population_impact_by_equity_group(
        self,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
    ):
        """
        `population_impact_summary()`, split by equity band -- e.g. "of the
        people whose journey improved, what share of each deprivation band
        actually benefited?". This is what distinguishes a candidate that
        helps everyone roughly equally from one that wins on the region-wide
        average while mostly helping people who were already well served
        (or, just as importantly, one where the most disadvantaged areas are
        the ones getting *worse* journeys) -- neither is visible from
        `population_impact_summary()`'s single region-wide total.

        The rate-normalised columns (`proportion_of_band_improved`/
        `proportion_of_band_worsened` -- share of THIS band's own
        population, not share of the region-wide improved-total) are the
        headline: a band with more people can look like it "benefits most"
        on a raw headcount alone even if only a small fraction of that band
        actually improved.

        Parameters
        ----------
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to the same frame-building logic as
            `population_impact_summary()`.

        Returns
        -------
        pandas.DataFrame
            Indexed by equity band (raw value, e.g. IMD decile 1-10),
            ordered from most- to least-disadvantaged per
            `add_equity_data(disadvantaged_end=...)` (ties within a tertile
            keep ascending band order; bands are listed individually, not
            bucketed into thirds). Columns: `regions_improved`/
            `regions_worsened`/`regions_unchanged`, `demand_improved`/
            `demand_worsened`/`demand_unchanged` (`NaN` if no demand data is
            registered), `band_total_demand`, `proportion_of_band_improved`/
            `proportion_of_band_worsened`.

        Raises
        ------
        ValueError
            If no equity data is registered on the problem -- call
            `add_equity_data()` first -- or any of the errors
            `population_impact_summary()` raises (mismatched demand
            locations between `set_a`/`set_b`, or an unknown secondary
            demand scenario).
        """
        frame = self._population_impact_frame(matrix, demand, config_a, config_b)
        baseline_cost = frame["baseline_cost"]
        current_cost = frame["current_cost"]
        demand_series = frame["demand_series"]
        problem_df_b = frame["problem_df_b"]

        equity_col = getattr(self.set_b.site_problem, "_equity_data_equity_col", None)
        if equity_col is None or equity_col not in problem_df_b.columns:
            raise ValueError(
                "population_impact_by_equity_group() requires equity data "
                "to be registered on the problem -- call add_equity_data() "
                "first."
            )

        rows = {}
        for band, group in problem_df_b.groupby(equity_col):
            band_demand = (
                None if demand_series is None else demand_series.loc[group.index]
            )
            band_impact = _population_impact_metrics(
                current=current_cost.loc[group.index].to_numpy(),
                baseline=baseline_cost.loc[group.index].to_numpy(),
                demand=None if band_demand is None else band_demand.to_numpy(),
                meaningful_change_threshold=meaningful_change_threshold,
            )
            rows[band] = {
                "regions_improved": band_impact["regions_improved"],
                "regions_worsened": band_impact["regions_worsened"],
                "regions_unchanged": band_impact["regions_unchanged"],
                "demand_improved": band_impact["demand_improved"],
                "demand_worsened": band_impact["demand_worsened"],
                "demand_unchanged": band_impact["demand_unchanged"],
                "band_total_demand": band_impact["total_demand"],
                "proportion_of_band_improved": band_impact[
                    "proportion_demand_improved"
                ],
                "proportion_of_band_worsened": band_impact[
                    "proportion_demand_worsened"
                ],
            }

        disadvantaged_end = getattr(
            self.set_b.site_problem, "_equity_data_disadvantaged_end", None
        )
        unique_bins = sorted(rows.keys())
        lower, middle, upper = _split_bins_into_tertiles(unique_bins, disadvantaged_end)
        if lower is not None:
            ordered_bins = lower + middle + upper
        elif disadvantaged_end == "high":
            # Fewer than 3 distinct bands -- no tertile split to reuse, but
            # the most-to-least-disadvantaged ordering promised above still
            # applies. Without this, disadvantaged_end="high" silently fell
            # back to plain ascending raw-bin order (least-disadvantaged
            # first) for exactly the same reason the old tertile code was
            # wrong before its own disadvantaged_end fix.
            ordered_bins = list(reversed(unique_bins))
        else:
            ordered_bins = unique_bins

        result = pd.DataFrame.from_dict(rows, orient="index")
        return result.loc[ordered_bins].rename_axis(equity_col)

    def population_impact_phrase(
        self,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
    ):
        """
        Stakeholder-facing sentence summarising `population_impact_summary()`,
        e.g.:

            "46,907 people (9.0% of the cohort) get a shorter journey,
            averaging 16.1 minutes off. 61 of 729 regions improved; 0
            worsened; 668 unchanged."

        A "get a longer journey" clause is included only if people are
        actually worse off (`demand_worsened > 0`) -- for the common
        superset comparison (adding a site while keeping every existing
        one), that clause would otherwise always read "0 people ...,
        averaging nan minutes more", which is not a sentence worth
        surfacing. The people-based clauses are omitted entirely if no
        demand data is registered (`total_demand` is `NaN`); the
        region-count clause always works.

        When `meaningful_change_threshold` is greater than 0, both clauses
        say so explicitly ("a journey shorter by more than 5.0 minutes"
        rather than a bare "a shorter journey") -- a headline of "N people
        benefit" is only as meaningful as the change it counts, and a
        reader shouldn't have to separately go and check what threshold was
        used to know whether "benefit" means "any improvement at all" or
        something larger.

        When equity data is registered on the problem (`add_equity_data()`),
        two further clauses are added: the rate-normalised comparison
        between the most- and least-disadvantaged equity-band tertiles
        (per `population_impact_by_equity_group()` -- share of EACH
        tertile's own population, not share of the region-wide total), and
        -- unconditionally, whenever it is non-zero, regardless of whether
        the overall worsened clause above already fired -- how many people
        in the most disadvantaged tertile specifically saw a worse outcome.
        A benefits-only summary risks reading as if a candidate helps
        disadvantaged areas uniformly when some of them are in fact worse
        off; this clause is never silently dropped.

        Parameters
        ----------
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_summary()`.

        Returns
        -------
        str
        """
        impact = self.population_impact_summary(
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
            as_dict=True,
        )

        total_regions = (
            impact["regions_improved"]
            + impact["regions_worsened"]
            + impact["regions_unchanged"]
        )

        sentences = []

        shorter_clause = (
            f"a journey shorter by more than {meaningful_change_threshold:.1f} "
            "minutes"
            if meaningful_change_threshold > 0
            else "a shorter journey"
        )
        longer_clause = (
            f"a journey longer by more than {meaningful_change_threshold:.1f} "
            "minutes"
            if meaningful_change_threshold > 0
            else "a longer journey"
        )

        has_demand = not pd.isna(impact["total_demand"])
        if has_demand and impact["demand_improved"] > 0:
            sentences.append(
                f"{impact['demand_improved']:,.0f} people "
                f"({impact['proportion_demand_improved']:.1%} of the cohort) get "
                f"{shorter_clause}, averaging "
                f"{impact['mean_reduction_among_improved']:.1f} minutes off."
            )
        if has_demand and impact["demand_worsened"] > 0:
            sentences.append(
                f"{impact['demand_worsened']:,.0f} people "
                f"({impact['proportion_demand_worsened']:.1%} of the cohort) get "
                f"{longer_clause}, averaging "
                f"{impact['mean_increase_among_worsened']:.1f} minutes more."
            )

        sentences.append(
            f"{impact['regions_improved']} of {total_regions} regions improved; "
            f"{impact['regions_worsened']} worsened; "
            f"{impact['regions_unchanged']} unchanged."
        )

        equity_col = getattr(self.set_b.site_problem, "_equity_data_equity_col", None)
        if has_demand and equity_col is not None:
            try:
                by_band = self.population_impact_by_equity_group(
                    matrix=matrix,
                    demand=demand,
                    meaningful_change_threshold=meaningful_change_threshold,
                    config_a=config_a,
                    config_b=config_b,
                )
            except ValueError:
                by_band = None

            if by_band is not None:
                disadvantaged_end = getattr(
                    self.set_b.site_problem, "_equity_data_disadvantaged_end", None
                )
                lower, _, upper = _split_bins_into_tertiles(
                    sorted(by_band.index), disadvantaged_end
                )
                if lower is not None:
                    lower_rows = by_band.loc[lower]
                    upper_rows = by_band.loc[upper]
                    lower_total = lower_rows["band_total_demand"].sum()
                    upper_total = upper_rows["band_total_demand"].sum()
                    lower_worsened = lower_rows["demand_worsened"].sum()

                    if lower_total > 0 and upper_total > 0:
                        lower_rate = lower_rows["demand_improved"].sum() / lower_total
                        upper_rate = upper_rows["demand_improved"].sum() / upper_total
                        sentences.append(
                            f"In the most disadvantaged third of areas, "
                            f"{lower_rate:.1%} of people see {shorter_clause}, "
                            f"compared to {upper_rate:.1%} in the least "
                            "disadvantaged third."
                        )

                    if lower_worsened > 0:
                        lower_worsened_rate = (
                            lower_worsened / lower_total if lower_total > 0 else np.nan
                        )
                        sentences.append(
                            f"{lower_worsened:,.0f} people "
                            f"({lower_worsened_rate:.1%}) in the most "
                            f"disadvantaged third get {longer_clause}."
                        )

        return " ".join(sentences)

    def decision_summary(
        self,
        matrix=None,
        demand=None,
        meaningful_change_threshold=0.0,
        config_a=None,
        config_b=None,
        worst_affected_n=3,
    ):
        """
        A single stakeholder-facing summary bundling everything a decision-
        maker needs to weigh `set_b` (the candidate) against `set_a` (the
        baseline): which sites close or open, how many people are better/
        worse off and by how much, the specific places hit hardest, and
        how equitable the candidate network is on its own terms. Combines
        `population_impact_phrase()`, `population_impact_worst_affected()`,
        and the candidate's own equity verdicts into one call, rather than
        assembling the pieces by hand as the example notebooks otherwise
        do.

        Parameters
        ----------
        matrix, demand, meaningful_change_threshold, config_a, config_b
            Passed straight through to `population_impact_phrase()` and
            `population_impact_worst_affected()`.
        worst_affected_n : int, default 3
            Number of hardest-hit places to name (see
            `population_impact_worst_affected(n=...)`). Kept small since
            this is meant to read as a paragraph, not a table -- call
            `population_impact_worst_affected()` directly for the full
            list.

        Returns
        -------
        str
            A blank-line-separated, multi-paragraph summary: which sites
            close/open (or a note that none do); the population-impact
            phrase; the hardest-hit places by name, worsened by more than
            `meaningful_change_threshold` (omitted entirely if nothing
            worsened); and the candidate network's own equity verdicts
            (best-vs-worst-group and most-vs-least-deprived-third),
            omitted if no equity data is registered.
        """
        config_a = config_a or {"solution_rank": 1}
        config_b = config_b or {"solution_rank": 1}

        sites_a = set(
            _select_solution(self.set_a.solution_df, **config_a)["site_names"].iloc[0]
        )
        sites_b = set(
            _select_solution(self.set_b.solution_df, **config_b)["site_names"].iloc[0]
        )
        closed = sorted(sites_a - sites_b)
        opened = sorted(sites_b - sites_a)

        site_change_sentences = []
        if closed:
            site_change_sentences.append(
                f"{len(closed)} site{'s' if len(closed) != 1 else ''} would "
                f"close: {', '.join(closed)}."
            )
        if opened:
            site_change_sentences.append(
                f"{len(opened)} site{'s' if len(opened) != 1 else ''} would "
                f"open: {', '.join(opened)}."
            )
        if not closed and not opened:
            site_change_sentences.append(
                "No sites differ between the two networks compared here."
            )

        paragraphs = [" ".join(site_change_sentences)]

        paragraphs.append(
            self.population_impact_phrase(
                matrix=matrix,
                demand=demand,
                meaningful_change_threshold=meaningful_change_threshold,
                config_a=config_a,
                config_b=config_b,
            )
        )

        worst_affected = self.population_impact_worst_affected(
            n=worst_affected_n,
            direction="worsened",
            matrix=matrix,
            demand=demand,
            meaningful_change_threshold=meaningful_change_threshold,
            config_a=config_a,
            config_b=config_b,
        )
        if len(worst_affected) > 0:
            before_col, after_col = worst_affected.columns[0], worst_affected.columns[1]
            places = "; ".join(
                f"{name} ({row[before_col]:.1f} -> {row[after_col]:.1f})"
                for name, row in worst_affected.iterrows()
            )
            paragraphs.append(f"Hit hardest: {places}.")

        equity_col = getattr(self.set_b.site_problem, "_equity_data_equity_col", None)
        if equity_col is not None:
            candidate = _select_solution(self.set_b.solution_df, **config_b).iloc[0]
            paragraphs.append(
                f"Equity in the proposed network: "
                f"{candidate['gap_relative_description']} "
                f"{candidate['inter_tertile_description']}"
            )

        return "\n\n".join(paragraphs)

    def site_overlap(self, top_n=1):
        """
        Analyzes how many sites are common between the top N solutions
        of both sets.
        """

        def get_all_sites(solution_df, n):
            # Flattens the list of site_indices from the top N rows
            all_indices = solution_df.head(n)["site_indices"].explode()
            return set(all_indices)

        sites_a = get_all_sites(self.set_a.solution_df, top_n)
        sites_b = get_all_sites(self.set_b.solution_df, top_n)

        common = sites_a.intersection(sites_b)
        only_a = sites_a - sites_b
        only_b = sites_b - sites_a

        return {
            "common_sites_count": len(common),
            "common_sites_indices": list(common),
            "unique_to_a": list(only_a),
            "unique_to_b": list(only_b),
            "jaccard_similarity": len(common) / len(sites_a.union(sites_b))
            if sites_a.union(sites_b)
            else 0,
        }

    def find_balanced_solution(
        self,
        top_n=25,
        method="rank_balanced",
        objective="weighted_average",
        secondary_objective="max",
        rank_weight=0.5,  # Controls how harshly to penalise high (poor) ranks
        return_details=False,
    ):
        """
        Finds the most similar top solutions from each set to approximate
        a balanced compromise without deep optimization.

        Parameters:
        -----------
        top_n : int
            Number of top solutions from each set to consider
        method : str
            Similarity metric to use:
            - 'jaccard': Jaccard similarity of site sets
            - 'overlap': Raw count of overlapping sites
            - 'combined': Weighted combination of Jaccard and normalized objectives
            - 'rank_balanced': Jaccard similarity penalised by the sum of their ranks (prioritises shared balance)
        rank_weight : float
            Only used if method='rank_balanced'. Determines the strength of the rank penalty.
        return_details : bool
            If True, returns full comparison details; if False, returns just the best pair

        Returns:
        --------
        dict or tuple
            If return_details=True: dict with best pair info and all comparisons
            If return_details=False: tuple of (solution_from_a, solution_from_b)
        """

        ascending_primary = not _is_maximise_metric(objective)
        ascending_secondary = not _is_maximise_metric(secondary_objective)

        sols_a_copy = self.set_a.solution_df.copy()
        sols_a_copy = _add_rank_column(
            sols_a_copy,
            score_col=objective,
            tiebreaker_col=secondary_objective,
            ascending=[ascending_primary, ascending_secondary],
        )
        sols_b_copy = self.set_b.solution_df.copy()
        sols_b_copy = _add_rank_column(
            sols_b_copy,
            score_col=objective,
            tiebreaker_col=secondary_objective,
            ascending=[ascending_primary, ascending_secondary],
        )

        # Get top N from each set
        top_a = sols_a_copy.head(top_n)
        top_b = sols_b_copy.head(top_n)

        # Get the objective column name
        obj = objective

        # Changed to negative infinity to allow for negative scores in rank_balanced
        best_similarity = -float("inf")
        best_pair = None
        all_comparisons = []

        # Use enumerate (start=1) to capture the rank of the solution in the top_n slice
        for rank_a, (idx_a, row_a) in enumerate(top_a.iterrows(), start=1):
            sites_a = set(row_a["site_indices"])

            for rank_b, (idx_b, row_b) in enumerate(top_b.iterrows(), start=1):
                sites_b = set(row_b["site_indices"])

                # Calculate similarity metrics
                intersection = sites_a.intersection(sites_b)
                union = sites_a.union(sites_b)

                jaccard = len(intersection) / len(union) if union else 0
                overlap_count = len(intersection)

                if method == "combined":
                    # Normalize objectives to [0, 1] range within the top_n
                    obj_range_a = top_a[obj].max() - top_a[obj].min()
                    obj_range_b = top_b[obj].max() - top_b[obj].min()

                    norm_a = (
                        (row_a[obj] - top_a[obj].min()) / obj_range_a
                        if obj_range_a > 0
                        else 1
                    )
                    norm_b = (
                        (row_b[obj] - top_b[obj].min()) / obj_range_b
                        if obj_range_b > 0
                        else 1
                    )

                    # Penalize large differences in objective quality
                    obj_similarity = 1 - abs(norm_a - norm_b)

                    # Combined score: 70% site overlap, 30% objective similarity
                    similarity = 0.7 * jaccard + 0.3 * obj_similarity

                elif method == "rank_balanced":
                    # Calculate rank penalty based on sum of ranks.
                    # Normalised so the maximum penalty is 1.0 (when both are ranked at top_n)
                    rank_penalty = (rank_a + rank_b) / (2 * top_n)

                    # Score is Jaccard overlap minus the weighted penalty.
                    # This ensures identical sets (Jaccard=1) are differentiated purely by rank sum.
                    similarity = jaccard - (rank_weight * rank_penalty)

                else:
                    similarity = jaccard if method == "jaccard" else overlap_count

                comparison = {
                    "index_a": idx_a,
                    "index_b": idx_b,
                    "rank_a": rank_a,  # Track rank A
                    "rank_b": rank_b,  # Track rank B
                    "similarity": similarity,
                    "jaccard": jaccard,
                    "overlap_count": overlap_count,
                    "obj_value_a": row_a[obj],
                    "obj_value_b": row_b[obj],
                    "sites_a": sites_a,
                    "sites_b": sites_b,
                    "common_sites": intersection,
                }

                all_comparisons.append(comparison)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (row_a, row_b, comparison)

        if best_pair is None:
            return None

        solution_a, solution_b, best_comparison = best_pair

        if return_details:
            # Sort all comparisons by similarity
            all_comparisons.sort(key=lambda x: x["similarity"], reverse=True)

            return {
                "best_from_a": solution_a.to_dict(),
                "best_from_b": solution_b.to_dict(),
                "best_rank_a": best_comparison[
                    "rank_a"
                ],  # Include winning ranks in output
                "best_rank_b": best_comparison["rank_b"],
                "similarity_score": best_similarity,
                "jaccard": best_comparison["jaccard"],
                "overlap_count": best_comparison["overlap_count"],
                "common_sites": list(best_comparison["common_sites"]),
                "unique_to_a": list(
                    best_comparison["sites_a"] - best_comparison["sites_b"]
                ),
                "unique_to_b": list(
                    best_comparison["sites_b"] - best_comparison["sites_a"]
                ),
                "top_5_matches": all_comparisons[:5],
                "method_used": method,
            }
        else:
            return (solution_a, solution_b)
