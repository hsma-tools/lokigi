## v0.6.0

- Add support for secondary travel matrices via `add_secondary_travel_matrix(travel_matrix_df, source_col, label, ...)`
    - Registers an additional travel/cost matrix (e.g. public transport alongside a primary car matrix) that is never used as the optimisation cost matrix -- the matrix registered via `add_travel_matrix()` always drives site selection, search, and pruning
    - Each registered secondary matrix contributes its own per-solution metric columns to `solution_df`, suffixed `__<label>` (e.g. `weighted_average__public_transport`, `min_cost__public_transport` on the per-region `problem_df`), so a single `solve()` produces one candidate ranking with metrics for every registered matrix side by side -- directly usable in `ParetoMetric(column=...)`, `rank_on=...`, and plots, without needing to `.copy()` the problem and solve twice
    - Any number of secondary matrices may be registered, each with its own `unit`/`from_unit`/`to_unit` and optional per-matrix `threshold_for_coverage` (falls back to the value passed to `solve()` if not set)
    - Secondary matrices must be complete (a row for every demand location, a column for every candidate site, no missing values) -- `solve()` raises a `KeyError` naming the label and the specific gap otherwise, rather than silently producing metrics over a different denominator than the primary matrix
    - By default, each secondary matrix only contributes its core five metrics plus float-valued equity aggregations to `solution_df` (not the dict-valued equity breakdowns or description strings, to keep the table from growing unboundedly with each registered matrix). Pass `solve(..., full_secondary_metrics=True)` to also include those, matching what the primary matrix already always returns
    - Plotting methods (`plot_best_combination`, `plot_n_best_combinations`, `plot_solution_comparison`, `plot_travel_time_distribution`, `check_solution_equity`, `plot_top_n_solution_equity`, `plot_combination_by_equity`) accept a new `matrix=` keyword to switch from the primary matrix to a registered secondary one
    - `plot_simple_pareto_front_pairs`'s `x_axis`/`y_axis` parameters now accept any `solution_df` column (previously typed as a fixed `Literal` set that already undersold what was accepted)
    - Ranking on a secondary matrix's columns (e.g. `rank_on="max__public_transport"`) only reorders candidates that were searched and pruned using the primary matrix -- see the new `add_secondary_travel_matrix()` docstring and the `multiple_travel_matrices` example for the `brute_force_keep_best_n`/`_worst_n` caveat this implies
    - `SolutionComparator` and the `problem.copy()`-per-mode workflow are unchanged and remain the right tool for two genuinely independent optimisations; secondary matrices are the alternative for trading modes off within one candidate ranking (see the new cross-reference in the `comparing_solutions` example)
- Add `expand_dict_columns` and `inplace` parameters to `show_solutions()`
    - `show_solutions(expand_dict_columns=True)` flattens every dict-valued column (`weighted_by_equity_group`, `coverage_by_equity_group`, etc., including their `__<label>` secondary-matrix equivalents under `full_secondary_metrics=True`) into one column per dict key, named `<column>__<key>`. Off by default, so `solution_df`'s shape is unchanged for existing callers
    - `show_solutions(expand_dict_columns=True, inplace=True)` also writes the expansion back to `solution_df` so it persists for later calls, plotting, and `rank_on`; `inplace` has no effect unless combined with `expand_dict_columns=True`. Rounding is never made permanent

## v0.5.0

- Add `n_jobs` parameter to `solve(search_strategy="brute-force", ...)` to evaluate combinations across multiple CPU cores (via `joblib`)
    - `n_jobs=1` (the default) is unchanged: byte-for-byte identical output to previous versions
    - `n_jobs>1`/`n_jobs=-1` always returns a correctly-ranked, correctly-bounded `keep_best_n`/`keep_worst_n` result; on an exact score tie spanning more combinations than the requested count, which specific tied combination is returned can differ from a serial run (their scores are identical either way)
    - Note: the first `solve(..., n_jobs=...)` call in a process (or any call after switching to a different `n_jobs` value) pays a one-time worker-pool startup cost -- on Windows this can be several seconds regardless of workload size, since each worker process re-imports pandas/numpy/etc. from scratch. Calls that reuse the same `n_jobs` value reuse the already-running pool and are fast; for a small combination count, a single one-off parallel call can look slower than `n_jobs=1` purely because of this startup cost
    - `joblib` is now a direct dependency (previously pulled in only transitively)

## v0.4.1

- Bugfixes for equity weighting
    - Fix `weights={"equity": ...}` giving the *most* weight to the least deprived regions instead of the most deprived, under both direction encodings
    - Rename `add_equity_data()`'s ambiguous `direction` parameter to `disadvantaged_end` (`direction` is kept as a deprecated alias and now raises a `FutureWarning`); also fix its default to match the documented DLUHC decile convention
    - Fix incomplete equity data silently dropping demand points from every metric (max, weighted/unweighted averages, coverage), not just equity-specific ones; `solve()` now raises a clear error when `"equity"` is weighted over incomplete data
- Bugfixes for search strategies (greedy / GRASP / brute-force)
    - Fix `max_value_cutoff` being silently ignored by greedy and GRASP (only brute-force enforced it)
    - Fix `required_sites_col` being ignored by GRASP, and greedy crashing when two or more sites were required
    - Fix `keep_best_n`/`keep_worst_n` being effectively random for `mclp`, and pruning before cost weighting was applied for brute-force
    - Fix a cost-weighting no-op silently inverting `mclp`'s search in greedy, GRASP, and the shared final sort, so the worst combination could be returned as "best"
    - Fix GRASP's `min_sites_different` diversity threshold using the wrong formula, accepting solutions as more diverse than requested
    - Add a clear error when more sites are required than `p` allows for brute-force (greedy/GRASP already had this)
- Bugfixes for `solve()` and single-solution evaluation
    - `solve()` now rejects unknown keyword arguments instead of silently swallowing typos
    - Fix mclp's missing-demand warning exemption not applying when `objectives` is passed as a list
    - Fix weight keys (`"demand"`/`"equity"`) not being truly case-insensitive, and a related unreachable error for genuinely unrecognised weight keys
    - Fix `evaluate_single_solution_single_objective` silently accepting partially-invalid, duplicate, or empty `site_indices`/`site_names`
- Fix equity plot titles rendering raw `np.int64(...)` reprs instead of plain site numbers; add `show_site_names=True` to list site names instead
- Add a backtest suite and document test conventions in `tests/README.md`

## v0.4.0

- Add support for setting site costs
    - `add_sites()` accepts a new `cost_col` parameter for each site's fixed (e.g. build or operating) cost
    - The total cost of the selected sites is now always reported in `solution_df` via `total_cost`, regardless of whether cost is used to rank solutions
    - `solve(weights={"cost": ...})` allows cost to influence which solution is chosen
    - Sites with a missing cost value now raise an error by default; pass `add_sites(..., allow_missing_cost=True)` to opt out and have `total_cost` propagate as `NaN` instead
    - Added an example notebook and sample dataset (Devon CDCs) demonstrating site costs
- Bugfixes for weights
    - Fix the `"cost"` weight key being silently ignored when passed with non-lowercase casing (e.g. `"Cost"`)
    - Fix `total_cost` silently treating a missing per-site cost as $0 instead of propagating it as unknown

## v0.3.0

- Add pareto front calculation and visualisation
- Add timeout to basemap calculations
- Documentation cleanups
- Bugfixes for weights
    - fix a failure when using equal demand
- Dependecy fixes to avoid importing optional dependencies by default

## v0.2.1

- Initial multiobjective optimisation work using weights

## v0.2.0

- Add hotspot calculation and plotting
- Add quadrant/ninth plots for demand/deprivation, travel/deprivation, travel/demand
- Add helpers and examples for travel time calculation with Valhalla and r5py
- Add helpers and examples for modification of max speeds in .osm.pbf files
- Add exploratory code for routing optimization (unfinished - paused indefinitely)

## v0.1.1

- Added missing plotly requirement.
- Made other requirements more permissive

## v0.1

Initial release.

**Please use with caution - testing suite is currently extremely limited**

Support for discrete location optimization problems.

Problems can be solved with brute force (including optionally setting a list of mandatory sites), greedy, and GRASP.

Supported problem types are simple p-median (unweighted travel times), standard p-median (demand-weighted travel times), and Maximal Covering Location Problem (MCLP). Hybrid variants of simple and standard p-median allow a maximum travel time constraint to be included.

A range of plotting options are included including maps of the problem and solutions, travel time distributions, solution equity, and comparisons of multiple solution sets (e.g. car vs public transport solutions to the same problem).
