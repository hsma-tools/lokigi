## v0.7.0

### ⚠️ Breaking changes

lokigi is pre-1.0, so breaking changes can land in any minor release. Read these before upgrading -- each is detailed in the notes below.

- `proportion_within_coverage_threshold` and `coverage_by_equity_group` now report **demand-weighted** values rather than counts of regions. Existing numbers change on any problem with non-uniform demand. The previous behaviour is available as `proportion_regions_within_coverage_threshold` / `coverage_regions_by_equity_group`
- The **`mclp` objective may now select a different combination of sites**, because it ranks on the metric above. It now maximises covered demand, matching the textbook Maximal Covering Location Problem
- **`rank_on=` with a coverage metric now returns the best-covering solution, not the worst.** Anything ranking on a travel-cost metric is unchanged
- Both coverage proportions are now **`NaN` rather than `0.0`** when no `threshold_for_coverage` was supplied
- **`show_basemap` has been removed** from `plot_region_geometry_layer()`, `plot_hotspots()` and `plot_quadrant_map()`. Use `add_basemap`, which is now the argument name on every plotting method. Passing the old name raises a `TypeError` naming the replacement

### Notes

- Add `two_step_floating_catchment()` to `SiteProblem` and `SiteSolutionSet`, computing 2SFCA accessibility -- how much supply (e.g. GPs, beds, appointment slots) is actually available to each demand region, once competition from other regions for the same sites is accounted for
    - Unlike binary threshold coverage, two regions equally within a site's catchment can get different accessibility scores if one of them shares that site with far more competing demand, or has fewer other sites in reach
    - `supply_col` names a numeric column on `candidate_sites` at call time rather than being registered via `add_sites()`, so the same problem can be scored under different supply definitions (doctors vs beds vs slots) without re-adding sites
    - `SiteProblem.two_step_floating_catchment()` scores an arbitrary site set directly -- no `solve()` is required. With no `site_names`/`site_indices`, every registered candidate site is scored, which is only the same thing as "the current network" if the candidate pool doesn't also include not-yet-built proposals -- pass the currently-open subset explicitly if it does. `SiteSolutionSet.two_step_floating_catchment()` takes the same `rank_on`/`solution_rank`/`site_names`/`site_indices`/`matrix` solution-selection arguments as `site_allocation_summary()` and scores that solution's selected sites
    - Catchment membership uses an inclusive `<=` against `catchment_size`, unlike the coverage metrics' strict `<` against `threshold_for_coverage` -- matches the standard 2SFCA convention rather than lokigi's own coverage precedent
    - A site with no demand within `catchment_size` has an undefined (`NaN`) supply-to-demand ratio and is excluded from every region's accessibility score, with a warning naming it. A demand region with no site within `catchment_size` correctly scores `accessibility == 0` instead -- a real "no supply available" result, kept distinguishable from the NaN case above
    - `return_site_ratios=True` also returns the step-1 per-site table (`supply`, `catchment_demand`, `n_regions_in_catchment`, `ratio`), useful for finding which site is driving an implausible regional score
    - Descriptive only for now: does not feed into `solution_df`, `rank_on=`, or any `solve()` objective
- Add `plot_accessibility()` to `SiteProblem` and `SiteSolutionSet`, mapping 2SFCA accessibility: a region choropleth of `accessibility`, overlaid with site markers coloured and sized by their step-1 supply-to-demand `ratio` (red and small for an overloaded site, green and large for an uncontested one)
    - Computes `two_step_floating_catchment()` automatically from `supply_col`/`catchment_size` and the usual selection arguments, following the same optional-precomputed-input pattern as `plot_hotspots(hotspots_df=...)`; a precomputed `region_frame`/`site_frame` pair can be passed in instead (needed to plot a `SiteSolutionSet` result at a `solution_rank`/`rank_on` other than the default)
    - `interactive=True` returns a Folium map with both layers; the default returns a static matplotlib Axes with two colorbars
    - A site with an undefined (NaN) ratio -- no demand in its catchment -- is drawn in a distinct grey with its own legend entry ("No catchment demand"), since geopandas' `missing_kwds` label only appears on a discrete `scheme=` legend, not the continuous colorbar used here
    - Site markers are only drawn when `candidate_sites` has real point geometry, which includes tabular lat/long input (not just a GeoDataFrame passed directly to `add_sites()`) -- unlike `plot_best_combination()`, which silently skips its own site markers for lat/long-registered problems because it checks `_candidate_sites_type` (the input format) rather than the actual geometry
- Add `site_allocation_summary()` to `SiteSolutionSet`, reporting the share of demand (or of regions) whose closest selected site is each site in a chosen solution, and the average travel cost incurred by each site's group
    - Answers "is this extra site worth opening?" -- a site that is closest to only a small share of demand is a weak case for the capital cost, even where it lowers the average travel time
    - Also answers "how much further would people have to travel if this site closed?" via the `average_travel_cost` column -- demand-weighted mean travel cost among a site's closest regions by default (`unweighted` when `by="regions"`), inspired by work from Gill Baker showing that centralising services onto fewer sites would roughly double typical travel distance for patients, while a third site offered only limited further benefit
    - `by="demand"` (the default) weights each region by the demand registered via `add_demand()`; `by="regions"` counts every region equally, following the same people-vs-places naming rule as the coverage metrics above. Raises a `ValueError` rather than falling back if `by="demand"` is requested on a problem with no demand data, so a region count is never silently reported under a demand label
    - Selected sites that are closest to no region at all appear as explicit `0` rows in `n_regions`/`proportion` rather than being dropped by the underlying grouping -- a near-zero share is usually the finding being looked for. `average_travel_cost` is `NaN` for such a site instead of `0`, since there is no travel cost to average over zero regions
    - Takes the same solution-selection arguments as the plotting methods (`rank_on`/`solution_rank`/`site_names`/`site_indices`) and the same `matrix=` keyword to summarise a registered secondary travel matrix instead of the primary one
    - Regions exactly equidistant from two selected sites are assigned to the lower-indexed one rather than split between them
- Add `plot_site_allocation_summary()`, a horizontal bar chart of `site_allocation_summary()`. Uses the same "Set2" site colours as `plot_best_combination(plot_site_allocation=True)`, so the chart reads as a quantitative version of the allocation map, and always labels each bar with its value so a site capturing 0% stays visible
    - `metric="proportion"` (the default) plots allocation share; `metric="average_travel_cost"` plots the average travel cost column instead, labelled with the travel matrix's registered unit (e.g. "10.0 miles")
    - A zero-allocation site's bar is drawn at zero length either way, but its label differs by metric: "0.0%" for `proportion` (a real, meaningful value), versus "N/A" for `average_travel_cost` (there is no travel cost to average, and a "0" label there would misleadingly read as "instant to reach")
- Add `SolutionComparator.compare_site_allocation()`, putting two solutions' `site_allocation_summary()` results side by side with their difference -- e.g. a 2-site solution against a 3-site one, showing how much of the new site's catchment is genuinely new rather than taken from an existing site, or how much further a site's former patients would now have to travel if it closed
    - `metric="proportion"` (the default) compares allocation share; `metric="average_travel_cost"` compares the average travel cost column instead
    - With `metric="proportion"`, a site absent from a solution is `NaN` and a site that is opened but closest to nothing is `0.0`, so "not opened" and "opened but unused" stay distinguishable. With `metric="average_travel_cost"` both cases are `NaN`, since neither has a travel cost to average
- **Behaviour change:** coverage metrics are now weighted by demand rather than counting every region equally
    - `proportion_within_coverage_threshold` now reports the proportion of total *demand* within `threshold_for_coverage`, weighted by the demand registered via `add_demand()`. Previously it was the proportion of *regions*, so a sparsely-populated LSOA counted as much as a dense one
    - `coverage_by_equity_group` changes in the same way, reporting demand-weighted coverage within each equity band
    - Because `mclp` ranks on `proportion_within_coverage_threshold`, **the `mclp` objective may now select a different combination of sites** on any problem with non-uniform demand. It now maximises covered demand, matching the textbook Maximal Covering Location Problem
    - Nothing changes for problems with uniform demand, including those that never call `add_demand()` -- `solve()` assumes equal demand in that case, which makes the demand-weighted and region-based figures identical
    - The weighting always uses the raw demand column, never the compound `weights=` vector used by `weighted_average`, so the metric means "proportion of demand covered" regardless of what is passed to `weights=`
- Add `proportion_regions_within_coverage_threshold` and `coverage_regions_by_equity_group`, preserving the previous region-counting behaviour
    - Naming rule: an unqualified coverage metric is demand-weighted ("what share of people"), and the `regions` variants count every region equally ("what share of places")
    - Both are added for registered secondary travel matrices too: `proportion_regions_within_coverage_threshold__<label>` sits alongside its demand-weighted counterpart in the default per-matrix metric set, and `coverage_regions_by_equity_group__<label>` appears under `full_secondary_metrics=True`
    - Both are picked up automatically by `show_solutions(expand_dict_columns=True)`, which detects dict columns by content rather than by name
- Fix coverage columns for secondary travel matrices being sorted backwards
    - `plot_simple_pareto_front_pairs`, `plot_all_metric_pareto_front_pairs`, and `SolutionComparator` inferred "higher is better" by exact match against `proportion_within_coverage_threshold`, so a suffixed column such as `proportion_within_coverage_threshold__public_transport` was treated as a metric to minimise. Direction is now inferred for any coverage-proportion column, suffixed or not
- Both coverage proportions are `NaN` when no `threshold_for_coverage` was supplied, rather than `0.0`
- Fix `rank_on=` returning the *worst* solution when ranking on a coverage metric
    - Every `rank_on` call site sorted ascending unconditionally, which is right for the travel-cost metrics but backwards for coverage proportions, where higher is better. `return_best_combination_details()`, `return_best_combination_site_names()`, `return_best_combination_site_indices()` and the `rank_on`-accepting plotting methods (`plot_n_best_combinations_bar`, `plot_best_combination`, `plot_n_best_combinations`, `plot_travel_time_distribution`, `plot_combination_by_equity`) all returned or plotted the least-covering solution when asked for the best one
    - Direction is now resolved per column, so `rank_on="proportion_within_coverage_threshold"` (or any `regions`/`__<label>` coverage column) ranks highest-first, while travel-cost metrics are unchanged and still rank lowest-first
    - `plot_travel_time_distribution(secondary_ranking=...)` resolves the tie-breaker's direction independently of the primary metric, so a coverage metric can be tie-broken by a travel cost with each sorted the right way round
    - Affects which solution these methods return for coverage rankings only; anything ranking on `weighted_average`, `unweighted_average`, `90th_percentile`, `max` or `total_cost` is byte-for-byte unchanged
- Solutions tied on a ranking metric now keep a stable, reproducible order
    - Every sort over solutions uses a stable sort (`kind="mergesort"`), so equally-good solutions are no longer reshuffled arbitrarily. pandas' default single-column sort is quicksort, which is not stable, so which of several tied solutions was reported as "best" could differ between runs, machines or library versions
    - Affects `rank_on=` ranking, `pareto_summary()`, and the solution the Pareto narrative methods anchor on. Ties are routine -- on the sample Brighton problem `max` takes only 5 distinct values across 15 candidate combinations
    - `solve()`'s own ranking already sorted on two columns, which pandas handles with a stable lexsort, so no existing solve output changes
- **BREAKING:** `add_basemap` is now the argument name on every plotting method, and `show_basemap` has been removed
    - `plot_region_geometry_layer()`, `plot_hotspots()` and `plot_quadrant_map()` previously spelled it `show_basemap`, while `plot_sites()` and `plot_resources()` used `add_basemap`. Because the first three also accept `**kwargs`, passing `add_basemap` to them never reached code that understood it -- it crashed the static path with a confusing `AttributeError` from matplotlib (`PatchCollection.set() got an unexpected keyword argument`) and was silently ignored on the interactive path, drawing the tile layer anyway
    - Migration is a rename: `show_basemap=` becomes `add_basemap=`, with identical behaviour and the same `True` default
    - Those three methods explicitly reject the removed name with a `TypeError` naming the replacement. Without that guard the same `**kwargs` forwarding would give the identical cryptic matplotlib error, or ignore it silently on the interactive path -- so a bare removal would have been harder to diagnose than the original inconsistency
- Fix `plot_solution_comparison()` and `plot_solution_sets_comparison()` raising `AttributeError: 'SiteSolutionSet' object has no attribute '_get_ordinal_suffix'`
    - The ordinal-suffix helper is a module-level function in `lokigi.utils` but was called as though it were a method on the solution set, so plotting any solution other than the top-ranked one crashed. `solution_rank=1` took a different branch and worked, which is why this went unnoticed
- Fix `plot_travel_time_distribution(bottom_n=...)` raising `TypeError: list.append() takes no keyword arguments`
    - The bottom-ranked slice was appended with a stray keyword argument, so passing `bottom_n` at all crashed. Passing `top_n` alone was unaffected

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
    - `show_solutions(expand_dict_columns=True, inplace=True)` also writes the expansion back to `solution_df` so it persists for later calls, plotting, and `rank_on`; `inplace` has no effect unless combined with `expand_dict_columns=True`, and warns if passed alone. Rounding is never made permanent

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
