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
