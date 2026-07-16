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
