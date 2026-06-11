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
