import json
from pathlib import Path

import pytest
import lokigi
import pandas as pd
from shapely.geometry import Point
import geopandas


# --- Backtest snapshot machinery (see tests/test_backtests.py) ---
#
# Backtests pin solve()'s current output rather than independently deriving
# it, so updating them by hand means re-running each case and re-typing
# numbers into the test file. Instead, expected values live in
# tests/backtest_snapshots.json, and `pytest --update-backtests` regenerates
# that file from whatever solve() currently returns.

BACKTEST_SNAPSHOT_PATH = Path(__file__).parent / "backtest_snapshots.json"


def pytest_addoption(parser):
    parser.addoption(
        "--update-backtests",
        action="store_true",
        default=False,
        help=(
            "Regenerate tests/backtest_snapshots.json from solve()'s current "
            "output instead of asserting against the stored snapshot. Use "
            "after a deliberate change to solver behaviour; review the diff "
            "before committing it."
        ),
    )


@pytest.fixture(scope="session")
def _backtest_snapshot_store(request):
    if BACKTEST_SNAPSHOT_PATH.exists():
        snapshots = json.loads(BACKTEST_SNAPSHOT_PATH.read_text())
    else:
        snapshots = {}
    updates = {}
    yield snapshots, updates, request.config.getoption("--update-backtests")
    if updates:
        snapshots.update(updates)
        BACKTEST_SNAPSHOT_PATH.write_text(
            json.dumps(snapshots, indent=2, sort_keys=True) + "\n"
        )


@pytest.fixture
def assert_backtest(_backtest_snapshot_store, request):
    """Compare a fingerprint (see test_backtests.py::_fingerprint) against its
    stored snapshot, keyed by the calling test's node id. In
    `--update-backtests` mode, records the current value to be written to
    tests/backtest_snapshots.json once the session finishes instead."""
    snapshots, updates, update_mode = _backtest_snapshot_store

    def _normalize(value):
        """Recursively convert tuples to lists so a freshly computed
        fingerprint compares equal to one that's round-tripped through JSON
        (which has no tuple type)."""
        if isinstance(value, (tuple, list)):
            return [_normalize(item) for item in value]
        return value

    def _assert(fingerprint):
        key = request.node.name
        normalized = _normalize(fingerprint)
        if update_mode:
            updates[key] = normalized
            return
        expected = snapshots.get(key)
        assert expected is not None, (
            f"No stored backtest snapshot for {key!r}. Run "
            "`pytest tests/test_backtests.py --update-backtests` to create it, "
            "then review tests/backtest_snapshots.json before committing."
        )
        assert normalized == expected, (
            f"Backtest {key!r} drifted from its stored snapshot in "
            "tests/backtest_snapshots.json. If this is an intentional solver "
            "change, rerun with --update-backtests and review the diff."
        )

    return _assert


@pytest.fixture
def brighton_problem():
    problem = lokigi.site.SiteProblem()

    problem.add_demand(
        "sample_data/brighton_demand.csv",
        demand_col="demand",
        location_id_col="LSOA",
    )

    problem.add_sites("sample_data/brighton_sites.geojson", candidate_id_col="site")

    problem.add_travel_matrix(
        travel_matrix_df="sample_data/brighton_travel_matrix_driving.csv",
        source_col="LSOA",
        from_unit="seconds",
        to_unit="minutes",
    )
    return problem


@pytest.fixture
def basic_problem():
    return lokigi.site.SiteProblem(debug_mode=False)


@pytest.fixture
def demand_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 200, 150],
        }
    )


@pytest.fixture
def low_demand_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [5, 3, 2],
        }
    )


@pytest.fixture
def demand_gdf():
    return geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 200, 150],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:27700",
    )


@pytest.fixture
def candidate_df():
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
        }
    )


@pytest.fixture
def candidate_gdf():
    return geopandas.GeoDataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "capacity": [500, 300, 400],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:27700",
    )


@pytest.fixture
def candidate_df_with_cost():
    """Same sites/coordinates as `candidate_df`, plus a deliberately
    non-generically-named cost column ("build_cost", not "cost") to exercise
    flexible column naming in add_sites(cost_col=...)."""
    return pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.5, 51.6, 51.7],
            "long": [-0.1, -0.2, -0.3],
            "build_cost": [10.0, 500.0, 50.0],
        }
    )


@pytest.fixture
def travel_df():
    """Rows = demand locations, columns = candidate sites."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 20.0, 30.0],
            "Site_B": [25.0, 5.0, 15.0],
            "Site_C": [30.0, 10.0, 8.0],
        }
    )


@pytest.fixture
def loaded_problem(basic_problem, demand_df, candidate_df, travel_df):
    """A SiteProblem with all three data sources added."""
    basic_problem.add_demand(
        demand_df, demand_col="demand", location_id_col="location_id"
    )
    basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
    basic_problem.add_travel_matrix(travel_df, source_col="source_id")
    return basic_problem


@pytest.fixture
def loaded_problem_with_cost(basic_problem, demand_df, candidate_df_with_cost, travel_df):
    """Same as `loaded_problem`, but sites carry a "build_cost" column
    (site_id -> cost: Site_A=10.0, Site_B=500.0, Site_C=50.0), so total_cost
    per p=2 combination is trivially hand-computable:
      {Site_A, Site_B}: 510.0
      {Site_A, Site_C}: 60.0
      {Site_B, Site_C}: 550.0
    """
    basic_problem.add_demand(
        demand_df, demand_col="demand", location_id_col="location_id"
    )
    basic_problem.add_sites(
        candidate_df_with_cost, candidate_id_col="site_id", cost_col="build_cost"
    )
    basic_problem.add_travel_matrix(travel_df, source_col="source_id")
    return basic_problem


@pytest.fixture
def cost_flips_winner_problem():
    """
    A minimal 2-site/1-demand-point problem (p=1) built so that the
    travel-optimal site is drastically the most expensive, and the
    slightly-worse-travel site is far cheaper:

      Site_Fast: travel=10, build_cost=1000.0
      Site_Cheap: travel=20, build_cost=10.0

    With a single demand point, weighted_average always equals the selected
    site's travel time regardless of the demand weight, so with no "cost"
    weight (or weights=None) the winner is always Site_Fast (10 < 20) --
    identical to what an equivalent problem with no cost_col configured
    would pick. Once a strong "cost" weight is supplied (e.g.
    weights={"demand": 0.1, "cost": 0.9}), batch-relative cost normalization
    flips the winner to Site_Cheap. Deterministic across brute-force,
    greedy, and GRASP (the RCL/local-search batches only ever contain these
    two candidates, so there is no seed-dependent ambiguity).
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1"],
            "demand": [100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_Fast", "Site_Cheap"],
            "lat": [51.1, 51.2],
            "long": [-0.1, -0.2],
            "build_cost": [1000.0, 10.0],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1"],
            "Site_Fast": [10.0],
            "Site_Cheap": [20.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", cost_col="build_cost")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def loaded_problem_low_demand(basic_problem, low_demand_df, candidate_df, travel_df):
    """A SiteProblem with all three data sources added."""
    basic_problem.add_demand(
        low_demand_df, demand_col="demand", location_id_col="location_id"
    )
    basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
    basic_problem.add_travel_matrix(travel_df, source_col="source_id")
    return basic_problem


@pytest.fixture
def manager():
    class DummyManager:
        """A minimal class to test the decorator."""

        def add_data(
            self, candidate_df, id_col="site_id", lat_col="lat", long_col="long"
        ):
            # If it reaches here, validation passed!
            lokigi.utils._validate_columns(
                candidate_df,
                col_names=[id_col, lat_col, long_col],
                numeric_col_names=[lat_col, long_col],
            )
            self.success = True
            return True

    return DummyManager()


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {
            "site_id": ["Site A", "Site B"],
            "lat": [51.5074, 53.4808],
            "long": [-0.1278, -2.2426],
        }
    )


@pytest.fixture
def invalid_df():
    return pd.DataFrame(
        {
            "site_id": ["Site A", "Site B"],
            "lat": [51.5074, 53.4808],
            "long": ["-0.1278", -2.2426],
        }
    )


@pytest.fixture
def five_site_problem():
    """
    A 5-site/4-demand-point problem (p=2, 10 possible combinations) whose
    travel times are deliberately adversarial: the combination with the best
    (lowest) weighted_average is NOT the combination with the best mclp
    coverage. This separation is what makes it possible to detect greedy/
    grasp search bugs that silently fall back to ranking by weighted_average
    instead of the true objective.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"],
            "lat": [51.1, 51.2, 51.3, 51.4, 51.5],
            "long": [-0.1, -0.2, -0.3, -0.4, -0.5],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [38.0, 18.0, 10.0, 28.0],
            "Site_2": [25.0, 40.0, 11.0, 31.0],
            "Site_3": [24.0, 13.0, 29.0, 13.0],
            "Site_4": [29.0, 16.0, 17.0, 16.0],
            "Site_5": [17.0, 15.0, 17.0, 36.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def tied_score_problem():
    """
    A 3-site/2-demand-point problem where Site_A and Site_B produce an
    exactly identical weighted_average (both 15.0) when picked alone, while
    Site_C is clearly better (10.0). Used to trigger the brute-force
    keep_best_n/keep_worst_n heap tie-break crash: pushing two equal-score
    (score, metrics) tuples onto the same heapq forces a fallback comparison
    of the `metrics` dicts, which aren't orderable.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2"],
            "demand": [100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_A": [10.0, 20.0],
            "Site_B": [10.0, 20.0],
            "Site_C": [5.0, 15.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def many_tied_sites_problem():
    """
    30 candidate sites drawn from a handful of exact travel-time values
    (3.0/5.0/8.0/12.0, each shared by several sites), so p=1 brute force
    produces both exact ties AND real heap churn (evictions) as better
    values are found. Used to stress-test parallel brute force (n_jobs>1)
    keep_best_n: with 30 combinations split across several chunks, a merge
    whose admission decision compares score alone (ignoring the tie-break
    index) can keep a different combination on a boundary tie than a
    single-process run would -- this fixture's specific pattern is a known
    repro for that (see test_brute_force_n_jobs_keep_best_n_matches_serial_
    under_exact_ties).
    """
    n_sites = 30
    site_ids = [f"Site_{i}" for i in range(n_sites)]
    pool = [3.0, 8.0, 3.0, 8.0, 5.0, 5.0, 8.0, 8.0, 8.0, 3.0,
            3.0, 5.0, 8.0, 5.0, 12.0, 5.0, 8.0, 5.0, 5.0, 5.0,
            3.0, 5.0, 12.0, 3.0, 5.0, 8.0, 3.0, 3.0, 12.0, 5.0]
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1"],
            "demand": [100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": site_ids,
            "lat": [51.0 + 0.01 * i for i in range(n_sites)],
            "long": [-0.01 * (i + 1) for i in range(n_sites)],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1"],
            **{site_id: [value] for site_id, value in zip(site_ids, pool)},
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def tied_score_problem_with_cost():
    """
    Same shape as `tied_score_problem` (Site_A and Site_B tie exactly on
    weighted_average, both 15.0, with Site_C clearly better at 10.0), but
    with a `build_cost` column added where Site_A and Site_B also share an
    identical cost. That keeps them tied on `composite_score` even after
    cost weighting is blended in, so the cost-weighted keep_best_n/
    keep_worst_n path (pandas `nsmallest`/`nlargest`) can be checked for
    the same exact-tie scenario that used to crash the heap-based path.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2"],
            "demand": [100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
            "build_cost": [5.0, 5.0, 1.0],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2"],
            "Site_A": [10.0, 20.0],
            "Site_B": [10.0, 20.0],
            "Site_C": [5.0, 15.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", cost_col="build_cost")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def cost_weighted_pruning_problem():
    """
    A 4-site/1-demand-point problem (p=1) built so that the combination
    with the true cost-weighted-best score has the WORST raw (pre-cost)
    weighted_average, dead last of 4:

      Site_Fast1: travel=5,  build_cost=1000.0
      Site_Fast2: travel=6,  build_cost=1000.0
      Site_Mid:   travel=12, build_cost=1000.0
      Site_Cheap: travel=15, build_cost=1.0    <- worst raw ranking, cheapest

    With weights={"demand": 0.05, "cost": 0.95}, Site_Cheap is the true
    winner once cost dominates -- but a brute_force_keep_best_n heap that
    prunes on the raw ranking value (before cost is blended in) would
    discard Site_Cheap immediately, since it ranks last of 4 on
    weighted_average alone.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["L1"],
            "demand": [100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_Fast1", "Site_Fast2", "Site_Mid", "Site_Cheap"],
            "lat": [51.1, 51.2, 51.3, 51.4],
            "long": [-0.1, -0.2, -0.3, -0.4],
            "build_cost": [1000.0, 1000.0, 1000.0, 1.0],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["L1"],
            "Site_Fast1": [5.0],
            "Site_Fast2": [6.0],
            "Site_Mid": [12.0],
            "Site_Cheap": [15.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", cost_col="build_cost")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def cost_weighted_five_site_problem():
    """
    Like `five_site_problem` (5 sites, 4 demand points, p=2, 10 possible
    combinations), but with a `build_cost` column added whose cheapest
    sites are not the sites with the best raw travel times -- so
    cost-weighted brute-force search is a genuinely different ranking than
    the unweighted one, giving a non-trivial adversarial case for
    comparing a pruned (brute_force_keep_best_n) run against a full run.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3", "Site_4", "Site_5"],
            "lat": [51.1, 51.2, 51.3, 51.4, 51.5],
            "long": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "build_cost": [50.0, 10.0, 30.0, 5.0, 100.0],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [38.0, 18.0, 10.0, 28.0],
            "Site_2": [25.0, 40.0, 11.0, 31.0],
            "Site_3": [24.0, 13.0, 29.0, 13.0],
            "Site_4": [29.0, 16.0, 17.0, 16.0],
            "Site_5": [17.0, 15.0, 17.0, 36.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id", cost_col="build_cost")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def hybrid_p_median_problem():
    """
    A 3-site/4-demand-point problem (p=2, 3 possible combinations) built so
    that the unconstrained p_median optimum has a bad worst-case travel time:
    {Site_1, Site_3} has the lowest weighted_average (5.5) but its worst
    demand point (LSOA_4) is 16 minutes away. {Site_1, Site_2} has a worse
    average (7.0) but caps every demand point at 10 minutes.

    This lets max_value_cutoff (hybrid_p_median's "safety net" constraint)
    exclude the unconstrained-best combination and force a different,
    independently verifiable winner.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [2, 16, 10, 16],
            "Site_2": [20, 8, 20, 8],
            "Site_3": [16, 2, 2, 16],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


def _demand_sensitivity_candidate_and_travel_dfs():
    """Shared 3-site/3-demand-point travel matrix for the p_median vs.
    simple_p_median demand-sensitivity fixtures below -- only the demand
    column differs between them."""
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B", "Site_C"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [20, 4, 6],
            "Site_B": [8, 10, 8],
            "Site_C": [20, 20, 3],
        }
    )
    return candidate_df, travel_df


@pytest.fixture
def demand_sensitivity_skewed_problem():
    """
    Demand is heavily skewed towards LSOA_3 (40 vs. 10 and 10), so the
    demand-weighted (p_median) and plain (simple_p_median) rankings of the
    three p=2 combinations disagree on which is best:

      {Site_A, Site_B}: mins [8, 4, 6]  -> weighted 6.0, unweighted 6.0
      {Site_A, Site_C}: mins [20, 4, 3] -> weighted 6.0, unweighted 9.0
      {Site_B, Site_C}: mins [8, 10, 3] -> weighted 5.0, unweighted 7.0

    p_median (lowest weighted_average) picks {Site_B, Site_C} (5.0), while
    simple_p_median (lowest unweighted_average) picks {Site_A, Site_B} (6.0)
    -- a different combination.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [10, 10, 40],
        }
    )
    candidate_df, travel_df = _demand_sensitivity_candidate_and_travel_dfs()

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def demand_sensitivity_uniform_problem():
    """Same site/travel data as `demand_sensitivity_skewed_problem`, but with
    identical demand at every location -- so weighted_average and
    unweighted_average are mathematically identical for every combination,
    and p_median/simple_p_median must agree on the best one:
    {Site_A, Site_B}, at 6.0."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 100, 100],
        }
    )
    candidate_df, travel_df = _demand_sensitivity_candidate_and_travel_dfs()

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def equity_df():
    """Three distinct equity bins (1, 5, 10) -- one per demand location --
    so tertile-ratio calculations (which need >= 3 unique bins) are exercised."""
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "imd_decile": [1, 5, 10],
        }
    )


@pytest.fixture
def loaded_problem_with_equity(loaded_problem, equity_df):
    """`loaded_problem` with equity data wired in via add_equity_data().
    IMD deciles: 1 = most deprived, so the disadvantaged end is low."""
    loaded_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        disadvantaged_end="low",
    )
    return loaded_problem


@pytest.fixture
def secondary_travel_df():
    """Rows = demand locations, columns = candidate sites -- deliberately
    reordered relative to `travel_df` (Site_C, Site_B, Site_A instead of
    Site_A, Site_B, Site_C) and with unrelated cost values, so a secondary
    travel matrix implementation that accidentally reuses the primary
    matrix's positional column indices produces visibly wrong numbers
    rather than plausible-looking ones. Cost is uniform per site across all
    demand locations (Site_C=5 fastest, Site_B=50, Site_A=90 slowest) --
    the reverse ranking of `travel_df`, where Site_B is the p=1
    weighted-average winner on the primary matrix."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_C": [5.0, 5.0, 5.0],
            "Site_B": [50.0, 50.0, 50.0],
            "Site_A": [90.0, 90.0, 90.0],
        }
    )


@pytest.fixture
def loaded_problem_with_secondary_matrix(loaded_problem, secondary_travel_df):
    """`loaded_problem` plus a registered secondary travel matrix labelled
    'public_transport' (see `secondary_travel_df`)."""
    loaded_problem.add_secondary_travel_matrix(
        secondary_travel_df, source_col="source_id", label="public_transport"
    )
    return loaded_problem


@pytest.fixture
def loaded_problem_with_equity_and_secondary_matrix(
    loaded_problem_with_equity, secondary_travel_df
):
    """`loaded_problem_with_equity` plus the same registered secondary
    travel matrix as `loaded_problem_with_secondary_matrix`."""
    loaded_problem_with_equity.add_secondary_travel_matrix(
        secondary_travel_df, source_col="source_id", label="public_transport"
    )
    return loaded_problem_with_equity


@pytest.fixture
def additional_data_df():
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "co2_emissions": [12.0, 8.0, 20.0],
        }
    )


@pytest.fixture
def loaded_problem_with_additional_data(loaded_problem, additional_data_df):
    """`loaded_problem` with a registered additional dataset (label "co2"),
    for testing the compound-weights direction-resolution branch when the
    label DOES have matching metadata (contrast case for the unmatched-label
    UnboundLocalError bug)."""
    loaded_problem.add_additional_data(
        additional_data_df,
        column_of_interest="co2_emissions",
        common_col="location_id",
        label="co2",
        direction="lower_better",
    )
    return loaded_problem


@pytest.fixture
def sfca_problem():
    """
    A 2SFCA fixture built so classic threshold coverage and 2SFCA
    disagree, plus a deliberately isolated site and a deliberately
    unreachable region for the empty-catchment / zero-accessibility edge
    cases. `catchment_size=15` throughout (see test_two_step_floating_
    catchment.py).

    Travel matrix (rows=LSOA, cols=Site_1/Site_2/Site_Isolated):
      LSOA_1:        Site_1=10,   Site_2=12,   Site_Isolated=1000
      LSOA_2:        Site_1=8,    Site_2=30,   Site_Isolated=1000
      LSOA_3:        Site_1=30,   Site_2=5,    Site_Isolated=1000
      LSOA_Isolated: Site_1=1000, Site_2=1000, Site_Isolated=1000

    Demand: LSOA_1=100, LSOA_2=100, LSOA_3=50, LSOA_Isolated=75.
    Supply: Site_1=10, Site_2=5, Site_Isolated=8.

    With site_names=["Site_1", "Site_2"] (Site_Isolated excluded):
    LSOA_1 is within catchment_size=15 of BOTH sites (10, 12); LSOA_2 only
    of Site_1 (8 < 15 < 30); LSOA_3 only of Site_2 (5 < 15 < 30).
    Binary coverage rates LSOA_1 and LSOA_2 identically ("covered", since
    both have min_cost < 15) despite LSOA_2 having to share Site_1's
    supply with LSOA_1 and having no second option -- exactly what 2SFCA
    is meant to separate out:

      Step 1: catchment_demand(Site_1) = 100 + 100 = 200 -> R_1 = 10/200 = 0.05
              catchment_demand(Site_2) = 100 + 50 = 150  -> R_2 = 5/150 = 1/30
      Step 2: accessibility(LSOA_1) = R_1 + R_2 = 0.05 + 1/30 = 0.08333...
              accessibility(LSOA_2) = R_1         = 0.05
              accessibility(LSOA_3) = R_2         = 1/30 = 0.03333...
              accessibility(LSOA_Isolated) = 0 (no site in catchment --
              a real zero, not a missing value)

    Conservation check (only holds when every scored site has a non-empty
    catchment, i.e. Site_Isolated excluded): sum(demand * accessibility)
    == 100*(0.05 + 1/30) + 100*0.05 + 50*(1/30) + 75*0 == 15.0 ==
    sum(supply) == 10 + 5.

    Site_Isolated is unreachable from (and to) every region at
    catchment_size=15, so its catchment_demand is 0 and its ratio is
    undefined (NaN) -- the empty-catchment edge case, only triggered when
    it is included in site_names/site_indices.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_Isolated"],
            "demand": [100, 100, 50, 75],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_Isolated"],
            "lat": [51.1, 51.2, 51.3],
            "long": [-0.1, -0.2, -0.3],
            "supply": [10, 5, 8],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_Isolated"],
            "Site_1": [10.0, 8.0, 30.0, 1000.0],
            "Site_2": [12.0, 30.0, 5.0, 1000.0],
            "Site_Isolated": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem


@pytest.fixture
def sfca_problem_with_geometry(sfca_problem):
    """`sfca_problem` plus a region geometry layer -- one small square per
    demand region -- for `plot_accessibility()` tests. Geometry doesn't
    need to be contiguous or realistic here, unlike the hotspot fixtures,
    since plot_accessibility() doesn't build spatial-contiguity weights."""
    regions = geopandas.GeoDataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_Isolated"],
            "geometry": [
                Point(0, 0).buffer(0.4, cap_style=3),
                Point(1, 0).buffer(0.4, cap_style=3),
                Point(0, 1).buffer(0.4, cap_style=3),
                Point(1, 1).buffer(0.4, cap_style=3),
            ],
        },
        crs="EPSG:27700",
    )
    sfca_problem.add_region_geometry_layer(regions, common_col="location_id")
    return sfca_problem


@pytest.fixture
def sfca_secondary_travel_df():
    """A secondary ('public_transport') matrix for `sfca_problem` where
    Site_1 is unreachable (cost 20 > catchment_size=15) for every region
    and Site_2 is reachable by all (cost 5), unlike the primary matrix
    where reachability varies by region. Site_Isolated stays unreachable
    from everywhere, same as on the primary matrix -- every candidate site
    must have a column in every registered secondary matrix, even one this
    fixture's tests only query with site_names=["Site_1", "Site_2"].

    Every one of the four demand regions ends up with the same
    accessibility: catchment_demand(Site_2) = 100+100+50+75 = 325,
    R_2 = 5/325 = 1/65, and Site_1/Site_Isolated contribute nothing
    anywhere."""
    return pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_Isolated"],
            "Site_1": [20.0, 20.0, 20.0, 20.0],
            "Site_2": [5.0, 5.0, 5.0, 5.0],
            "Site_Isolated": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )


@pytest.fixture
def sfca_problem_with_secondary_matrix(sfca_problem, sfca_secondary_travel_df):
    """`sfca_problem` plus the `public_transport` secondary matrix from
    `sfca_secondary_travel_df`."""
    sfca_problem.add_secondary_travel_matrix(
        sfca_secondary_travel_df, source_col="source_id", label="public_transport"
    )
    return sfca_problem


@pytest.fixture
def gaussian_decay_problem():
    """A single-site problem with four regions placed at cost 0, 15, 30 and
    100 from the only site, for `distance_decay={"method": "gaussian",
    "catchment_size": 30, "bandwidth": 15}` tests.

    Dai (2010)'s truncated Gaussian weight is exactly 1.0 at distance 0 and
    exactly 0.0 at distance == catchment_size (both algebraically exact, not
    approximations), and 0.0 again beyond catchment_size (truncation) --
    covering the boundary and truncation cases in one fixture. LSOA_Mid's
    distance of 15 gives a weight strictly between 0 and 1, derivable from
    the formula independently in the test.

    Demand: LSOA_Zero=100, LSOA_Mid=100, LSOA_Boundary=50, LSOA_Beyond=75.
    Supply: Site_1=10.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_Zero", "LSOA_Mid", "LSOA_Boundary", "LSOA_Beyond"],
            "demand": [100, 100, 50, 75],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1"],
            "lat": [51.1],
            "long": [-0.1],
            "supply": [10],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_Zero", "LSOA_Mid", "LSOA_Boundary", "LSOA_Beyond"],
            "Site_1": [0.0, 15.0, 30.0, 100.0],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id")
    return problem
