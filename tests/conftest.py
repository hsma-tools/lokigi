import pytest
import lokigi
import pandas as pd
from shapely.geometry import Point
import geopandas


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
    IMD deciles: 10 = least deprived, so higher is better."""
    loaded_problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        direction="higher_is_better",
    )
    return loaded_problem


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
