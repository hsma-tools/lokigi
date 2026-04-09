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
