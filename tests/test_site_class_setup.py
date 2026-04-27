import pytest
import geopandas

# ---------------------------------------------------------------------------
# add_demand
# ---------------------------------------------------------------------------


class TestAddDemand:
    def test_adds_dataframe_successfully(self, basic_problem, demand_df):
        basic_problem.add_demand(
            demand_df, demand_col="demand", location_id_col="location_id"
        )
        assert basic_problem.demand_data is not None
        assert basic_problem._demand_data_demand_col == "demand"
        assert basic_problem._demand_data_id_col == "location_id"

    def test_adds_geodataframe_successfully(self, basic_problem, demand_gdf):
        basic_problem.add_demand(
            demand_gdf, demand_col="demand", location_id_col="location_id"
        )
        assert basic_problem.demand_data is not None
        assert basic_problem._demand_data_type == "geopandas"

    def test_raises_on_missing_demand_col(self, basic_problem, demand_df):
        with pytest.raises(Exception):
            basic_problem.add_demand(
                demand_df, demand_col="nonexistent_col", location_id_col="location_id"
            )

    def test_raises_on_missing_id_col(self, basic_problem, demand_df):
        with pytest.raises(Exception):
            basic_problem.add_demand(
                demand_df, demand_col="demand", location_id_col="nonexistent_col"
            )


# ---------------------------------------------------------------------------
# add_sites
# ---------------------------------------------------------------------------


class TestAddSites:
    def test_adds_dataframe_with_lat_long(self, basic_problem, candidate_df):
        basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
        assert basic_problem.candidate_sites is not None
        assert basic_problem.total_n_sites == 3

    def test_adds_geodataframe(self, basic_problem, candidate_gdf):
        basic_problem.add_sites(candidate_gdf, candidate_id_col="site_id")
        assert basic_problem.candidate_sites is not None
        assert basic_problem._candidate_sites_candidate_id_col == "site_id"

    def test_stores_capacity_col(self, basic_problem, candidate_gdf):
        basic_problem.add_sites(
            candidate_gdf, candidate_id_col="site_id", capacity_col="capacity"
        )
        assert basic_problem._candidate_sites_capacity_col == "capacity"

    def test_raises_on_missing_id_col(self, basic_problem, candidate_df):
        with pytest.raises(Exception):
            basic_problem.add_sites(candidate_df, candidate_id_col="nonexistent_col")

    def test_total_n_sites_correct(self, basic_problem, candidate_df):
        basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
        assert basic_problem.total_n_sites == len(candidate_df)

    def test_output_is_always_geodataframe(self, basic_problem, candidate_df):
        basic_problem.add_sites(candidate_df, candidate_id_col="site_id")
        assert isinstance(basic_problem.candidate_sites, geopandas.GeoDataFrame)


# ---------------------------------------------------------------------------
# add_travel_matrix
# ---------------------------------------------------------------------------


class TestAddTravelMatrix:
    def test_adds_travel_matrix_successfully(self, basic_problem, travel_df):
        basic_problem.add_travel_matrix(travel_df, source_col="source_id")
        assert basic_problem.travel_matrix is not None
        assert basic_problem._travel_matrix_source_col == "source_id"

    def test_raises_on_missing_source_col(self, basic_problem, travel_df):
        with pytest.raises(Exception):
            basic_problem.add_travel_matrix(travel_df, source_col="nonexistent_col")

    def test_unit_conversion_seconds_to_minutes(self, basic_problem, travel_df):
        basic_problem.add_travel_matrix(
            travel_df, source_col="source_id", from_unit="seconds", to_unit="minutes"
        )
        # Original value for Site_A / LSOA_1 was 10 seconds → 10/60 minutes
        result = basic_problem.travel_matrix.loc[
            basic_problem.travel_matrix["source_id"] == "LSOA_1", "Site_A"
        ].values[0]
        assert result == pytest.approx(10 / 60, rel=1e-3)

    def test_unit_conversion_minutes_to_hours(self, basic_problem, travel_df):
        basic_problem.add_travel_matrix(
            travel_df, source_col="source_id", from_unit="minutes", to_unit="hours"
        )
        result = basic_problem.travel_matrix.loc[
            basic_problem.travel_matrix["source_id"] == "LSOA_1", "Site_A"
        ].values[0]
        assert result == pytest.approx(10 / 60, rel=1e-3)

    def test_no_conversion_when_units_not_specified(self, basic_problem, travel_df):
        basic_problem.add_travel_matrix(travel_df, source_col="source_id")
        result = basic_problem.travel_matrix.loc[
            basic_problem.travel_matrix["source_id"] == "LSOA_1", "Site_A"
        ].values[0]
        assert result == 10.0  # unchanged


# ---------------------------------------------------------------------------
# _create_joined_demand_travel_df
# ---------------------------------------------------------------------------


class TestJoinDemandTravel:
    def test_join_succeeds_with_common_ids(self, loaded_problem):
        loaded_problem._create_joined_demand_travel_df(index_col="location_id")
        assert loaded_problem.travel_and_demand_df is not None
        assert len(loaded_problem.travel_and_demand_df) == 3

    def test_join_raises_on_no_common_keys(self, basic_problem, demand_df, travel_df):
        travel_df_bad = travel_df.copy()
        travel_df_bad["source_id"] = ["MISSING_1", "MISSING_2", "MISSING_3"]
        basic_problem.add_demand(
            demand_df, demand_col="demand", location_id_col="location_id"
        )
        basic_problem.add_travel_matrix(travel_df_bad, source_col="source_id")
        with pytest.raises(KeyError):
            basic_problem._create_joined_demand_travel_df(index_col="location_id")
