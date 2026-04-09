import lokigi


def test_default_crs():
    prob = lokigi.site.SiteProblem()
    assert prob.preferred_crs == "EPSG:27700", (
        f"Preferred CRS should be EPSG:27700 but is {prob.preferred_crs}"
    )


def test_custom_crs():
    prob = lokigi.site.SiteProblem(preferred_crs="EPSG:4326")
    assert prob.preferred_crs == "EPSG:4326"


def test_debug_mode_sets_verbose():
    prob = lokigi.site.SiteProblem(debug_mode=True)
    assert prob._verbose is True


def test_non_debug_mode_disables_verbose():
    prob = lokigi.site.SiteProblem(debug_mode=False)
    assert prob._verbose is False


def test_initial_data_attributes_are_none():
    prob = lokigi.site.SiteProblem()
    assert prob.demand_data is None
    assert prob.candidate_sites is None
    assert prob.travel_matrix is None
    assert prob.travel_and_demand_df is None
