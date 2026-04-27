import pandas as pd
import numpy as np
import pytest
from lokigi.site import SiteProblem


@pytest.fixture
def simple_problem():

    problem = SiteProblem()

    # Candidate sites
    candidate_sites = pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "lat": [50.62083, 50.68161, 50.53926],
            "long": [-3.40198, -3.23966, -3.61224],
        }
    )

    # Travel matrix (columns = site names)
    travel_and_demand_df = pd.DataFrame(
        {
            "LSOA": ["East Devon 001A", "East Devon 001B", "East Devon 001C"],
            "A": [10, 5, 3],
            "B": [2, 8, 6],
            "C": [7, 1, 4],
        }
    )

    problem.add_sites(
        candidate_sites,
        candidate_id_col="site_id",
        vertical_geometry_col="lat",
        horizontal_geometry_col="long",
    )

    problem.add_travel_matrix(travel_and_demand_df, source_col="LSOA", unit="minutes")

    return problem


####################################
# Simple unit tests
####################################


def test_indices_resolve_to_correct_names(simple_problem):
    problem = simple_problem

    site_indices = [2, 0]

    resolved_names = problem.candidate_sites[
        problem.candidate_sites["canonical_site_index"].isin(site_indices)
    ]["site_id"].tolist()

    assert set(resolved_names) == {"A", "C"}


def test_sorting_keeps_alignment():
    site_indices = [2, 0]  # unsorted
    resolved_names = ["C", "A"]
    matrix_cols = [2, 0]

    combined = list(zip(site_indices, resolved_names, matrix_cols))
    combined.sort(key=lambda x: x[0])

    final_indices = [x[0] for x in combined]
    final_names = [x[1] for x in combined]
    final_cols = [x[2] for x in combined]

    assert final_indices == [0, 2]
    assert final_names == ["A", "C"]
    assert final_cols == [0, 2]


def test_min_cost_and_selected_site(simple_problem):
    problem = simple_problem
    solution = problem.solve(p=2)
    active = solution.site_problem.travel_and_demand_df[["A", "B"]].copy()

    active["min_cost"] = active.min(axis=1)
    active["selected_site"] = active.idxmin(axis=1)

    assert list(active["min_cost"]) == [2, 5, 3]
    assert list(active["selected_site"]) == ["B", "A", "A"]


def test_invalid_index_raises(simple_problem):
    problem = simple_problem
    solution = problem.solve(p=2)

    with pytest.raises(IndexError):
        _ = solution.site_problem.travel_and_demand_df.iloc[:, [10]]


def test_no_empty_combined_when_using_names(simple_problem):
    problem = simple_problem

    resolved_names = ["A", "C"]
    matrix_cols = [0, 2]

    # Simulate fix
    original_indices = [0, 2]

    combined = list(zip(original_indices, resolved_names, matrix_cols))

    assert len(combined) == 2


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
def test_solution_matches_indices(brighton_problem, p):
    solution = brighton_problem.solve(p=p)
    for i, row in solution.show_solutions().iterrows():
        print(f"p = {p}, solution {i}")
        solution_indices = row.site_indices
        solution_names = row.site_names

        # print(solution_indices)
        # print(solution_names)

        sites = solution.site_problem.show_sites()

        resolved_sites = sites[sites["canonical_site_index"].isin(solution_indices)]
        # print(resolved_sites["site"])

        assert set(resolved_sites["site"].to_list()) == set(solution_names), (
            f"Resolved sites don't match canonical index in site dataframe for p={p}"
        )

        # Confirm within the individual problem dataframe that only these sites
        # appear in the solution
        # print(row.problem_df)
        only_selected_sites = list(row.problem_df.selected_site.unique())

        assert set(solution_names) == set(only_selected_sites), (
            f"Sites in detailed solution dataframe don't match sites listed in SolutionSet for p={p}"
        )
        assert set(resolved_sites["site"].to_list()) == set(only_selected_sites), (
            f"Resolved sites don't match canonical index in site dataframe for p={p}"
        )
        assert set(row.problem_df.selected_site).issubset(set(solution_names)), (
            f"Assignments include sites not in the solution or are missing sites for p={p}"
        )

        selected_cols = [col for col in row.problem_df.columns if col in solution_names]

        assert set(selected_cols) == set(solution_names), (
            f"Unexpected additional columns present in active facilities dataframe for p={p}"
        )
