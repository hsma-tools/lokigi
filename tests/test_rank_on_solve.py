"""Tests for `solve(rank_on=...)` -- searching on any metric solve()
computes, rather than only the one its `objectives` implies.

Every objective already computes the full metric set for every candidate;
`_get_ranking_by_objective` only decided which column ranked and pruned.
`rank_on` lets the caller name that column instead, so the *search* (not
just a post-hoc re-sort) optimises the measure they actually care about.

Direction is the subtle part. It used to be decided independently in four
places, all spelled `objective == "mclp"`, which works only while every
ranking column is either plainly higher- or plainly lower-is-better. An
inter-tertile ratio is neither: 1.0 means the most- and least-disadvantaged
thirds travel equally far, and both larger and smaller values are a
departure from that. So all four now read one resolved `Metric` and
compare `Metric.normalise()`d values, where lower is always better.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import lokigi
from lokigi.multiobjective import Metric


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def tertile_spread_problem():
    """
    A 3-region/3-site problem, p=1, built so `inter_tertile_ratio` takes a
    value below, at, and above 1.0 depending on which single site is opened.

    The three regions sit in IMD deciles 1/5/10 with `disadvantaged_end=
    "low"`, so with three equity bands each tertile holds exactly one
    region and `inter_tertile_ratio` reduces to
    cost(LSOA_1) / cost(LSOA_3) -- most-disadvantaged over least:

        Site_A: 10 / 20 = 0.5   most-disadvantaged travel *less*
        Site_B: 20 / 20 = 1.0   both travel equally -- parity
        Site_C: 40 / 20 = 2.0   most-disadvantaged travel *twice as far*

    Which makes the three directions pick three different winners, so a
    test can tell them apart:

        lower_better       -> Site_A (0.5)
        higher_better      -> Site_C (2.0)
        closest_to_target  -> Site_B (1.0), neither extreme

    That last one is the case a bare column name cannot express, and the
    reason `rank_on` accepts a `Metric` at all.
    """
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 100, 100],
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
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 15.0, 20.0],
            "Site_B": [20.0, 15.0, 20.0],
            "Site_C": [40.0, 15.0, 20.0],
        }
    )
    equity_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "imd_decile": [1, 5, 10],
        }
    )

    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", unit="minutes")
    problem.add_equity_data(
        equity_df,
        equity_col="imd_decile",
        common_col="location_id",
        label="IMD decile",
        disadvantaged_end="low",
    )
    return problem


def _winner(result):
    return result.solution_df.iloc[0]["site_names"]


# --- The fixture's own premise --------------------------------------------


def test_fixture_produces_the_intended_ratio_spread(tertile_spread_problem):
    """Pin the ratios the direction tests below depend on. If this fails,
    those tests are no longer testing what their names claim."""
    ratios = {}
    for index, name in enumerate(["Site_A", "Site_B", "Site_C"]):
        combination = tertile_spread_problem.evaluate_single_solution_single_objective(
            objective="p_median", site_indices=[index]
        )
        ratios[name] = combination.return_solution_metrics()["inter_tertile_ratio"]

    assert ratios["Site_A"] == pytest.approx(0.5)
    assert ratios["Site_B"] == pytest.approx(1.0)
    assert ratios["Site_C"] == pytest.approx(2.0)


# --- Direction ------------------------------------------------------------


def test_closest_to_target_picks_the_middle_value(tertile_spread_problem):
    """The discriminating case: the best inter-tertile ratio is the one
    nearest parity, which is neither the smallest nor the largest. A
    direction resolved as a single higher/lower boolean cannot express
    this -- it would pick Site_A or Site_C.
    """
    result = tertile_spread_problem.solve(
        p=1,
        rank_on=Metric(
            "inter_tertile_ratio", direction="closest_to_target", target=1.0
        ),
        show_progress=False,
    )
    assert _winner(result) == ["Site_B"]
    assert result.solution_df.iloc[0]["inter_tertile_ratio"] == pytest.approx(1.0)


def test_bare_string_rank_on_uses_lower_is_better_by_default(tertile_spread_problem):
    """`inter_tertile_ratio` matches none of `_is_maximise_metric`'s
    substrings, so a bare string ranks it lower-is-better."""
    result = tertile_spread_problem.solve(
        p=1, rank_on="inter_tertile_ratio", show_progress=False
    )
    assert _winner(result) == ["Site_A"]


def test_explicit_higher_better_reverses_the_bare_string_default(
    tertile_spread_problem,
):
    result = tertile_spread_problem.solve(
        p=1,
        rank_on=Metric("inter_tertile_ratio", direction="higher_better"),
        show_progress=False,
    )
    assert _winner(result) == ["Site_C"]


def test_rank_on_none_is_unchanged_by_this_feature(tertile_spread_problem):
    """The default path must be identical to before `rank_on` existed:
    p_median ranks on weighted_average, so Site_A (15.0) wins over
    Site_B (18.33) and Site_C (25.0)."""
    with_default = tertile_spread_problem.solve(p=1, show_progress=False)
    explicit = tertile_spread_problem.solve(
        p=1, rank_on="weighted_average", show_progress=False
    )
    assert _winner(with_default) == ["Site_A"]
    assert _winner(explicit) == _winner(with_default)


def test_coverage_rank_on_is_higher_is_better(tertile_spread_problem):
    """A coverage column ranked by bare string must sort the right way
    round -- `_is_maximise_metric` matches "within_coverage_threshold".
    At a 25-minute threshold Site_A and Site_B cover all 3 regions and
    Site_C only 2, so the winner must be a full-coverage site, not Site_C.
    """
    result = tertile_spread_problem.solve(
        p=1,
        rank_on="proportion_within_coverage_threshold",
        threshold_for_coverage=25,
        show_progress=False,
    )
    assert result.solution_df.iloc[0]["proportion_within_coverage_threshold"] == 1.0
    assert _winner(result) != ["Site_C"]


# --- Consistency across search strategies ---------------------------------


@pytest.mark.parametrize("search_strategy", ["brute-force", "greedy", "grasp"])
def test_every_search_strategy_agrees_on_the_closest_to_target_winner(
    tertile_spread_problem, search_strategy
):
    """Direction is applied in the brute-force heap, the greedy per-step
    sort, GRASP's construction/local search, and the final cross-strategy
    sort. A flip missed at any one of them shows up here as a strategy
    disagreeing about the best solution.

    `grasp_alpha=0` makes GRASP's restricted candidate list hold only the
    best candidate, so its construction is deterministic and this can
    assert an exact winner rather than a distribution.
    """
    result = tertile_spread_problem.solve(
        p=1,
        rank_on=Metric(
            "inter_tertile_ratio", direction="closest_to_target", target=1.0
        ),
        search_strategy=search_strategy,
        grasp_num_solutions=1,
        grasp_alpha=0.0,
        show_progress=False,
    )
    assert _winner(result) == ["Site_B"]


# --- Streaming/pruning is preserved ---------------------------------------


def test_keep_best_n_prunes_on_the_custom_metric(tertile_spread_problem):
    """`rank_on` is a per-combination transform needing no view of the
    batch, so brute-force's streaming heap keeps working -- unlike
    `weights={"cost": ...}`, which must materialise every combination and
    warns that it is doing so. The top 2 kept under pruning must match the
    top 2 of an unpruned run, and no such warning may be raised.
    """
    metric = Metric("inter_tertile_ratio", direction="closest_to_target", target=1.0)

    unpruned = tertile_spread_problem.solve(
        p=1, rank_on=metric, show_progress=False
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pruned = tertile_spread_problem.solve(
            p=1, rank_on=metric, brute_force_keep_best_n=2, show_progress=False
        )

    assert len(pruned.solution_df) == 2
    assert list(pruned.solution_df["site_names"]) == list(
        unpruned.solution_df["site_names"].head(2)
    )
    assert not [
        str(w.message) for w in caught if "materialis" in str(w.message).lower()
    ]


def test_keep_best_n_of_one_keeps_the_true_closest_to_target_winner(
    tertile_spread_problem,
):
    """The heap scores one combination at a time, so it has its own
    direction handling separate from the DataFrame sorts. Pruning to a
    single solution is what pins it: the ratios are 0.5/1.0/2.0, so a heap
    scoring raw values keeps Site_A (smallest) while one scoring
    normalised distance-from-target keeps Site_B. With keep_best_n=2 both
    orderings happen to retain the same pair and the bug hides.
    """
    result = tertile_spread_problem.solve(
        p=1,
        rank_on=Metric(
            "inter_tertile_ratio", direction="closest_to_target", target=1.0
        ),
        brute_force_keep_best_n=1,
        show_progress=False,
    )
    assert len(result.solution_df) == 1
    assert _winner(result) == ["Site_B"]


def test_rank_score_helper_column_does_not_leak_into_solution_df(
    tertile_spread_problem,
):
    """The normalised column exists only to make the sort ascending; it is
    fully derivable and must not appear in the public results."""
    result = tertile_spread_problem.solve(
        p=1, rank_on="inter_tertile_ratio", show_progress=False
    )
    assert "_rank_score" not in result.solution_df.columns


# --- Composing with an objective's constraints ----------------------------


def test_rank_on_composes_with_a_hybrid_objectives_cutoff(tertile_spread_problem):
    """The reason `rank_on` is a separate parameter rather than a new
    objective value: the objective still supplies the feasibility
    constraint while `rank_on` supplies the ranking. Here that means "cap
    the worst journey at 25 minutes, then pick the most equitable option
    that qualifies" -- Site_C has the parity-closest... no: Site_C's max is
    40, so it is ruled out, and among the survivors Site_B is nearest 1.0.
    """
    result = tertile_spread_problem.solve(
        p=1,
        objectives="hybrid_p_median",
        max_value_cutoff=25,
        rank_on=Metric(
            "inter_tertile_ratio", direction="closest_to_target", target=1.0
        ),
        show_progress=False,
    )
    assert _winner(result) == ["Site_B"]
    assert (result.solution_df["max"] <= 25).all()


# --- The "custom" objective -----------------------------------------------


def test_custom_objective_requires_rank_on(tertile_spread_problem):
    with pytest.raises(ValueError, match="requires rank_on"):
        tertile_spread_problem.solve(p=1, objectives="custom", show_progress=False)


def test_custom_objective_ranks_on_the_given_metric(tertile_spread_problem):
    result = tertile_spread_problem.solve(
        p=1,
        objectives="custom",
        rank_on=Metric(
            "inter_tertile_ratio", direction="closest_to_target", target=1.0
        ),
        show_progress=False,
    )
    assert _winner(result) == ["Site_B"]
    assert result.objectives == "custom"


def test_custom_objective_rejects_max_value_cutoff(tertile_spread_problem):
    """'custom' applies no model constraints, so a cutoff has nowhere to
    live -- use hybrid_p_median + rank_on for that combination instead."""
    with pytest.raises(ValueError, match="doesn't support it"):
        tertile_spread_problem.solve(
            p=1,
            objectives="custom",
            rank_on="inter_tertile_ratio",
            max_value_cutoff=25,
            show_progress=False,
        )


def test_custom_objective_is_described_by_describe_models(capsys, basic_problem):
    basic_problem.describe_models()
    printed = capsys.readouterr().out
    assert "Custom" in printed


# --- Validation, before the search runs -----------------------------------


def test_unknown_rank_on_column_raises_and_lists_alternatives(
    tertile_spread_problem,
):
    with pytest.raises(KeyError) as excinfo:
        tertile_spread_problem.solve(
            p=1, rank_on="not_a_real_metric", show_progress=False
        )
    message = str(excinfo.value)
    assert "not_a_real_metric" in message
    assert "weighted_average" in message


def test_dict_valued_rank_on_column_raises(tertile_spread_problem):
    """Per-equity-band columns hold one value per band, so there is no
    single number to order solutions by."""
    with pytest.raises(ValueError, match="one value per equity band"):
        tertile_spread_problem.solve(
            p=1, rank_on="weighted_by_equity_group", show_progress=False
        )


def test_nan_rank_on_column_raises_naming_the_precondition(loaded_problem):
    """`loaded_problem` has no equity data, so `inter_tertile_ratio` is
    NaN for every combination -- ranking on it would tie every solution
    and produce a meaningless order."""
    with pytest.raises(ValueError, match="add_equity_data"):
        loaded_problem.solve(p=1, rank_on="inter_tertile_ratio", show_progress=False)


def test_coverage_rank_on_without_threshold_raises(tertile_spread_problem):
    with pytest.raises(ValueError, match="threshold_for_coverage"):
        tertile_spread_problem.solve(
            p=1, rank_on="proportion_within_coverage_threshold", show_progress=False
        )


def test_baseline_only_rank_on_without_baseline_raises(tertile_spread_problem):
    """The population-impact family isn't merely NaN without a baseline --
    the columns are absent entirely, so this is the "unknown column" path.
    Its message has to point at `baseline=` anyway, or the reader is left
    thinking they mistyped a column name."""
    with pytest.raises(KeyError, match="baseline"):
        tertile_spread_problem.solve(
            p=1, rank_on="proportion_demand_improved", show_progress=False
        )


def test_validation_runs_before_the_search(tertile_spread_problem, monkeypatch):
    """The whole point of validating up front is not paying for a full
    search first. Fail the solvers outright: a bad rank_on must still
    raise its own error, proving no search was attempted."""

    def explode(*args, **kwargs):
        raise AssertionError("search should not have started")

    monkeypatch.setattr(
        type(tertile_spread_problem), "_brute_force", explode, raising=True
    )
    with pytest.raises(KeyError, match="not_a_real_metric"):
        tertile_spread_problem.solve(
            p=1, rank_on="not_a_real_metric", show_progress=False
        )


def test_non_string_non_metric_rank_on_raises_typeerror(tertile_spread_problem):
    with pytest.raises(TypeError, match="rank_on"):
        tertile_spread_problem.solve(p=1, rank_on=123, show_progress=False)


# --- unreachable_cost interaction -----------------------------------------


@pytest.fixture
def stranding_problem():
    """Site_B is faster for everyone it reaches, but strands LSOA_3
    entirely. Ranking on the reachable-only `max`/`weighted_average` would
    reward it for that; the `_for_ranking` twins exist to prevent it."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "demand": [100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_A", "Site_B"],
            "lat": [51.5, 51.6],
            "long": [-0.1, -0.2],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "Site_A": [10.0, 10.0, 10.0],
            "Site_B": [5.0, 5.0, np.nan],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)
    return problem


def test_rank_on_bare_aggregate_upgrades_to_for_ranking_with_a_warning(
    stranding_problem,
):
    """Honouring a literal `rank_on="max"` under unreachable_cost would
    rank on the reachable-only figure and so reintroduce exactly the
    silent-reward bug that column exists to prevent. Upgrade, but say so."""
    with pytest.warns(UserWarning, match="max_for_ranking"):
        result = stranding_problem.solve(
            p=1, rank_on="max", unreachable_cost=1000, show_progress=False
        )
    assert result.ranking_metric.column == "max_for_ranking"
    assert _winner(result) == ["Site_A"]


def test_rank_on_a_coverage_metric_does_not_require_unreachable_cost(
    stranding_problem,
):
    """The requirement is keyed off the ranking column, not the objective:
    a coverage proportion already counts an unreachable pair as "not
    covered", so it has no silent-reward failure mode to guard against."""
    result = stranding_problem.solve(
        p=1,
        objectives="custom",
        rank_on="proportion_within_coverage_threshold",
        threshold_for_coverage=8,
        show_progress=False,
    )
    assert len(result.solution_df) == 2


def test_rank_on_an_aggregate_still_requires_unreachable_cost(stranding_problem):
    with pytest.raises(NotImplementedError, match="unreachable_cost"):
        stranding_problem.solve(
            p=1, objectives="custom", rank_on="max", show_progress=False
        )


@pytest.fixture
def universally_stranding_problem():
    """4 sites, 4 regions, p=2 -- and LSOA_4 is unreachable from every
    site. So *every* combination strands demand, which makes the
    `*_for_ranking` substitution assertable on whichever solutions GRASP
    happens to return, without depending on its randomised construction
    reaching a particular pair."""
    demand_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "demand": [100, 100, 100, 100],
        }
    )
    candidate_df = pd.DataFrame(
        {
            "site_id": ["Site_1", "Site_2", "Site_3", "Site_4"],
            "lat": [51.1, 51.2, 51.3, 51.4],
            "long": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    travel_df = pd.DataFrame(
        {
            "source_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "Site_1": [10.0, 20.0, 30.0, np.nan],
            "Site_2": [20.0, 10.0, 30.0, np.nan],
            "Site_3": [30.0, 30.0, 10.0, np.nan],
            "Site_4": [40.0, 40.0, 40.0, np.nan],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_demand(demand_df, demand_col="demand", location_id_col="location_id")
    problem.add_sites(candidate_df, candidate_id_col="site_id")
    problem.add_travel_matrix(travel_df, source_col="source_id", allow_missing=True)
    return problem


def test_grasp_returns_metrics_computed_with_unreachable_cost(
    universally_stranding_problem,
):
    """REGRESSION: GRASP's final-accept evaluation used to omit
    `unreachable_cost`, so it *searched* on substituted values but
    *returned* reachable-only ones -- and solve()'s final sort orders the
    returned pool on exactly those `*_for_ranking` columns. Brute-force
    and greedy always passed it.

    Needs more than one solution in the pool: with grasp_num_solutions=1
    the mis-sorted pool has nothing to mis-sort, which is why the existing
    unreachable-cost GRASP test could not catch this.
    """
    result = universally_stranding_problem.solve(
        p=2,
        unreachable_cost=1000,
        search_strategy="grasp",
        grasp_num_solutions=2,
        grasp_max_attempts=50,
        show_progress=False,
    )

    assert len(result.solution_df) >= 1
    for _, row in result.solution_df.iterrows():
        # Every combination strands LSOA_4, so the honest max can never
        # exceed the largest real travel time (40) while the ranking max
        # must be the substituted cost. Before the fix these were equal.
        assert row["max"] <= 40.0
        assert row["max_for_ranking"] == pytest.approx(1000.0)
        assert row["weighted_average_for_ranking"] > row["weighted_average"]
        assert row["weighted_average_for_ranking"] == pytest.approx(
            (row["weighted_average"] * 3 + 1000.0) / 4
        )


def test_grasp_and_brute_force_agree_under_unreachable_cost(
    universally_stranding_problem,
):
    """The consequence of the bug above: the returned pool was ordered on
    unsubstituted values, so GRASP could disagree with brute-force about
    which solution was best."""
    common = dict(
        p=2,
        unreachable_cost=1000,
        show_progress=False,
    )
    brute = universally_stranding_problem.solve(
        search_strategy="brute-force", **common
    )
    grasp = universally_stranding_problem.solve(
        search_strategy="grasp",
        grasp_num_solutions=4,
        grasp_max_attempts=60,
        grasp_alpha=0.0,
        **common,
    )
    assert _winner(grasp) == _winner(brute)


# --- The results object stays honest --------------------------------------


def test_ranking_metric_is_exposed_on_the_solution_set(tertile_spread_problem):
    metric = Metric("inter_tertile_ratio", direction="closest_to_target", target=1.0)
    result = tertile_spread_problem.solve(p=1, rank_on=metric, show_progress=False)
    assert result.ranking_metric.column == "inter_tertile_ratio"
    assert result.ranking_metric.direction == "closest_to_target"


def test_default_solve_reports_the_objectives_own_metric(tertile_spread_problem):
    """With no rank_on, nothing is "custom" and display code must behave
    exactly as it did before this feature."""
    result = tertile_spread_problem.solve(p=1, show_progress=False)
    assert result.ranking_metric.column == "weighted_average"
    assert result._ranks_on_custom_metric() is False
    assert result._ranking_metric_line() is None


def test_custom_rank_on_is_named_for_display(tertile_spread_problem):
    """Plot annotations pick which numbers to show from `self.objectives`,
    which stops describing the ranking under rank_on -- so the real metric
    has to be nameable."""
    result = tertile_spread_problem.solve(
        p=1,
        rank_on=Metric(
            "inter_tertile_ratio",
            direction="closest_to_target",
            target=1.0,
            label="equity ratio",
        ),
        show_progress=False,
    )
    assert result._ranks_on_custom_metric() is True
    line = result._ranking_metric_line(result.solution_df.iloc[0])
    assert "equity ratio" in line
    assert "1.0" in line


def test_unreachable_cost_default_ranking_is_not_treated_as_custom(
    stranding_problem,
):
    """solve() swaps in the `_for_ranking` twin whenever unreachable_cost
    is set. That is still the objective's own metric, so display must not
    start announcing it as a custom ranking."""
    result = stranding_problem.solve(p=1, unreachable_cost=1000, show_progress=False)
    assert result.ranking_metric.column == "weighted_average_for_ranking"
    assert result._ranks_on_custom_metric() is False
