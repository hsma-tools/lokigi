"""Tests for registering several equity dimensions via repeated
`add_equity_data()` calls.

Before the registry, `add_equity_data()` wrote into a single set of
attributes, so a second call silently overwrote the first and only one
dimension (deprivation, in practice) could ever be analysed. Real equity
work is rarely one-dimensional -- deprivation *and* rurality *and*
ethnicity is the normal ask -- so dimensions are now registered by label,
with exactly one flagged as primary.

The primary dimension is the only one `solve()` computes `solution_df`
equity columns against, and the only one `weights={"equity": ...}` weights
by. Every registered dimension is merged into `problem_df`, which is what
makes the non-primary ones analysable at all.
"""

import pickle

import pytest
import pandas as pd

import lokigi


@pytest.fixture
def rurality_df():
    """A second equity dimension over the same three demand locations,
    deliberately ordered the opposite way round to `equity_df` so a
    dimension mix-up shows up as a visibly different number rather than a
    plausible one.

    `equity_df` is IMD deciles with `disadvantaged_end="low"`, so LSOA_1
    (decile 1) is the most disadvantaged. This is banded 1-10 with
    `disadvantaged_end="high"`, so LSOA_3 (band 10) is the most
    disadvantaged instead -- LSOA_1 is the most deprived but the least
    rural. Note it takes *both* changes to achieve that: mirroring the band
    values and flipping `disadvantaged_end` would cancel out and leave the
    same region most disadvantaged under both dimensions."""
    return pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "rurality_band": [1, 5, 10],
        }
    )


def _add_rurality(problem, rurality_df, **kwargs):
    problem.add_equity_data(
        rurality_df,
        equity_col="rurality_band",
        common_col="location_id",
        label="rurality",
        disadvantaged_end="high",
        **kwargs,
    )
    return problem


# --- Single-dimension behaviour is unchanged -------------------------------


def test_first_registration_is_primary_without_saying_so(loaded_problem_with_equity):
    problem = loaded_problem_with_equity
    assert problem._primary_equity_label == "equity"
    assert problem.primary_equity["equity_col"] == "imd_decile"
    assert problem.primary_equity["disadvantaged_end"] == "low"


def test_legacy_attributes_still_read_the_primary_dimension(
    loaded_problem_with_equity,
):
    """The five pre-registry attributes are read in ~50 places across the
    solvers, metrics and plots. They must keep resolving to the primary
    dimension so none of those call sites had to change."""
    problem = loaded_problem_with_equity
    assert problem._equity_data_equity_col == "imd_decile"
    assert problem._equity_data_common_col == "location_id"
    assert problem._equity_data_label == "equity"
    assert problem._equity_data_disadvantaged_end == "low"
    assert problem._equity_data_type == "pandas"
    assert list(problem.equity_data.columns) == ["location_id", "imd_decile"]


def test_legacy_attributes_are_none_before_any_registration(loaded_problem):
    assert loaded_problem.equity_data is None
    assert loaded_problem._equity_data_equity_col is None
    assert loaded_problem._equity_data_disadvantaged_end is None
    assert loaded_problem.primary_equity is None


def test_legacy_attributes_are_read_only(loaded_problem_with_equity):
    """Writing one of these used to half-configure a problem -- e.g. a new
    equity_col alongside a stale common_col. The registry makes that
    unrepresentable, so the views reject writes rather than accepting one
    that would be ignored."""
    with pytest.raises(AttributeError):
        loaded_problem_with_equity._equity_data_equity_col = "something_else"


# --- Registering a second dimension ----------------------------------------


def test_second_dimension_registers_alongside_the_first(
    loaded_problem_with_equity, rurality_df
):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)

    assert sorted(problem.equity_datasets) == ["equity", "rurality"]
    # The original stays primary, so nothing about the metrics moves.
    assert problem._primary_equity_label == "equity"
    assert problem._equity_data_equity_col == "imd_decile"


def test_second_dimension_can_take_over_as_primary(
    loaded_problem_with_equity, rurality_df
):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=True)

    assert problem._primary_equity_label == "rurality"
    assert problem._equity_data_equity_col == "rurality_band"
    assert problem._equity_data_disadvantaged_end == "high"
    # ...and the demoted dimension is still registered for analysis.
    assert "equity" in problem.equity_datasets


def test_second_dimension_without_explicit_primary_raises(
    loaded_problem_with_equity, rurality_df
):
    """The load-bearing break. Code written before the registry called
    add_equity_data() twice expecting the second to replace the first;
    registering alongside and silently leaving the first as primary would
    keep that code running while changing which bands its numbers
    describe."""
    with pytest.raises(ValueError, match="needs primary= stated explicitly"):
        _add_rurality(loaded_problem_with_equity, rurality_df)


def test_reusing_a_label_requires_replace(loaded_problem_with_equity, equity_df):
    with pytest.raises(ValueError, match="already\\s+registered"):
        loaded_problem_with_equity.add_equity_data(
            equity_df,
            equity_col="imd_decile",
            common_col="location_id",
            label="equity",
        )


def test_replace_overwrites_in_place_and_keeps_primary_status(
    loaded_problem_with_equity,
):
    problem = loaded_problem_with_equity
    updated = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [3, 6, 9]}
    )
    problem.add_equity_data(
        updated,
        equity_col="imd_decile",
        common_col="location_id",
        label="equity",
        replace=True,
    )

    assert list(problem.equity_datasets) == ["equity"]
    assert problem._primary_equity_label == "equity"
    assert problem.equity_data["imd_decile"].tolist() == [3, 6, 9]


def test_replace_for_an_unregistered_label_raises(loaded_problem, equity_df):
    with pytest.raises(ValueError, match="no equity dimension 'equity' is"):
        loaded_problem.add_equity_data(
            equity_df,
            equity_col="imd_decile",
            common_col="location_id",
            label="equity",
            replace=True,
        )


def test_only_dimension_cannot_be_non_primary(loaded_problem, equity_df):
    with pytest.raises(ValueError, match="cannot be registered with primary=False"):
        loaded_problem.add_equity_data(
            equity_df,
            equity_col="imd_decile",
            common_col="location_id",
            label="equity",
            primary=False,
        )


def test_demoting_the_current_primary_in_place_raises(loaded_problem_with_equity):
    """primary=False on a replace of the *current primary* would leave
    _primary_equity_label pointing at a dimension that no longer claims to
    be primary."""
    updated = pd.DataFrame(
        {"location_id": ["LSOA_1", "LSOA_2", "LSOA_3"], "imd_decile": [3, 6, 9]}
    )
    with pytest.raises(ValueError, match="cannot be re-registered with primary=False"):
        loaded_problem_with_equity.add_equity_data(
            updated,
            equity_col="imd_decile",
            common_col="location_id",
            label="equity",
            replace=True,
            primary=False,
        )


def test_two_dimensions_cannot_share_an_equity_column(
    loaded_problem_with_equity, equity_df
):
    """Both are merged into the same problem_df, so one would overwrite the
    other's bands there -- caught at registration rather than showing up as
    silently wrong band values much later."""
    with pytest.raises(ValueError, match="already uses the column 'imd_decile'"):
        loaded_problem_with_equity.add_equity_data(
            equity_df,
            equity_col="imd_decile",
            common_col="location_id",
            label="a_different_label",
            primary=False,
        )


# --- Resolution ------------------------------------------------------------


def test_resolve_equity_defaults_to_primary(loaded_problem_with_equity, rurality_df):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    assert problem._resolve_equity()["label"] == "equity"
    assert problem._resolve_equity("rurality")["label"] == "rurality"


def test_resolve_unknown_label_raises_and_lists_registered(
    loaded_problem_with_equity, rurality_df
):
    """Falling back to the primary would misreport which bands a chart or
    metric describes, so an unknown label is an error, not a default."""
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    with pytest.raises(ValueError, match=r"Registered: \['equity', 'rurality'\]"):
        problem._resolve_equity("ethnicity")


def test_resolve_returns_none_when_nothing_registered(loaded_problem):
    assert loaded_problem._resolve_equity() is None


def test_show_equity_data_selects_by_label(loaded_problem_with_equity, rurality_df):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    assert "imd_decile" in problem.show_equity_data().columns
    assert "rurality_band" in problem.show_equity_data("rurality").columns


def test_show_equity_dimensions_lists_everything(
    loaded_problem_with_equity, rurality_df
):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    frame = problem.show_equity_dimensions()

    assert frame["label"].tolist() == ["equity", "rurality"]
    assert frame["primary"].tolist() == [True, False]
    assert frame.set_index("label").loc["rurality", "disadvantaged_end"] == "high"


def test_show_equity_dimensions_is_empty_not_none_when_unregistered(loaded_problem):
    frame = loaded_problem.show_equity_dimensions()
    assert frame.empty
    assert "label" in frame.columns


# --- Effect on solving -----------------------------------------------------


def test_every_registered_dimension_reaches_problem_df(
    loaded_problem_with_equity, rurality_df
):
    """The per-region frame is where the `equity=` analysis views read
    their bands from, so a registered dimension missing here could not be
    selected at all."""
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    problem_df = problem.solve(p=1).solution_df.iloc[0]["problem_df"]

    assert "imd_decile" in problem_df.columns
    assert "rurality_band" in problem_df.columns


def test_non_primary_dimension_does_not_change_any_metric(
    loaded_problem_with_equity, rurality_df
):
    """Registering an extra dimension is an analysis affordance, not a
    change to what is optimised or reported -- every solution_df value must
    be identical with and without it."""
    before = loaded_problem_with_equity.copy().solve(p=1).solution_df.iloc[0]

    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    after = problem.solve(p=1).solution_df.iloc[0]

    assert after["inter_tertile_ratio"] == before["inter_tertile_ratio"]
    assert after["weighted_by_equity_group"] == before["weighted_by_equity_group"]
    assert after["gap_absolute_weighted"] == before["gap_absolute_weighted"]


def test_promoting_a_dimension_changes_the_reported_bands(
    loaded_problem_with_equity, rurality_df
):
    """The counterpart to the test above: making rurality primary *should*
    move the numbers, since its bands run the opposite way to the IMD
    deciles."""
    as_imd = loaded_problem_with_equity.copy().solve(p=1).solution_df.iloc[0]

    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=True)
    as_rurality = problem.solve(p=1).solution_df.iloc[0]

    # Winning site's costs are 25 / 5 / 15 for LSOA_1 / 2 / 3, and every
    # band holds exactly one region.
    assert as_imd["weighted_by_equity_group"] == {1: 25.0, 5: 5.0, 10: 15.0}
    assert as_rurality["weighted_by_equity_group"] == {1: 25.0, 5: 5.0, 10: 15.0}

    # Same per-band means, opposite ends: under IMD the most disadvantaged
    # third is decile 1 (cost 25) against decile 10 (cost 15); under
    # rurality it is band 10 (cost 15) against band 1 (cost 25). So the
    # ratios are reciprocals -- 25/15 vs 15/25 -- and the headline flips
    # from inequitable to progressive purely by promoting the other
    # dimension.
    assert as_imd["inter_tertile_ratio"] == pytest.approx(25 / 15)
    assert as_rurality["inter_tertile_ratio"] == pytest.approx(15 / 25)


def test_copy_carries_the_whole_registry(loaded_problem_with_equity, rurality_df):
    problem = _add_rurality(loaded_problem_with_equity, rurality_df, primary=False)
    duplicate = problem.copy()

    assert sorted(duplicate.equity_datasets) == ["equity", "rurality"]
    assert duplicate._primary_equity_label == "equity"
    # Deep, not shared: mutating the copy must not reach the original.
    duplicate.equity_datasets["rurality"]["label"] = "changed"
    assert problem.equity_datasets["rurality"]["label"] == "rurality"


# --- Pre-registry pickles --------------------------------------------------
#
# Solving offline and shipping the pickled result is a normal pattern for
# this package, so problems and solution sets written by <= v0.9.1 exist in
# the wild. Their state carries the flat `_equity_data_*` attributes and no
# registry. Those attributes are now properties, and a property is a data
# descriptor -- it wins over the unpickled `__dict__` -- so without
# migration the load appears to succeed and then raises a bare
# AttributeError naming an internal attribute the user never set.


def _as_legacy_pickle(problem, equity=None):
    """Round-trip `problem` through a pickle rewritten into the pre-registry
    attribute layout, i.e. what v0.9.1 would have written."""
    clone = pickle.loads(pickle.dumps(problem))
    del clone.__dict__["equity_datasets"]
    del clone.__dict__["_primary_equity_label"]
    clone.__dict__.update(
        {
            "equity_data": None,
            "_equity_data_type": None,
            "_equity_data_equity_col": None,
            "_equity_data_common_col": None,
            "_equity_data_label": None,
            "_equity_data_disadvantaged_end": None,
        }
    )
    if equity is not None:
        clone.__dict__.update(equity)
    return pickle.loads(pickle.dumps(clone))


def test_legacy_pickle_migrates_into_the_registry(loaded_problem_with_equity):
    restored = _as_legacy_pickle(
        loaded_problem_with_equity,
        {
            "equity_data": loaded_problem_with_equity.equity_data,
            "_equity_data_type": "pandas",
            "_equity_data_equity_col": "imd_decile",
            "_equity_data_common_col": "location_id",
            "_equity_data_label": "equity",
            "_equity_data_disadvantaged_end": "low",
        },
    )

    assert list(restored.equity_datasets) == ["equity"]
    assert restored._primary_equity_label == "equity"
    assert restored._equity_data_equity_col == "imd_decile"
    assert restored._equity_data_disadvantaged_end == "low"


def test_legacy_pickle_still_solves_with_equity_metrics(loaded_problem_with_equity):
    """The point of migrating rather than just failing politely: the equity
    analysis has to keep working across the upgrade."""
    expected = loaded_problem_with_equity.copy().solve(p=1).solution_df.iloc[0]
    restored = _as_legacy_pickle(
        loaded_problem_with_equity,
        {
            "equity_data": loaded_problem_with_equity.equity_data,
            "_equity_data_type": "pandas",
            "_equity_data_equity_col": "imd_decile",
            "_equity_data_common_col": "location_id",
            "_equity_data_label": "equity",
            "_equity_data_disadvantaged_end": "low",
        },
    )

    row = restored.solve(p=1).solution_df.iloc[0]
    assert row["weighted_by_equity_group"] == expected["weighted_by_equity_group"]
    assert row["inter_tertile_ratio"] == pytest.approx(expected["inter_tertile_ratio"])


def test_legacy_pickle_without_equity_data_restores_empty(loaded_problem):
    restored = _as_legacy_pickle(loaded_problem)
    assert restored.equity_datasets == {}
    assert restored._primary_equity_label is None
    assert restored.equity_data is None
    assert restored.primary_equity is None


def test_legacy_pickle_falls_back_to_column_name_when_unlabelled(
    loaded_problem_with_equity,
):
    """`label` was effectively optional on the old call, but the registry
    keys off it and cannot register a dimension under None."""
    restored = _as_legacy_pickle(
        loaded_problem_with_equity,
        {
            "equity_data": loaded_problem_with_equity.equity_data,
            "_equity_data_type": "pandas",
            "_equity_data_equity_col": "imd_decile",
            "_equity_data_common_col": "location_id",
            "_equity_data_label": None,
            "_equity_data_disadvantaged_end": "low",
        },
    )
    assert list(restored.equity_datasets) == ["imd_decile"]
    assert restored._primary_equity_label == "imd_decile"


def test_current_pickle_round_trips_without_stale_keys(loaded_problem_with_equity):
    restored = pickle.loads(pickle.dumps(loaded_problem_with_equity))
    assert list(restored.equity_datasets) == ["equity"]
    assert restored._equity_data_equity_col == "imd_decile"
    # The flat names must not linger in __dict__, where they would be
    # permanently shadowed by the properties and mislead anyone inspecting
    # the instance.
    stale = [
        key
        for key in restored.__dict__
        if key == "equity_data" or key.startswith("_equity_data_")
    ]
    assert stale == []


# --- add_equity_data() must not mutate the caller's dataframe --------------


def test_continuous_measure_does_not_mutate_the_callers_dataframe():
    """`continuous_measure=True` writes the bins back over `equity_col` and
    adds a `_raw` column. Those writes used to land in the caller's own
    frame, because a DataFrame argument is passed through by reference --
    destroying the scores they handed in."""
    user_df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "imd_score": [12.5, 30.1, 8.2, 44.0],
        }
    )
    problem = lokigi.site.SiteProblem(debug_mode=False)
    problem.add_equity_data(
        user_df,
        equity_col="imd_score",
        common_col="location_id",
        label="imd",
        continuous_measure=True,
        n_bins=4,
        disadvantaged_end="high",
    )

    assert list(user_df.columns) == ["location_id", "imd_score"]
    assert user_df["imd_score"].tolist() == [12.5, 30.1, 8.2, 44.0]
    # ...while the registered copy carries both the bins and the raw values.
    assert problem.equity_data["imd_score"].tolist() == [2, 3, 1, 4]
    assert problem.equity_data["imd_score_raw"].tolist() == [12.5, 30.1, 8.2, 44.0]


def test_same_dataframe_can_be_registered_to_two_problems():
    """The cascade the mutation caused: the second registration re-binned
    already-binned values, so its `_raw` column -- the whole point of which
    is preserving the raw measure -- held bin numbers instead."""
    shared = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3", "LSOA_4"],
            "score": [12.5, 30.1, 8.2, 44.0],
        }
    )
    kwargs = dict(
        equity_col="score",
        common_col="location_id",
        label="s",
        continuous_measure=True,
        n_bins=4,
        disadvantaged_end="high",
    )
    first = lokigi.site.SiteProblem(debug_mode=False)
    first.add_equity_data(shared, **kwargs)
    second = lokigi.site.SiteProblem(debug_mode=False)
    second.add_equity_data(shared, **kwargs)

    assert (
        first.equity_data["score"].tolist() == second.equity_data["score"].tolist()
    )
    assert second.equity_data["score_raw"].tolist() == [12.5, 30.1, 8.2, 44.0]


def test_replace_re_registration_does_not_rebin(loaded_problem):
    """Same cascade via `replace=True`, which the registry makes an obvious
    thing to reach for."""
    df = pd.DataFrame(
        {
            "location_id": ["LSOA_1", "LSOA_2", "LSOA_3"],
            "score": [12.5, 30.1, 8.2],
        }
    )
    kwargs = dict(
        equity_col="score",
        common_col="location_id",
        label="s",
        continuous_measure=True,
        n_bins=3,
        disadvantaged_end="high",
    )
    loaded_problem.add_equity_data(df, **kwargs)
    before = loaded_problem.equity_data["score"].tolist()

    loaded_problem.add_equity_data(df, replace=True, **kwargs)

    assert loaded_problem.equity_data["score"].tolist() == before
    assert loaded_problem.equity_data["score_raw"].tolist() == [12.5, 30.1, 8.2]
