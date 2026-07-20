# Tests

## Running the suite

From the repo root, with the project's dev environment installed (`uv sync` /
`pip install -e ".[dev]"` or equivalent):

```
pytest
```

`pytest.ini` points `testpaths` at this directory, so `pytest` from the repo
root is equivalent to `pytest tests/`. Useful variants:

```
pytest tests/test_search_strategies.py     # one file
pytest -k mclp                              # anything matching "mclp"
pytest -x                                   # stop at the first failure
pytest -q                                   # quiet output
```

Fixtures shared across test files (sample `SiteProblem`s, demand/candidate/
travel `DataFrame`s, etc.) live in [conftest.py](conftest.py). Most fixtures
build small synthetic problems inline with hand-computable travel times, so
that "known solution" tests can assert against independently derived
numbers. `brighton_problem` is the exception: it loads real data from
`sample_data/brighton_*` (CSV demand, geojson candidate sites, CSV travel
matrix), for coverage of the CSV/geojson loading path.

## Fixture and assertion conventions

Most fixtures in this suite follow three related conventions, worth keeping
up when adding new ones:

**Hand-checkable numbers, not arbitrary ones.** Fixture travel times, demand,
and costs are chosen so the expected result can be derived by hand (or with
a calculator) and asserted as an exact value, rather than just checking
"solve() returns *something* plausible." E.g. equity-weighted fixtures in
[test_equity_weighting_direction.py](test_equity_weighting_direction.py)
assert exact fractions like `145/14`; the cost/travel tradeoff in
[conftest.py](conftest.py)'s `cost_weighted_pruning_problem` is chosen so
the true cost-weighted winner is obvious by inspection (one site cheap and
slow, others fast and expensive). Prefer this over hardcoding whatever a
first correct run happens to output -- an independently-derived number
catches bugs that both the code and a copied-from-output assertion would
agree on.

**Sanity-check that adversarial fixtures actually discriminate.** When a
fixture is built to separate two things that usually agree (e.g.
`five_site_problem` in [conftest.py](conftest.py), where the best
`weighted_average` combination is deliberately *not* the best `mclp`
coverage combination), pair it with a small test that pins the
un-confounded baseline -- e.g.
`test_unconstrained_greedy_walks_into_the_trap` in
[test_max_value_cutoff_strategies.py](test_max_value_cutoff_strategies.py)
confirms unconstrained greedy actually walks into the trap before the
cutoff-respecting tests rely on that trap existing. If the fixture stops
discriminating (e.g. after an unrelated change), this fails loudly instead
of every dependent test silently passing for the wrong reason.

**"Trap" fixtures engineered to catch a specific plausible-but-wrong
implementation.** Rather than a generic random problem, several fixtures
(`greedy_trap_problem`, `cost_weighted_pruning_problem`,
`tied_score_problem`) are shaped so that one specific incorrect behaviour
-- a myopic greedy choice, cost-blind pruning, an unhandled exact tie --
produces a *different, checkable* answer than the correct implementation.
This makes the test a real pin against that exact failure mode, not just a
smoke test that happens to also exercise the code path.

## Backtests (`test_backtests.py`)

Most test files check *correctness* -- they hand-derive the expected answer
for a small problem and assert `solve()` matches it. `test_backtests.py` is
different: it's a regression/snapshot suite that pins `solve()`'s *current*
output for a spread of fixtures and solver configurations (brute-force,
greedy, grasp; p_median, p_center, mclp, hybrid; synthetic and
`brighton_problem`), so that an unintended change to solver internals shows
up as a failing test even when it doesn't happen to break one of the
narrower correctness checks.

Expected values are **not** hardcoded in the test file. They live in
[backtest_snapshots.json](backtest_snapshots.json), keyed by test name, and
are compared via the `assert_backtest` fixture (defined in
[conftest.py](conftest.py)).

### When to regenerate

Only when a change to `solve()`, an objective, or a search strategy is
**intentional** and you've confirmed the new numbers are correct -- e.g.
after a deliberate algorithm change, not as a way to silence a failure you
don't understand. If a backtest fails and you didn't expect solver output to
change, treat it as a real regression and investigate before touching the
snapshot.

### How to regenerate

```
pytest tests/test_backtests.py --update-backtests
```

This recomputes every backtest case and overwrites
`tests/backtest_snapshots.json`. Then:

```
git diff tests/backtest_snapshots.json
```

Review the diff -- it's the entire review surface for "did this change
solver behaviour, and is that expected?". Only commit the updated snapshot
file once the diff looks right.

### Adding a new backtest case

Add a test function to `test_backtests.py` that calls `solve()` (or another
solver entry point) on an existing or new fixture, builds a fingerprint with
`_fingerprint(result, cols=...)`, and passes it to `assert_backtest(...)` --
no expected value needed in the test itself. Then run with
`--update-backtests` once to create its entry in `backtest_snapshots.json`,
and check the generated values look sane before committing.
