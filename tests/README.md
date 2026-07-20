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
